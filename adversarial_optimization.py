# =============================================================================
# Import required libraries
# =============================================================================
import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from criteria.cosine_loss import CosineLoss
from criteria.nce_loss import NCELoss
from criteria.frequency_aware_loss import FrequencyAwareLoss
from attention_control import AttentionControlEdit
from utils import *


@torch.enable_grad()
class Adversarial_Opt:
    def __init__(self, args, model):
        self.device = args.device
        self.dataloader = args.dataloader
        #
        self.diff_model = model
        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)
        #
        self.source_dir = args.source_dir
        self.protected_image_dir = args.protected_image_dir
        self.comparison_null_text = args.comparison_null_text
        #
        self.target_choice = args.target_choice
        #
        self.is_makeup = args.is_makeup
        self.source_text = args.source_text
        self.makeup_prompt = args.makeup_prompt
        self.augment = transforms.RandomPerspective(fill=0, p=1,
                                                    distortion_scale=0.5)
        #
        self.MTCNN_cropping = args.MTCNN_cropping
        #
        self.is_obfuscation = args.is_obfuscation
        #
        self.image_size = args.image_size
        self.prot_steps = args.prot_steps
        self.diffusion_steps = args.diffusion_steps
        self.start_step = args.start_step
        self.null_optimization_steps = args.null_optimization_steps
        #
        self.adv_optim_weight = args.adv_optim_weight
        self.makeup_weight = args.makeup_weight
        #
        # Frequency-aware adversarial perturbation parameters
        self.enable_freq_adv = getattr(args, 'enable_freq_adv', False)
        self.freq_threshold = getattr(args, 'freq_threshold', 0.1)
        self.freq_reg_weight = getattr(args, 'freq_reg_weight', 0.01)
        self.freq_perturbation_strength = getattr(args, 'freq_perturbation_strength', 1.0)
        self.freq_adv_weight = getattr(args, 'freq_adv_weight', 0.003)  # Weight for frequency-enhanced adversarial loss
        #
        # Face quality enhancement conditioning parameters
        self.use_quality_conditioning = getattr(args, 'use_quality_conditioning', True)
        self.face_conditioning_strategy = getattr(args, 'face_conditioning_strategy', 'balanced')
        self.enable_progressive_conditioning = getattr(args, 'enable_progressive_conditioning', False)
        self.progressive_conditioning_mode = getattr(args, 'progressive_conditioning_mode', 'quality_focused')
        #
        self.augment = transforms.RandomPerspective(
            fill=0, p=1, distortion_scale=0.5)
        # Set up loss functions
        self.cosine_loss = CosineLoss(self.is_obfuscation)
        self.nce_loss = NCELoss(self.device, clip_model="ViT-B/32")
        # Set up frequency-aware loss if enabled
        if self.enable_freq_adv:
            self.freq_aware_loss = FrequencyAwareLoss(
                freq_threshold=self.freq_threshold,
                reg_weight=self.freq_reg_weight,
                is_obfuscation=self.is_obfuscation
            )
        # set up FR models
        self.surrogate_models = load_FR_models(
            args, args.surrogate_model_names)
        self.test_model_name = args.test_model_name

    def get_FR_embeddings(self, image):
        features = []
        for model_name in self.surrogate_models.keys():
            input_size = self.surrogate_models[model_name][0]
            fr_model = self.surrogate_models[model_name][1]
            emb_source = fr_model(F.interpolate(
                image, size=input_size, mode='bilinear'))
            features.append(emb_source)
        return features

    def set_attention_control(self, controller):
        def ca_forward(self, place_in_unet):

            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                attn = controller(attn, is_cross, place_in_unet)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")

        controller.num_att_layers = cross_att_count

    def reset_attention_control(self):
        def ca_forward(self):
            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_)
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    register_recr(net__)

        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                register_recr(net[1])
            elif "up" in net[0]:
                register_recr(net[1])
            elif "mid" in net[0]:
                register_recr(net[1])

    def diffusion_step(self, latent, null_context, t,
                       is_null_optimization=False, step_idx=None, total_steps=None, image_tensor=None):
        # Update conditioning if progressive conditioning is enabled
        if (hasattr(self, 'enable_progressive_conditioning') and
            self.enable_progressive_conditioning and
            step_idx is not None and total_steps is not None and
            hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning):

            # Get progressive conditioning for current step
            progressive_context = self.get_progressive_quality_conditioning(
                step_idx, total_steps, image_tensor)

            # Ensure correct batch dimension for progressive context
            batch_size = latent.shape[0]
            if progressive_context.shape[0] != batch_size:
                progressive_context = progressive_context.repeat(batch_size, 1, 1)

            null_context = progressive_context

        # Ensure null_context has correct batch dimension
        if null_context is not None:
            batch_size = latent.shape[0]
            if not is_null_optimization:
                # For CFG, we need 2 * batch_size
                required_batch_size = batch_size * 2
                if null_context.shape[0] == batch_size:
                    null_context = torch.cat([null_context] * 2)
            else:
                required_batch_size = batch_size
                if null_context.shape[0] != required_batch_size:
                    null_context = null_context.repeat(required_batch_size, 1, 1)

        if not is_null_optimization:
            latent_input = torch.cat([latent] * 2)
            noise_pred = self.diff_model.unet(
                latent_input, t, encoder_hidden_states=null_context)["sample"]
            noise_pred, _ = noise_pred.chunk(2)
        else:
            noise_pred = self.diff_model.unet(
                latent, t, encoder_hidden_states=null_context)["sample"]
        return self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]

    def null_text_embeddings(self, step=None, total_steps=None, image_tensor=None):
        """Generate quality-enhanced conditional embeddings for facial images"""
        # Use face-specific quality enhancement prompts instead of empty text
        if hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning:
            # Use progressive conditioning if enabled and step info is available
            if (hasattr(self, 'enable_progressive_conditioning') and
                self.enable_progressive_conditioning and
                step is not None and total_steps is not None):
                return self.get_progressive_quality_conditioning(step, total_steps, image_tensor)
            else:
                return self.get_face_quality_embeddings()
        else:
            # Original empty text embeddings
            uncond_input = self.diff_model.tokenizer([""],
                                                     padding="max_length",
                                                     max_length=self.diff_model.tokenizer.model_max_length,
                                                     return_tensors="pt")
            return self.diff_model.text_encoder(uncond_input.input_ids.to(self.device))[0]

    def get_face_quality_embeddings(self):
        """Generate face-specific quality enhancement embeddings"""
        # Carefully crafted prompts for facial image quality enhancement
        face_quality_prompts = [
            "high quality professional portrait photograph",
            "detailed facial features with natural skin texture",
            "clear sharp focus on face with professional lighting",
            "photorealistic human face with fine details",
            "studio quality headshot with natural appearance"
        ]

        # Select prompt based on conditioning strategy
        conditioning_strategy = getattr(self, 'face_conditioning_strategy', 'balanced')

        if conditioning_strategy == 'detail':
            prompt = "highly detailed facial features, natural skin texture, clear eyes and lips, professional portrait photography"
        elif conditioning_strategy == 'quality':
            prompt = "high quality, sharp focus, professional lighting, photorealistic portrait, 8k resolution"
        elif conditioning_strategy == 'natural':
            prompt = "natural facial appearance, realistic skin tone, soft lighting, authentic human portrait"
        elif conditioning_strategy == 'professional':
            prompt = "professional headshot photography, studio lighting, high definition, masterpiece quality"
        else:  # balanced
            prompt = "high quality detailed portrait photograph, natural facial features, professional lighting"

        # Tokenize and encode the prompt
        text_input = self.diff_model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.diff_model.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return embeddings

    def get_progressive_quality_conditioning(self, step, total_steps, image_tensor=None):
        """Generate progressive quality conditioning based on diffusion step"""

        # Calculate progress ratio (0.0 = start, 1.0 = end)
        progress = step / total_steps if total_steps > 0 else 0.0

        # Define progressive conditioning modes
        if self.progressive_conditioning_mode == 'quality_focused':
            return self._get_quality_focused_progressive_conditioning(progress, image_tensor)
        elif self.progressive_conditioning_mode == 'structure_preserving':
            return self._get_structure_preserving_progressive_conditioning(progress, image_tensor)
        elif self.progressive_conditioning_mode == 'detail_enhancing':
            return self._get_detail_enhancing_progressive_conditioning(progress, image_tensor)
        else:  # balanced
            return self._get_balanced_progressive_conditioning(progress, image_tensor)

    def _get_quality_focused_progressive_conditioning(self, progress, image_tensor=None):
        """Quality-focused progressive conditioning for PSNR/SSIM/FID optimization"""

        if progress < 0.2:  # Early stage (80-100% noise): Global structure and composition
            base_prompt = "well-structured facial composition, balanced proportions, professional portrait setup"
            quality_terms = ", high quality foundation, proper facial geometry"

        elif progress < 0.5:  # Mid-early stage (50-80% noise): Feature definition
            base_prompt = "clear facial features definition, natural skin tone, proper lighting distribution"
            quality_terms = ", enhanced clarity, detailed facial structure"

        elif progress < 0.8:  # Mid-late stage (20-50% noise): Detail refinement
            base_prompt = "high resolution facial details, natural skin texture, clear eyes and lips"
            quality_terms = ", sharp focus, photorealistic details, professional quality"

        else:  # Final stage (0-20% noise): Quality perfection
            base_prompt = "ultra sharp photorealistic portrait, masterpiece quality, exceptional detail preservation"
            quality_terms = ", 8k resolution, crystal clear, professional photography excellence"

        # Add adaptive terms based on image characteristics
        if image_tensor is not None:
            adaptive_terms = self._get_adaptive_quality_terms(image_tensor, progress)
            full_prompt = base_prompt + quality_terms + adaptive_terms
        else:
            full_prompt = base_prompt + quality_terms

        return self._encode_progressive_prompt(full_prompt)

    def _get_structure_preserving_progressive_conditioning(self, progress, image_tensor=None):
        """Structure-preserving progressive conditioning for SSIM optimization"""

        if progress < 0.3:  # Early: Preserve overall facial structure
            prompt = "maintain facial structure integrity, preserve original proportions, stable composition"

        elif progress < 0.6:  # Mid: Enhance structural details
            prompt = "enhance facial feature definition while preserving structure, natural facial geometry"

        else:  # Late: Refine structural details
            prompt = "refine facial details with structural consistency, maintain authentic facial proportions"

        return self._encode_progressive_prompt(prompt)

    def _get_detail_enhancing_progressive_conditioning(self, progress, image_tensor=None):
        """Detail-enhancing progressive conditioning for texture and fine details"""

        if progress < 0.25:  # Early: Basic detail foundation
            prompt = "establish fine detail foundation, prepare for texture enhancement"

        elif progress < 0.6:  # Mid: Active detail enhancement
            prompt = "enhance skin texture details, improve facial feature clarity, add fine details"

        else:  # Late: Detail perfection
            prompt = "perfect fine details, ultra-detailed skin texture, exceptional facial detail quality"

        return self._encode_progressive_prompt(prompt)

    def _get_balanced_progressive_conditioning(self, progress, image_tensor=None):
        """Balanced progressive conditioning combining all aspects"""

        if progress < 0.2:  # Early: Foundation
            prompt = "establish high quality facial foundation, balanced composition and structure"

        elif progress < 0.4:  # Early-mid: Structure + basic quality
            prompt = "develop facial structure with quality enhancement, clear feature definition"

        elif progress < 0.7:  # Mid-late: Details + quality
            prompt = "enhance facial details with high quality, natural texture and sharp focus"

        else:  # Final: Quality perfection
            prompt = "achieve photorealistic quality, exceptional detail and clarity, masterpiece portrait"

        return self._encode_progressive_prompt(prompt)

    def _get_adaptive_quality_terms(self, image_tensor, progress):
        """Get adaptive quality terms based on image characteristics and progress"""

        with torch.no_grad():
            mean_brightness = torch.mean(image_tensor).item()
            std_contrast = torch.std(image_tensor).item()

            adaptive_terms = ""

            # Lighting adaptation
            if mean_brightness < -0.2:
                adaptive_terms += ", enhanced lighting, brightness optimization"
            elif mean_brightness > 0.2:
                adaptive_terms += ", balanced exposure, natural lighting"

            # Contrast adaptation
            if std_contrast < 0.3:
                adaptive_terms += ", improved contrast, enhanced definition"

            # Progress-specific adaptations
            if progress > 0.7:  # Final stages need more quality emphasis
                adaptive_terms += ", ultra-high quality, exceptional clarity"

        return adaptive_terms

    def _encode_progressive_prompt(self, prompt):
        """Encode progressive prompt to embeddings"""

        text_input = self.diff_model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.diff_model.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return embeddings

    def get_adaptive_face_conditioning(self, image_tensor=None):
        """Generate adaptive conditioning based on image characteristics"""
        if image_tensor is None:
            return self.get_face_quality_embeddings()

        # Analyze image characteristics (simplified version)
        # In practice, you could use more sophisticated analysis
        with torch.no_grad():
            # Calculate image statistics for adaptive conditioning
            mean_brightness = torch.mean(image_tensor).item()
            std_contrast = torch.std(image_tensor).item()

            # Adaptive prompt selection based on image characteristics
            if mean_brightness < -0.2:  # Dark image
                prompt = "well-lit professional portrait, enhanced lighting, clear facial features, high quality"
            elif mean_brightness > 0.2:  # Bright image
                prompt = "balanced exposure portrait, natural lighting, detailed facial features, professional quality"
            elif std_contrast < 0.3:  # Low contrast
                prompt = "enhanced contrast portrait, sharp details, clear facial definition, high quality photography"
            else:  # Normal image
                prompt = "high quality detailed portrait photograph, natural facial features, professional lighting"

        # Add quality enhancement terms
        quality_terms = ", photorealistic, 8k resolution, masterpiece quality"
        full_prompt = prompt + quality_terms

        # Tokenize and encode
        text_input = self.diff_model.tokenizer(
            [full_prompt],
            padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.diff_model.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return embeddings

    def get_negative_conditioning(self):
        """Generate negative conditioning to avoid unwanted artifacts"""
        negative_prompts = [
            "blurry, low quality, distorted, pixelated",
            "noise, artifacts, compressed, low resolution",
            "unnatural, artificial, synthetic, fake",
            "overexposed, underexposed, poor lighting",
            "deformed, disfigured, bad anatomy"
        ]

        negative_prompt = ", ".join(negative_prompts)

        text_input = self.diff_model.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.diff_model.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return embeddings

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            generator = torch.Generator().manual_seed(8888)
            gpu_generator = torch.Generator(device=image.device)
            gpu_generator.manual_seed(generator.initial_seed())
            latents = self.diff_model.vae.encode(
                image).latent_dist.sample(generator=gpu_generator)
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latent):
        latent = 1 / 0.18215 * latent
        image = self.diff_model.vae.decode(latent)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def ddim_inversion(self, image):
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)
        latent = self.image2latent(image)
        all_latents = [latent]

        for i in tqdm(range(self.diffusion_steps - 1)):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]

            # Get progressive conditioning if enabled
            if (hasattr(self, 'enable_progressive_conditioning') and
                self.enable_progressive_conditioning and
                hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning):
                # For inversion, we go from step 0 to diffusion_steps-1
                uncond_embeddings = self.get_progressive_quality_conditioning(
                    i, self.diffusion_steps - 1, image)
            elif hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning:
                uncond_embeddings = self.get_adaptive_face_conditioning(image)
            else:
                uncond_embeddings = self.null_text_embeddings()

            # Ensure correct batch dimension for uncond_embeddings
            batch_size = latent.shape[0]
            if uncond_embeddings.shape[0] != batch_size:
                uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)

            noise_pred = self.diff_model.unet(latent,
                                              t,
                                              encoder_hidden_states=uncond_embeddings)["sample"]

            next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
            alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep] \
                if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
            reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t]) * (
                latent - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
            latent = reverse_x0 * \
                torch.sqrt(alpha_bar_next) + \
                torch.sqrt(1 - alpha_bar_next) * noise_pred
            all_latents.append(latent)

        return all_latents

    def null_optimization(self, inversion_latents):
        """
        Optimizing the unconditional embeddings based on the paper:
            Null-text Inversion for Editing Real Images using Guided Diffusion Models
        GiHub:
            https://github.com/google/prompt-to-prompt
        """
        all_uncond_embs = []

        latent = inversion_latents[self.start_step - 1]

        # Use quality-enhanced conditioning for null optimization
        if hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning:
            uncond_embeddings = self.get_face_quality_embeddings()
        else:
            uncond_embeddings = self.null_text_embeddings()

        # Ensure correct batch dimension for uncond_embeddings
        batch_size = latent.shape[0]
        if uncond_embeddings.shape[0] != batch_size:
            uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)

        uncond_embeddings.requires_grad_(True)
        optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
        #
        # criterion torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        #
        for i in tqdm(range(self.start_step, self.diffusion_steps)):
            t = self.diff_model.scheduler.timesteps[i]
            for _ in range(self.null_optimization_steps):
                out_latent = self.diffusion_step(latent, uncond_embeddings, t,
                                                 True)
                optimizer.zero_grad()
                loss = criterion(
                    out_latent, inversion_latents[i])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                latent = self.diffusion_step(latent, uncond_embeddings, t,
                                             True).detach()
                all_uncond_embs.append(uncond_embeddings.detach().clone())
        #
        uncond_embeddings.requires_grad_(False)
        return all_uncond_embs

    def visualize(self, image_name, real_image, latents, controller):
        adversarial_image = self.latent2image(latents)
        adversarial_image = adversarial_image[1:]

        # Add target_choice suffix to image name for easy identification
        # But avoid duplicate suffixes if image_name already contains directory number
        if f"_{self.target_choice.zfill(2)}" in image_name:
            # Image name already contains the directory suffix (e.g., image_0001_02)
            image_name_with_suffix = image_name
        elif f"_{self.target_choice}" in image_name:
            # Image name already contains the target choice suffix
            image_name_with_suffix = image_name
        else:
            # Add target_choice suffix
            image_name_with_suffix = f"{image_name}_{self.target_choice}"

        result_dir = self.protected_image_dir + '/' + \
            self.test_model_name[0] + '/' + \
            self.target_choice + '/' + image_name_with_suffix

        adversarial_img = cv2.cvtColor(adversarial_image[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_dir + ".png", adversarial_img)

        adversarial_image = adversarial_image.astype(np.float32) / 255

        real = (real_image / 2 + 0.5).clamp(0,
                                            1).permute(0, 2, 3, 1).cpu().numpy()
        """
        diff = adversarial_image - real
        diff_absolute = np.abs(diff)
        Image.fromarray(
            (diff_absolute[0] * 255).astype(np.uint8)).save(result_dir + "_diff_absolute.png")
        """

    def attacker(self,
                 image,
                 image_name,
                 source_embeddings,
                 target_embeddings,
                 controller,
                 null_text_dir=None,
                 bb_src1=None,
                 image_progress=None):
        # lat[0], lat[1], lat[2], ...
        inversion_latents = self.ddim_inversion(image)
        # reverse
        inversion_latents = inversion_latents[::-1]
        latent = inversion_latents[self.start_step - 1]
        #
        all_uncond_embs = self.null_optimization(inversion_latents)

        #######################################################################
        '''
        comparison between null_text and null_text optimized:
        '''
        if self.comparison_null_text:
            latent_holder = latent_holder_opt = latent.clone()
            # Use quality-enhanced conditioning for comparison
            if hasattr(self, 'use_quality_conditioning') and self.use_quality_conditioning:
                uncond_embeddings = self.get_face_quality_embeddings()
            else:
                uncond_embeddings = self.null_text_embeddings()

            # Ensure correct batch dimension for uncond_embeddings
            batch_size = latent.shape[0]
            if uncond_embeddings.shape[0] != batch_size:
                uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
            #
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    #
                    latent_holder = self.diffusion_step(latent_holder,
                                                        uncond_embeddings,
                                                        t, True)
                    #
                    latent_holder_opt = self.diffusion_step(latent_holder_opt,
                                                            all_uncond_embs[i -
                                                                            self.start_step],
                                                            t, True)
                    #
                image_rec = self.latent2image(latent_holder)
                image_rec = cv2.cvtColor(image_rec[0], cv2.COLOR_RGB2BGR)

                # Determine suffix for reconstruction images
                if f"_{self.target_choice.zfill(2)}" in image_name or f"_{self.target_choice}" in image_name:
                    suffix = ""
                else:
                    suffix = f"_{self.target_choice}"

                result_dir = os.path.join(
                    null_text_dir, f"{image_name}{suffix}_rec.png")
                cv2.imwrite(result_dir, image_rec)
                #
                image_rec_opt = self.latent2image(latent_holder_opt)
                image_rec_opt = cv2.cvtColor(
                    image_rec_opt[0], cv2.COLOR_RGB2BGR)
                result_dir = os.path.join(
                    null_text_dir, f"{image_name}{suffix}_rec_opt.png")
                cv2.imwrite(result_dir, image_rec_opt)
                #
            return None
        #######################################################################

        if self.is_makeup:
            latent_holder = latent.clone()
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    latent_holder = self.diffusion_step(latent_holder,
                                                        all_uncond_embs[i -
                                                                        self.start_step],
                                                        t, True)
            fast_render_image = self.diff_model.vae.decode(
                1 / 0.18215 * latent_holder)['sample']

        #
        self.set_attention_control(controller)

        null_context_guidance = [[torch.cat([all_uncond_embs[i]] * 4)]
                        for i in range(len(all_uncond_embs))]
        null_context_guidance = [torch.cat(i) for i in null_context_guidance]

        init_latent = latent.clone()
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=1e-2)

        progress_desc = f"Optimizing {image_progress}" if image_progress else "Optimizing"
        for step_idx, _ in enumerate(tqdm(range(self.prot_steps), desc=progress_desc)):
            controller.loss = 0
            controller.reset()

            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents,
                                              null_context_guidance[i -
                                                           self.start_step],
                                              t)

            out_image = self.diff_model.vae.decode(
                1 / 0.18215 * latents)['sample'][1:]
            #
            if self.MTCNN_cropping:
                out_image_hold = out_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                _, _, h, w = out_image_hold.shape
                if h != 0 and w != 0:
                    out_image = out_image_hold
            #
            if self.is_makeup:
                output_image_aug = torch.cat(
                    [self.augment(out_image) for i in range(1)], dim=0)
                clip_loss = self.nce_loss(fast_render_image,
                                          self.source_text,
                                          output_image_aug,
                                          self.makeup_prompt).sum()
                clip_loss = clip_loss * self.makeup_weight
            #
            output_embeddings = self.get_FR_embeddings(out_image)

            # Calculate standard adversarial loss (always computed)
            adv_loss = self.cosine_loss(
                output_embeddings, target_embeddings, source_embeddings) * self.adv_optim_weight

            # Calculate frequency-enhanced adversarial loss if enabled
            if self.enable_freq_adv:
                freq_total_loss = self.freq_aware_loss(
                    output_embeddings, target_embeddings, source_embeddings,
                    original_image=out_image,
                    perturbation_strength=self.freq_perturbation_strength,
                    fr_model_func=self.get_FR_embeddings
                )
                freq_adv_component = freq_total_loss * self.freq_adv_weight
            else:
                freq_adv_component = torch.tensor(0.0, device=adv_loss.device)

            self_attn_loss = controller.loss

            # Combined loss: original adversarial + frequency-enhanced adversarial (with regularization) + attention
            loss = adv_loss + freq_adv_component + self_attn_loss
            if self.is_makeup:
                loss += clip_loss
            #
            # Print loss information with image progress
            progress_info = f" [{image_progress}]" if image_progress else ""
            print(f"\nLoss values{progress_info}:")
            print(f"  standard_adv_loss: {adv_loss.item():.6f}")
            if self.enable_freq_adv:
                print(f"  freq_adv_loss (with regularization): {freq_adv_component.item():.6f}")
            print(f"  self_attn_loss: {self_attn_loss.item():.6f}")
            if self.is_makeup:
                print(f"  clip_loss: {clip_loss.item():.6f}")
            print(f"  total_loss: {loss.item():.6f}")
            if self.enable_freq_adv:
                print(f"  -> Standard + Frequency-enhanced adversarial perturbation (regularization embedded)")
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            controller.loss = 0
            controller.reset()
            #
            latents = torch.cat([init_latent, latent])
            #
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents,
                                              null_context_guidance[i -
                                                           self.start_step],
                                              t,
                                              step_idx=i,
                                              total_steps=self.diffusion_steps,
                                              image_tensor=out_image)
        #
        self.reset_attention_control()
        return latents.detach()

    def run(self):
        timer = MyTimer()
        time_list = []
        result_dir = self.protected_image_dir + '/' + \
            self.test_model_name[0] + '/' + self.target_choice
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        target_image, _ = get_target_test_images(
            self.target_choice, self.device, self.MTCNN_cropping)
        with torch.no_grad():
            target_embeddings = self.get_FR_embeddings(target_image)

        # Get total number of images for progress tracking
        total_images = len(self.dataloader)
        print(f"\n{'='*60}")
        print(f"Starting facial privacy protection processing...")
        print(f"Total images to process: {total_images}")
        print(f"{'='*60}\n")

        for i, (fname, image) in enumerate(self.dataloader):
            current_image = i + 1
            remaining_images = total_images - current_image
            print(f"\n{'='*60}")
            print(f"Processing image {current_image}/{total_images} (Remaining: {remaining_images})")
            print(f"Image name: {fname[0]}")
            print(f"{'='*60}")
            image_name = fname[0]
            image = image.to(self.device)
            #
            bb_src1 = None
            if self.MTCNN_cropping:
                # Try different image extensions
                possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                path = None
                for ext in possible_extensions:
                    test_path = self.source_dir + '/' + image_name + ext
                    if os.path.exists(test_path):
                        path = test_path
                        break

                if path is None:
                    print(f"Warning: Image file not found for {image_name}")
                    bb_src1 = [0, 0, self.image_size, self.image_size]  # Use full image as fallback
                else:
                    img = Image.open(path)
                    if img.size[0] != self.image_size:
                        img = img.resize((self.image_size, self.image_size))
                    bb_src1 = alignment(img)
            #
            controller = AttentionControlEdit(num_steps=self.diffusion_steps,
                                              self_replace_steps=1.0)
            #
            if self.comparison_null_text:
                null_text_dir = os.path.join(
                    self.protected_image_dir, "null_text_opt")
                os.makedirs(null_text_dir, exist_ok=True)
            else:
                null_text_dir = None
            #
            if self.is_obfuscation:
                image_hold = image.clone()
                if self.MTCNN_cropping:
                    out_image_hold = image_hold[:, :, round(bb_src1[1]):round(
                        bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                    _, _, h, w = out_image_hold.shape
                    if h != 0 and w != 0:
                        image_hold = out_image_hold
                with torch.no_grad():
                    source_embeddings = self.get_FR_embeddings(image_hold)
            else:
                source_embeddings = None
            #
            timer.tic()
            #
            # Create progress string for current image
            progress_str = f"{current_image}/{total_images}"

            latents = self.attacker(image,
                                    image_name,
                                    source_embeddings,
                                    target_embeddings,
                                    controller,
                                    null_text_dir,
                                    bb_src1,
                                    image_progress=progress_str)
            #
            avg_time = timer.toc()
            time_list.append(avg_time)

            print(f"\nImage {current_image}/{total_images} processing completed in {avg_time:.2f}s")
            estimated_remaining = avg_time * remaining_images
            print(f"Estimated remaining time: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}min)")

            if latents is not None:
                self.visualize(image_name, image, latents, controller)
                print(f"✓ Protected image saved for {fname[0]}")
            else:
                print(f"✗ Failed to generate protected image for {fname[0]}")
        #
        # Final summary
        avg_time = round(np.average(time_list), 2)
        total_time = round(sum(time_list), 2)
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images}")
        print(f"Average time per image: {avg_time}s")
        print(f"Total processing time: {total_time}s ({total_time/60:.1f}min)")
        print(f"Results saved to: {result_dir}")
        print(f"{'='*60}\n")

        result_fn = os.path.join(result_dir, "time.txt")
        with open(result_fn, 'a') as f:
            f.write(f"Time: {avg_time}\n")
            f.write(f"Total images: {total_images}\n")
            f.write(f"Total time: {total_time}s\n")
        f.close()