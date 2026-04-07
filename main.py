# =============================================================================
# Import required libraries
# =============================================================================
import os
import random
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler

from dataset import ImageDataset
from adversarial_optimization import Adversarial_Opt
from tests import attack_local_models


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--source_dir',
                        default="assets/datasets/test",
                        type=str,
                        help="source images folder path for impersonation")
    parser.add_argument('--protected_image_dir',
                        default="results",
                        type=str)
    parser.add_argument('--target_choice',
                        default='1',
                        type=str,
                        help='Choice of target identity, as in AMT-GAN.')
    parser.add_argument('--comparison_null_text',
                        default=False,
                        type=bool)
    parser.add_argument("--test_model_name",
                        default=['mobile_face'])
    parser.add_argument("--surrogate_model_names",
                        default=['facenet', 'ir152', 'irse50'])



    parser.add_argument('--is_obfuscation',
                        default=False,
                        type=bool)
    # When applying makeup to the image
    parser.add_argument('--is_makeup',
                        default=False,
                        type=bool)
    parser.add_argument('--source_text',
                        default='face',
                        type=str)
    parser.add_argument('--makeup_prompt',
                        default='red lipstick',
                        type=str)

    parser.add_argument('--MTCNN_cropping',
                        default=True,
                        type=bool)
    parser.add_argument('--image_size',
                        default=256,
                        type=int)
    parser.add_argument('--prot_steps',
                        default=40,
                        type=int)
    parser.add_argument('--diffusion_steps',
                        default=20,
                        type=int)
    parser.add_argument('--start_step',
                        default=17,     
                        type=int,
                        help='Which DDIM step to start the protection (20 - 17 = 3)')
    parser.add_argument('--null_optimization_steps',
                        default=20,
                        type=int)

    parser.add_argument('--adv_optim_weight',
                        default=0.004,
                        type=float)
    parser.add_argument('--makeup_weight',
                        default=0,
                        type=float)

    # Frequency-aware adversarial perturbation parameters
    parser.add_argument('--enable_freq_adv',
                        default=True,
                        type=bool,
                        help='Enable frequency-aware adversarial perturbation')
    parser.add_argument('--freq_threshold',
                        default=0.1,
                        type=float,
                        help='Frequency threshold ω₁ for high-frequency mask (0.0-1.0)')
    parser.add_argument('--freq_reg_weight',
                        default=0.01,
                        type=float,
                        help='Regularization weight λ for frequency perturbation magnitude (should be small, e.g., 0.001-0.01)')
    parser.add_argument('--freq_perturbation_strength',
                        default=1.5,
                        type=float,
                        help='Strength of frequency domain perturbation')
    parser.add_argument('--freq_adv_weight',
                        default=0.004,
                        type=float,
                        help='Weight for frequency-enhanced adversarial loss')

    # Face quality enhancement conditioning parameters
    parser.add_argument('--use_quality_conditioning',
                        default=True,
                        type=bool,
                        help='Enable face quality enhancement conditioning instead of empty text')
    parser.add_argument('--face_conditioning_strategy',
                        default='quality',
                        type=str,
                        choices=['detail', 'quality', 'natural', 'professional', 'balanced'],
                        help='Strategy for face quality conditioning: '
                             'detail (focus on facial details), '
                             'quality (focus on image quality), '
                             'natural (focus on natural appearance), '
                             'professional (focus on professional photography), '
                             'balanced (balanced approach)')

    # Progressive quality conditioning parameters
    parser.add_argument('--enable_progressive_conditioning',
                        default=True,
                        type=bool,
                        help='Enable progressive quality conditioning that adapts to diffusion steps')
    parser.add_argument('--progressive_conditioning_mode',
                        default='quality_focused',
                        type=str,
                        choices=['quality_focused', 'structure_preserving', 'detail_enhancing', 'balanced'],
                        help='Mode for progressive conditioning: '
                             'quality_focused (optimize PSNR/SSIM/FID), '
                             'structure_preserving (optimize SSIM), '
                             'detail_enhancing (enhance texture details), '
                             'balanced (balanced progressive approach)')

    args = parser.parse_args()
    return args



def initialize_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    #
    initialize_seed(seed=10)
    #
    args = parse_args()
    #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the stable diffusion pretrained parameters
    diff_model = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base').to(args.device)
    diff_model.scheduler = DDIMScheduler.from_config(
        diff_model.scheduler.config)

    # Load the dataset
    dataset = ImageDataset(
        args.source_dir,
        transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])]
                           )
    )
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    adversarial_opt = Adversarial_Opt(args, diff_model)
    adversarial_opt.run()

    attack_local_models(args, protection=False)
    attack_local_models(args, protection=True)
