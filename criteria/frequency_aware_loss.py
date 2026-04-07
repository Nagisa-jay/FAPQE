import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-Aware Adversarial Perturbation Loss
    
    This class implements the frequency-aware adversarial perturbation method
    that targets high-frequency components for more effective adversarial attacks
    while preserving low-frequency content for better image quality.
    
    Core formulas:
    1. Image decomposition: I = I_low + I_high, where I_low = F^(-1)(1_{|ω|≤ω₁} F(I))
    2. Frequency perturbation: Δf = F^(-1)(w_high(ω) · c(ω)), where w_high(ω) = 1_{|ω|≥ω₁}
    3. Joint optimization: L = L_adv(I + Δf) + λ||Δf||₂²
    """
    
    def __init__(self, freq_threshold=0.1, reg_weight=0.01, is_obfuscation=False):
        """
        Initialize Frequency-Aware Loss
        
        Args:
            freq_threshold (float): Frequency threshold ω₁ for high-frequency mask (0.0-1.0)
            reg_weight (float): Regularization weight λ for perturbation magnitude
            is_obfuscation (bool): Whether to use obfuscation mode
        """
        super(FrequencyAwareLoss, self).__init__()
        self.freq_threshold = freq_threshold
        self.reg_weight = reg_weight
        self.is_obfuscation = is_obfuscation
        
    def create_high_freq_mask(self, shape, device, dtype):
        """
        Create high-frequency mask w_high(ω) = 1_{|ω|≥ω₁}
        
        Args:
            shape (tuple): Shape of the frequency domain (H, W)
            device: Device to create the mask on
            dtype: Data type for the mask
            
        Returns:
            torch.Tensor: High-frequency mask of shape (H, W)
        """
        H, W = shape
        
        # Create frequency coordinates
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        
        # Create meshgrid for 2D frequencies
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Calculate frequency magnitude |ω|
        freq_magnitude = torch.sqrt(fx**2 + fy**2)
        
        # Create high-frequency mask: 1 for |ω| ≥ ω₁, 0 otherwise
        high_freq_mask = (freq_magnitude >= self.freq_threshold).float()
        
        return high_freq_mask
    
    def apply_freq_perturbation(self, image, perturbation_strength=1.0):
        """
        Apply frequency-domain perturbation to image
        
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W)
            perturbation_strength (float): Strength of the perturbation
            
        Returns:
            tuple: (perturbed_image, freq_perturbation)
        """
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # Create high-frequency mask
        high_freq_mask = self.create_high_freq_mask((H, W), device, dtype)
        
        # Apply FFT to each channel
        perturbed_image = torch.zeros_like(image)
        total_perturbation = torch.zeros_like(image)

        for c in range(C):
            # Forward FFT
            image_fft = torch.fft.fft2(image[:, c, :, :])

            # Create perturbation in frequency domain
            # c(ω) represents the noise/perturbation in frequency domain
            noise_real = torch.randn_like(image_fft.real) * perturbation_strength
            noise_imag = torch.randn_like(image_fft.imag) * perturbation_strength
            freq_noise = torch.complex(noise_real, noise_imag)

            # Apply high-frequency mask: Δf = F^(-1)(w_high(ω) · c(ω))
            masked_freq_noise = freq_noise * high_freq_mask.unsqueeze(0)

            # Add perturbation to original frequency domain
            perturbed_fft = image_fft + masked_freq_noise

            # Inverse FFT to get perturbed image
            perturbed_channel = torch.fft.ifft2(perturbed_fft).real
            perturbed_image[:, c, :, :] = perturbed_channel

            # Calculate perturbation in image domain
            freq_perturbation = torch.fft.ifft2(masked_freq_noise).real
            total_perturbation[:, c, :, :] = freq_perturbation

        # Calculate perturbation magnitude as RMS (Root Mean Square) of the total perturbation
        # This gives a more meaningful scale that's comparable to image values
        # RMS = sqrt(mean(x^2)) gives the effective magnitude of the perturbation
        freq_perturbation_magnitude = torch.sqrt(torch.mean(total_perturbation**2))
        
        return perturbed_image, freq_perturbation_magnitude
    
    def cos_simi(self, emb_1, emb_2):
        """Calculate cosine similarity between embeddings"""
        return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))
    
    def forward(self, protected_feature, target_feature, source_feature,
                original_image=None, perturbation_strength=1.0, fr_model_func=None):
        """
        Forward pass for frequency-aware adversarial loss

        Args:
            protected_feature (list): Protected image features from FR models
            target_feature (list): Target identity features from FR models
            source_feature (list): Source identity features from FR models
            original_image (torch.Tensor): Original image for frequency perturbation
            perturbation_strength (float): Strength of frequency perturbation
            fr_model_func (callable): Function to extract FR features from images

        Returns:
            tuple: (freq_adv_loss, freq_reg_loss) - frequency-enhanced adversarial loss and regularization loss
        """
        # If no original image provided, return zero losses
        if original_image is None or fr_model_func is None:
            device = protected_feature[0].device if protected_feature else torch.device('cpu')
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Apply frequency perturbation to get perturbed image
        perturbed_image, freq_perturbation_magnitude = self.apply_freq_perturbation(
            original_image, perturbation_strength)

        # Extract features from perturbed image using the same FR models
        perturbed_features = fr_model_func(perturbed_image)

        # Calculate adversarial loss using perturbed image features
        cos_loss_list = []
        for i in range(len(perturbed_features)):
            if not self.is_obfuscation:
                # For impersonation: minimize distance to target
                cos_loss_list.append(
                    1 - self.cos_simi(perturbed_features[i],
                                      target_feature[i].detach()))
            else:
                # For obfuscation: maximize distance to source while minimizing to target
                imp = 1 - self.cos_simi(perturbed_features[i],
                                       target_feature[i].detach())
                obf = 1 - self.cos_simi(perturbed_features[i],
                                       source_feature[i].detach())
                cos_loss_list.append(imp - obf)

        freq_adv_loss = torch.sum(torch.stack(cos_loss_list))

        # Add regularization term directly to the adversarial loss: L_adv + λ||Δf||₂²
        regularization_term = self.reg_weight * freq_perturbation_magnitude
        total_freq_loss = freq_adv_loss + regularization_term

        return total_freq_loss
