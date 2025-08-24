"""
Convolutional Variational Autoencoder (ConvVAE) for World Models

This module implements the Vision Model (V) component of World Models architecture.
The ConvVAE compresses 64x64 RGB frames into a 32-dimensional latent representation.

Architecture:
- Input: 64x64x3 RGB images
- Encoder: 4 Conv layers (stride 2) -> flattened features -> mu, logvar
- Latent: z_dim = 32 with reparameterization trick
- Decoder: 4 ConvTranspose layers to reconstruct 64x64x3 image
- Loss: MSE reconstruction + KL divergence

Compatible with PyTorch 2.x and CUDA acceleration.

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (ConvVAE) implementation.
    
    This VAE compresses 64x64 RGB images into a 32-dimensional latent space
    using convolutional encoder-decoder architecture with reparameterization trick.
    
    Args:
        latent_size (int): Dimensionality of latent space z (default: 32)
        input_channels (int): Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, latent_size: int = 32, input_channels: int = 3):
        super().__init__()
        
        self.latent_size = latent_size
        self.input_channels = input_channels
        
        # =====================================================================
        # ENCODER: 64x64x3 -> 32x32x32 -> 16x16x64 -> 8x8x128 -> 4x4x256
        # =====================================================================
        
        self.encoder = nn.Sequential(
            # Layer 1: 64x64x3 -> 32x32x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2: 32x32x32 -> 16x16x64  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 4: 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Calculate flattened feature size: 4x4x256 = 4096
        self.encoder_output_size = 4 * 4 * 256
        
        # =====================================================================
        # LATENT SPACE: Fully connected layers for mu and logvar
        # =====================================================================
        
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_size)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_size)
        
        # =====================================================================
        # DECODER: z -> 4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x32 -> 64x64x3
        # =====================================================================
        
        # Project latent vector back to feature map
        self.fc_decode = nn.Linear(latent_size, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            # Layer 1: 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 2: 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 4: 32x32x32 -> 64x64x3 (final reconstruction)
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid activation for pixel values in [0,1]
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization.
        This helps with training stability and convergence.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Xavier initialization for convolutional layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization for fully connected layers
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard batch norm initialization
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent parameters (mu, logvar).
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 64, 64)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - mu: Mean of latent Gaussian distribution (batch_size, latent_size)
                - logvar: Log-variance of latent Gaussian distribution (batch_size, latent_size)
        """
        # Pass through convolutional encoder
        encoded = self.encoder(x)  # Shape: (batch_size, 256, 4, 4)
        
        # Flatten for fully connected layers
        encoded_flat = encoded.view(encoded.size(0), -1)  # Shape: (batch_size, 4096)
        
        # Compute latent parameters
        mu = self.fc_mu(encoded_flat)        # Mean: (batch_size, latent_size)
        logvar = self.fc_logvar(encoded_flat)  # Log-variance: (batch_size, latent_size)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent Gaussian distribution.
        
        Instead of sampling z ~ N(mu, sigma^2), we sample epsilon ~ N(0,1) 
        and compute z = mu + sigma * epsilon. This maintains differentiability
        for backpropagation through the sampling process.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution (batch_size, latent_size)
            logvar (torch.Tensor): Log-variance of latent distribution (batch_size, latent_size)
            
        Returns:
            torch.Tensor: Sampled latent vector z (batch_size, latent_size)
        """
        if self.training:
            # Sample epsilon from standard normal distribution
            epsilon = torch.randn_like(mu)  # Shape: (batch_size, latent_size)
            
            # Convert log-variance to standard deviation: sigma = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)
            
            # Reparameterization: z = mu + sigma * epsilon  
            z = mu + std * epsilon
        else:
            # During inference, use the mean (no sampling)
            z = mu
            
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector z back to image reconstruction.
        
        Args:
            z (torch.Tensor): Latent vector (batch_size, latent_size)
            
        Returns:
            torch.Tensor: Reconstructed images (batch_size, 3, 64, 64)
        """
        # Project latent vector to decoder input size
        decoded = self.fc_decode(z)  # Shape: (batch_size, 4096)
        
        # Reshape to feature map for transposed convolutions
        decoded = decoded.view(-1, 256, 4, 4)  # Shape: (batch_size, 256, 4, 4)
        
        # Pass through transposed convolutional decoder
        reconstruction = self.decoder(decoded)  # Shape: (batch_size, 3, 64, 64)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass through ConvVAE.
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 64, 64)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - reconstruction: Reconstructed images (batch_size, 3, 64, 64)  
                - mu: Latent means (batch_size, latent_size)
                - logvar: Latent log-variances (batch_size, latent_size)
        """
        # Encode input to latent parameters
        mu, logvar = self.encode(x)
        
        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode latent vector to reconstruction
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new images by sampling from the learned latent distribution.
        
        Args:
            num_samples (int): Number of samples to generate
            device (torch.device): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated images (num_samples, 3, 64, 64)
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_size, device=device)
        
        # Decode to generate images
        with torch.no_grad():
            generated_images = self.decode(z)
            
        return generated_images
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor, 
                    beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction loss + beta * KL divergence.
        
        Args:
            x (torch.Tensor): Original input images
            reconstruction (torch.Tensor): Reconstructed images
            mu (torch.Tensor): Latent means
            logvar (torch.Tensor): Latent log-variances
            beta (float): Beta parameter for β-VAE (default: 1.0)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Combined reconstruction + KL loss
                - reconstruction_loss: MSE reconstruction loss
                - kl_loss: KL divergence loss
        """
        # Reconstruction loss: Mean Squared Error (MSE)
        # Measures how well the VAE reconstructs the input images
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')
        reconstruction_loss = reconstruction_loss / x.size(0)  # Average over batch
        
        # KL divergence loss: KL(q(z|x) || p(z))
        # Measures how close the learned latent distribution is to standard normal
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        
        # Total loss: reconstruction + beta * KL divergence
        # Beta parameter allows for β-VAE training (beta > 1 encourages disentanglement)
        total_loss = reconstruction_loss + beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation of input images (for downstream tasks).
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 64, 64)
            
        Returns:
            torch.Tensor: Latent representations (batch_size, latent_size)
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu  # Use mean as deterministic representation


def test_convvae():
    """
    Test function to verify ConvVAE implementation.
    Tests forward pass, loss computation, and CUDA compatibility.
    """
    print("Testing ConvVAE implementation...")
    
    # Create model
    model = ConvVAE(latent_size=32)
    
    # Test CPU
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    
    reconstruction, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss computation
    total_loss, recon_loss, kl_loss = model.compute_loss(x, reconstruction, mu, logvar)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA compatibility...")
        device = torch.device('cuda')
        model_cuda = model.to(device)
        x_cuda = x.to(device)
        
        reconstruction_cuda, mu_cuda, logvar_cuda = model_cuda(x_cuda)
        print(f"CUDA forward pass successful!")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Test sampling
    samples = model.sample(2, torch.device('cpu'))
    print(f"Generated samples shape: {samples.shape}")
    
    print("✅ ConvVAE implementation test completed successfully!")


if __name__ == "__main__":
    test_convvae()
