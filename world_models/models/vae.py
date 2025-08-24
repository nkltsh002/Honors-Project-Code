"""
Convolutional Variational Autoencoder (VAE) for World Models

This module implements the Vision Model (V) component of World Models architecture.
The VAE compresses 64x64 RGB frames into a lower-dimensional latent representation z.

Key features:
- Convolutional encoder/decoder architecture
- β-VAE formulation for controlled disentanglement  
- Batch normalization for training stability
- GPU-optimized implementation with mixed precision support

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict, Any
import numpy as np

class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation.
    Used as building block for encoder and decoder.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 4, 
        stride: int = 2, 
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        )
        
        # Batch normalization (improves training stability)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class ConvTransposeBlock(nn.Module):
    """
    Transposed convolutional block for decoder.
    Upsamples feature maps while learning spatial patterns.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 4, 
        stride: int = 2, 
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Transposed convolution layer  
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv_transpose(x)))

class VAEEncoder(nn.Module):
    """
    Convolutional encoder that maps 64x64 RGB images to latent parameters.
    
    Architecture:
    64x64x3 -> 32x32x32 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> FC -> (mu, logvar)
    
    The encoder outputs both mean (mu) and log-variance (logvar) for the latent
    Gaussian distribution, enabling the reparameterization trick during training.
    """
    
    def __init__(self, latent_size: int = 32, hidden_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        
        self.latent_size = latent_size
        
        # Convolutional layers
        # Each layer reduces spatial dimensions by factor of 2 (stride=2)
        layers = []
        in_channels = 3  # RGB input
        
        for out_channels in hidden_channels:
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened feature size after convolutions
        # For 64x64 input with 4 layers of stride-2 convolutions: 64/16 = 4x4
        self.feature_size = hidden_channels[-1] * 4 * 4
        
        # Fully connected layers to latent parameters
        self.fc_mu = nn.Linear(self.feature_size, latent_size)
        self.fc_logvar = nn.Linear(self.feature_size, latent_size)
        
        # Initialize weights for better training dynamics
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_size)
            logvar: Log-variance of latent distribution (batch_size, latent_size)
        """
        # Normalize input to [-1, 1] range for better training stability
        x = x / 255.0 * 2.0 - 1.0
        
        # Pass through convolutional layers
        features = self.conv_layers(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Compute latent parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Clamp log-variance to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20, max=20)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """
    Convolutional decoder that reconstructs 64x64 RGB images from latent vectors.
    
    Architecture mirrors encoder in reverse:
    latent -> FC -> 4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x32 -> 64x64x3
    """
    
    def __init__(self, latent_size: int = 32, hidden_channels: Tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_channels = hidden_channels
        
        # Fully connected layer from latent to feature map
        self.feature_size = hidden_channels[0] * 4 * 4
        self.fc = nn.Linear(latent_size, self.feature_size)
        
        # Transposed convolutional layers
        layers = []
        in_channels = hidden_channels[0]
        
        for i, out_channels in enumerate(hidden_channels[1:]):
            layers.append(ConvTransposeBlock(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ))
            in_channels = out_channels
            
        # Final layer to RGB output (no batch norm, use sigmoid activation)
        layers.append(ConvTransposeBlock(
            in_channels, 3, kernel_size=4, stride=2, padding=1,
            use_bn=False, activation='sigmoid'
        ))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_size)
            
        Returns:
            reconstructed: Reconstructed image (batch_size, 3, 64, 64) in [0, 1]
        """
        # Map latent to feature map
        features = self.fc(z)
        features = features.view(features.size(0), self.hidden_channels[0], 4, 4)
        
        # Pass through transposed convolutions
        reconstructed = self.conv_layers(features)
        
        return reconstructed

class ConvVAE(nn.Module):
    """
    Complete Convolutional Variational Autoencoder for World Models.
    
    This is the Vision Model (V) that learns to compress high-dimensional
    observations into a compact latent representation z. The latent space
    captures the essential visual information needed for decision making.
    
    Key properties:
    - Uses β-VAE formulation for controlled disentanglement
    - Outputs both reconstruction and latent distribution parameters
    - Supports both training and inference modes
    - GPU-optimized with mixed precision support
    """
    
    def __init__(
        self, 
        latent_size: int = 32,
        hidden_channels: Tuple[int, ...] = (32, 64, 128, 256),
        beta: float = 4.0
    ):
        """
        Initialize ConvVAE.
        
        Args:
            latent_size: Dimensionality of latent space z
            hidden_channels: Number of channels in each conv layer
            beta: β parameter for β-VAE (controls KL regularization strength)
        """
        super().__init__()
        
        self.latent_size = latent_size
        self.beta = beta
        
        # Encoder and decoder networks
        self.encoder = VAEEncoder(latent_size, hidden_channels)
        self.decoder = VAEDecoder(latent_size, tuple(reversed(hidden_channels)))
        
        # Track training statistics
        self.register_buffer('num_batches', torch.tensor(0))
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters.
        
        Args:
            x: Input images (batch_size, 3, 64, 64)
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution  
        """
        return self.encoder(x)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, I) and sigma = exp(0.5 * logvar)
        
        This allows gradients to flow through the sampling operation.
        """
        if self.training:
            # Sample from latent distribution during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean during inference for deterministic behavior
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to reconstructed images.
        
        Args:
            z: Latent vectors (batch_size, latent_size)
            
        Returns:
            Reconstructed images (batch_size, 3, 64, 64)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through VAE.
        
        Args:
            x: Input images (batch_size, 3, 64, 64)
            
        Returns:
            Dictionary containing:
                - 'reconstruction': Reconstructed images
                - 'mu': Latent mean
                - 'logvar': Latent log-variance
                - 'z': Sampled latent vectors
        """
        # Encode input
        mu, logvar = self.encode(x)
        
        # Sample latent vectors
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruction
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction_weight: float = 1.0,
        warmup_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss components: reconstruction loss + β * KL divergence.
        
        Args:
            x: Input images (batch_size, 3, 64, 64)
            reconstruction_weight: Weight for reconstruction loss
            warmup_factor: Factor for KL warmup (gradually increase KL weight)
            
        Returns:
            Dictionary with loss components and metrics
        """
        # Forward pass
        outputs = self.forward(x)
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (Binary Cross Entropy)
        # We use BCE because output is in [0, 1] from sigmoid activation
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, x / 255.0, reduction='sum'
        ) / x.size(0)  # Normalize by batch size
        
        # KL divergence loss
        # KL[q(z|x) || p(z)] where p(z) = N(0, I)
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with β-VAE formulation
        total_loss = (
            reconstruction_weight * reconstruction_loss + 
            self.beta * warmup_factor * kl_loss
        )
        
        # Compute additional metrics
        with torch.no_grad():
            # Reconstruction accuracy (percentage of pixels within threshold)
            pixel_accuracy = (
                torch.abs(reconstruction - x / 255.0) < 0.1
            ).float().mean()
            
            # Latent statistics
            latent_mean_norm = torch.norm(mu, dim=1).mean()
            latent_std_mean = torch.exp(0.5 * logvar).mean()
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'pixel_accuracy': pixel_accuracy,
            'latent_mean_norm': latent_mean_norm,
            'latent_std_mean': latent_std_mean
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample new images from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images (num_samples, 3, 64, 64)
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior distribution N(0, I)
            z = torch.randn(num_samples, self.latent_size, device=device)
            
            # Decode samples
            samples = self.decode(z)
            
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images (for evaluation).
        
        Args:
            x: Input images
            
        Returns:
            Reconstructed images
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['reconstruction']
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation of input images.
        This is used by the MDN-RNN for temporal modeling.
        
        Args:
            x: Input images (batch_size, 3, 64, 64)
            
        Returns:
            Latent vectors (batch_size, latent_size)
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            # Use mean for deterministic encoding
            return mu

def test_vae():
    """Test VAE functionality with dummy data"""
    print("Testing ConvVAE...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    vae = ConvVAE(latent_size=32, beta=4.0).to(device)
    print(f"Model has {sum(p.numel() for p in vae.parameters()):,} parameters")
    
    # Test with dummy batch
    batch_size = 8
    x = torch.randint(0, 256, (batch_size, 3, 64, 64), device=device, dtype=torch.float32)
    
    # Forward pass
    outputs = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    
    # Loss computation
    losses = vae.compute_loss(x)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    print(f"KL loss: {losses['kl_loss'].item():.4f}")
    
    # Test sampling
    samples = vae.sample(num_samples=4, device=device)
    print(f"Generated samples shape: {samples.shape}")
    
    # Test latent encoding
    latents = vae.get_latent_representation(x)
    print(f"Latent representation shape: {latents.shape}")
    
    print("ConvVAE test passed!")

if __name__ == "__main__":
    test_vae()
