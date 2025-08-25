import torch
import numpy as np
import os
import sys

# Add the world_models directory to path
sys.path.append(os.path.join(os.getcwd(), 'world_models'))

from models.conv_vae_dynamic import ConvVAE

print("🔍 VAE Dimension Debug Test")
print("="*50)

# Test VAE with 32x32 images (same as training)
img_size = 32
img_channels = 3
batch_size = 4
latent_dim = 32

print(f"Image size: {img_size}x{img_size}")
print(f"Image channels: {img_channels}")
print(f"Batch size: {batch_size}")
print(f"Latent dim: {latent_dim}")

# Create VAE
vae = ConvVAE(
    img_channels=img_channels,
    img_size=img_size,
    latent_dim=latent_dim,
    enc_channels=(32, 64, 128, 256)
)

print(f"\nVAE Internal Dimensions:")
print(f"Conv shape: {vae._conv_shape}")
print(f"Conv flat: {vae._conv_flat}")

# Test with different batch sizes
for test_batch_size in [1, 4, 8]:
    print(f"\n--- Testing with batch size {test_batch_size} ---")

    # Create dummy input
    x = torch.randn(test_batch_size, img_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")

    # Test encoder
    try:
        h = vae.encoder(x)
        print(f"After encoder: {h.shape}")

        h_flat = h.reshape(h.size(0), -1)
        print(f"Flattened: {h_flat.shape}")

        # Test the linear layers
        mu = vae.fc_mu(h_flat)
        logvar = vae.fc_logvar(h_flat)
        print(f"Mu shape: {mu.shape}")
        print(f"Logvar shape: {logvar.shape}")

        # Test full forward pass
        recon, mu, logvar = vae(x)
        print(f"✅ SUCCESS: Reconstruction shape: {recon.shape}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Expected linear input: {h_flat.shape[1] if 'h_flat' in locals() else 'unknown'}")
        print(f"Linear layer expects: {vae.fc_mu.in_features}")

print("\n" + "="*50)
print("🔧 Diagnosis Complete")
