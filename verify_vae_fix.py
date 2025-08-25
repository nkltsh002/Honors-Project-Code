import torch
import numpy as np
import os
import sys

# Add the world_models directory to path
sys.path.append(os.path.join(os.getcwd(), 'world_models'))

from models.conv_vae_dynamic import ConvVAE

print("🔧 Testing Fixed VAE Configuration")
print("="*50)

# Test with the exact configuration from the training
img_size = 32  # Same as --vae-img-size 32
img_channels = 3
batch_size = 4  # Same as --vae-batch 4
latent_dim = 32  # Default vae_latent_size

print(f"Configuration:")
print(f"  Image size: {img_size}x{img_size}")
print(f"  Channels: {img_channels}")
print(f"  Batch size: {batch_size}")
print(f"  Latent dim: {latent_dim}")

# Create VAE with explicit parameters (as fixed)
vae = ConvVAE(
    img_channels=img_channels,
    img_size=img_size,
    latent_dim=latent_dim,
    enc_channels=(32, 64, 128, 256)
)

print(f"\n✅ VAE created successfully")
print(f"Conv shape: {vae._conv_shape}")
print(f"Conv flat: {vae._conv_flat}")

# Test the exact scenario that was failing
print(f"\n🧪 Testing batch size {batch_size} (training scenario)")
x = torch.randn(batch_size, img_channels, img_size, img_size)
print(f"Input shape: {x.shape}")

try:
    # Test encoder step by step
    h = vae.encoder(x)
    print(f"After encoder: {h.shape}")

    h_flat = h.reshape(h.size(0), -1)
    print(f"Flattened shape: {h_flat.shape}")

    # Test the problematic linear layers
    mu = vae.fc_mu(h_flat)
    logvar = vae.fc_logvar(h_flat)
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Test full forward pass
    recon, mu, logvar = vae(x)
    print(f"✅ SUCCESS: Full forward pass works!")
    print(f"Reconstruction shape: {recon.shape}")

    print(f"\n🎉 VAE fix verified - training should work now!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    print(f"This indicates the fix didn't work properly")

print("\n" + "="*50)
