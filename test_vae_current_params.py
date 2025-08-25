#!/usr/bin/env python3
"""Test VAE dimensions with actual parameters used in training."""

import torch
import sys
import os
sys.path.append('world_models')

from models.conv_vae_dynamic import ConvVAE

def test_vae_dimensions():
    """Test VAE with the exact parameters from training."""

    print("Testing VAE dimensions...")

    # These are the exact parameters from the error log
    img_channels = 3  # Should be 3 for color images
    img_size = 32
    latent_dim = 32
    batch_size = 4

    print(f"Parameters: img_channels={img_channels}, img_size={img_size}, latent_dim={latent_dim}")
    print(f"Batch size: {batch_size}")

    # Create VAE
    vae = ConvVAE(
        img_channels=img_channels,
        img_size=img_size,
        latent_dim=latent_dim
    )

    print(f"VAE created successfully")
    print(f"Encoder: {vae.encoder}")
    print(f"Conv shape: {vae._conv_shape}")
    print(f"Conv flat: {vae._conv_flat}")

    # Test with dummy input
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    print(f"Input tensor shape: {x.shape}")

    # Test encoder
    try:
        h = vae.encoder(x)
        print(f"Encoder output shape: {h.shape}")

        h_flat = h.reshape(h.size(0), -1)
        print(f"Flattened shape: {h_flat.shape}")
        print(f"Expected flat features: {vae._conv_flat}")

        if h_flat.shape[1] != vae._conv_flat:
            print(f"ERROR: Dimension mismatch!")
            print(f"  Got: {h_flat.shape[1]}")
            print(f"  Expected: {vae._conv_flat}")
            return False

        # Test full forward pass
        recon, mu, logvar = vae(x)
        print(f"Forward pass successful:")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        return True

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_vae_dimensions()
    if success:
        print("VAE test passed!")
    else:
        print("VAE test failed!")
