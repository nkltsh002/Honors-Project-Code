#!/usr/bin/env python3
"""
Quick smoke test for dynamic ConvVAE with 32px images
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'world_models'))

from models.conv_vae_dynamic import ConvVAE

def test_dynamic_vae():
    print("Testing dynamic ConvVAE with 32px images...")

    # Test with 32px images (the problematic size)
    batch_size = 4
    img_channels = 3
    img_size = 32
    latent_dim = 32

    # Create VAE
    vae = ConvVAE(img_channels=img_channels, img_size=img_size, latent_dim=latent_dim)
    print(f"Created VAE for {img_size}x{img_size} images with {latent_dim} latent dims")

    # Create test batch
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")

    # Test forward pass
    try:
        x_recon, mu, logvar = vae(x)
        print(f"✅ Forward pass successful!")
        print(f"  Reconstruction shape: {x_recon.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")

        # Test individual components
        mu_test, logvar_test = vae.encode(x)
        z = vae.reparameterize(mu_test, logvar_test)
        x_recon_test = vae.decode(z)

        print(f"  Encoded latent shape: {z.shape}")
        print(f"  Decoded shape: {x_recon_test.shape}")

        # Check shapes are consistent
        assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} != {x.shape}"
        assert mu.shape == (batch_size, latent_dim), f"Mu shape wrong: {mu.shape}"
        assert logvar.shape == (batch_size, latent_dim), f"Logvar shape wrong: {logvar.shape}"

        print("✅ All shape checks passed!")

        # Test with different size too (64px)
        print("\nTesting with 64px images...")
        vae64 = ConvVAE(img_channels=3, img_size=64, latent_dim=32)
        x64 = torch.randn(2, 3, 64, 64)
        x64_recon, mu64, logvar64 = vae64(x64)
        print(f"✅ 64px test successful! Output shape: {x64_recon.shape}")

        return True

    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dynamic_vae()
    if success:
        print("\n🎉 Dynamic VAE smoke test PASSED! Ready for training.")
    else:
        print("\n💥 Dynamic VAE smoke test FAILED!")
        sys.exit(1)
