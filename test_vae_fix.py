#!/usr/bin/env python3
"""Test VAE with problematic channel dimensions."""

import torch
import numpy as np
import sys
import os
sys.path.append('world_models')

from models.conv_vae_dynamic import ConvVAE

def test_vae_with_many_channels():
    """Test VAE creation and processing with many-channel input."""

    print("Testing VAE with high-channel input data...")

    # Simulate the problematic scenario
    batch_size = 4
    img_size = 32
    raw_channels = 35  # This would cause the 35840 error

    # Create fake frame data with many channels
    all_frames = np.random.randint(0, 256, size=(100, img_size, img_size, raw_channels), dtype=np.uint8)
    print(f"Fake frame data shape: {all_frames.shape}")

    # Apply the fixed logic from our training code
    if len(all_frames.shape) == 4:  # Color or stacked frames
        raw_channels_detected = all_frames.shape[-1]
        if raw_channels_detected <= 4:
            input_channels = raw_channels_detected
        else:
            # For frames with many channels, use only first 3 (RGB)
            input_channels = 3
            print(f"Frame data has {raw_channels_detected} channels, using first 3 (RGB)")
    else:
        input_channels = 1

    print(f"VAE will be created with {input_channels} input channels")

    # Create VAE with correct input channels
    vae = ConvVAE(
        img_channels=input_channels,
        img_size=img_size,
        latent_dim=32
    )

    print(f"VAE created successfully")
    print(f"VAE expects {input_channels} input channels")
    print(f"Conv flat size: {vae._conv_flat}")

    # Process frames with channel slicing
    processed_frames = all_frames
    if len(all_frames.shape) == 4 and all_frames.shape[-1] > input_channels:
        # Slice to use only the required number of channels
        processed_frames = all_frames[..., :input_channels]
        print(f"Sliced frames from {all_frames.shape[-1]} to {input_channels} channels")

    print(f"Processed frames shape: {processed_frames.shape}")

    # Test with a small batch
    sample_frames = processed_frames[:batch_size]
    frames_tensor = torch.FloatTensor(sample_frames).permute(0, 3, 1, 2) / 255.0

    print(f"Tensor shape after permute: {frames_tensor.shape}")

    # Test forward pass
    try:
        recon, mu, logvar = vae(frames_tensor)
        print(f"Forward pass successful!")
        print(f"  Input shape: {frames_tensor.shape}")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        return True
    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_vae_with_many_channels()
    if success:
        print("\n✅ VAE fix test passed! The channel slicing works correctly.")
    else:
        print("\n❌ VAE fix test failed!")
