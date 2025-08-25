#!/usr/bin/env python3
"""Debug script to investigate VAE dimension issue with real data."""

import numpy as np
import torch
import sys
import os
from pathlib import Path
sys.path.append('world_models')

from models.conv_vae_dynamic import ConvVAE

def debug_data_and_vae():
    """Debug the actual data and VAE creation process."""

    # Check if the data file exists
    data_path = Path("runs/full_20250825_164529/ALE/Pong-v5/random_data/episodes.npz")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        # Look for any recent data files
        runs_dir = Path("runs")
        if runs_dir.exists():
            print("\nAvailable training runs:")
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    print(f"  {run_dir}")
                    episodes_file = run_dir / "ALE" / "Pong-v5" / "random_data" / "episodes.npz"
                    if episodes_file.exists():
                        print(f"    Found episodes.npz: {episodes_file}")
                        data_path = episodes_file
                        break

    if not data_path.exists():
        print("No episode data found. Cannot debug.")
        return

    print(f"Loading data from: {data_path}")

    # Load the actual collected data
    data = np.load(data_path, allow_pickle=True)

    print("Keys in data file:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  {key}: {type(data[key])}")

    # Find the frame data
    frames_key = None
    if 'observations' in data:
        frames_key = 'observations'
    elif 'frames' in data:
        frames_key = 'frames'
    else:
        print("No frame data found in file")
        data.close()
        return

    all_frames = data[frames_key]
    print(f"\nFrame data analysis:")
    print(f"  Key: {frames_key}")
    print(f"  Shape: {all_frames.shape}")
    print(f"  Dtype: {all_frames.dtype}")
    print(f"  Min value: {all_frames.min()}")
    print(f"  Max value: {all_frames.max()}")

    # Simulate the VAE creation logic from the training code
    print(f"\n=== VAE Creation Logic ===")
    if len(all_frames.shape) == 4:  # Color or stacked frames
        raw_channels = all_frames.shape[-1]
        print(f"Raw channels detected: {raw_channels}")
        if raw_channels <= 4:
            input_channels = raw_channels
            print(f"Using raw channels: {input_channels}")
        else:
            # For frames with many channels, use only first 3 (RGB)
            input_channels = 3
            print(f"Frame data has {raw_channels} channels, using first 3 (RGB)")
    else:
        input_channels = 1
        print(f"Grayscale or single channel, using 1 channel")

    print(f"VAE will be created with {input_channels} input channels")

    # Create VAE with the determined input channels
    vae = ConvVAE(
        img_channels=input_channels,
        img_size=32,
        latent_dim=32
    )

    print(f"VAE created successfully")
    print(f"  Conv shape: {vae._conv_shape}")
    print(f"  Conv flat: {vae._conv_flat}")

    # Simulate the tensor processing logic
    print(f"\n=== Tensor Processing Logic ===")

    # Take a small sample for testing
    sample_frames = all_frames[:8]  # 8 frames for testing
    print(f"Sample frames shape: {sample_frames.shape}")

    # Apply channel slicing if needed
    _vae_input_channels = input_channels
    processed_frames = sample_frames
    if len(sample_frames.shape) == 4 and sample_frames.shape[-1] > _vae_input_channels:
        processed_frames = sample_frames[..., :_vae_input_channels]
        print(f"Sliced frames from {sample_frames.shape[-1]} to {_vae_input_channels} channels")
        print(f"Processed frames shape: {processed_frames.shape}")

    # Convert to tensor
    frames_tensor = torch.FloatTensor(processed_frames).permute(0, 3, 1, 2) / 255.0
    print(f"Tensor shape after permute: {frames_tensor.shape}")

    # Test with VAE batch size
    batch_size = 4
    batch = frames_tensor[:batch_size]
    print(f"Batch tensor shape: {batch.shape}")

    # Test VAE forward pass
    try:
        print(f"\n=== VAE Forward Pass Test ===")
        with torch.no_grad():
            # Test encoder
            h = vae.encoder(batch)
            print(f"Encoder output shape: {h.shape}")

            h_flat = h.reshape(h.size(0), -1)
            print(f"Flattened shape: {h_flat.shape}")
            print(f"Expected flat features: {vae._conv_flat}")

            if h_flat.shape[1] != vae._conv_flat:
                print(f"❌ DIMENSION MISMATCH DETECTED!")
                print(f"  Got: {h_flat.shape[1]} features")
                print(f"  Expected: {vae._conv_flat} features")
                print(f"  Ratio: {h_flat.shape[1] / vae._conv_flat:.2f}")

                # Calculate what the expected shape should be
                expected_channels = h_flat.shape[1] // (vae._conv_shape[1] * vae._conv_shape[2])
                print(f"  Implied channels: {expected_channels}")

                data.close()
                return False

            # Full forward pass
            recon, mu, logvar = vae(batch)
            print(f"✅ Full forward pass successful!")
            print(f"  Reconstruction shape: {recon.shape}")
            print(f"  Mu shape: {mu.shape}")
            print(f"  Logvar shape: {logvar.shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        data.close()
        return False

    data.close()
    print(f"\n✅ All tests passed! VAE should work correctly.")
    return True

if __name__ == "__main__":
    debug_data_and_vae()
