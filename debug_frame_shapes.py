#!/usr/bin/env python3
"""Debug script to examine frame shapes from the collected data."""

import numpy as np
import os
from pathlib import Path

def analyze_collected_data():
    """Analyze the frame shapes in the collected episode data."""

    # Path to the collected data
    data_path = Path("runs/full_20250825_163746/ALE/Pong-v5/random_data/episodes.npz")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    print(f"Loading data from: {data_path}")

    # Load the data
    data = np.load(data_path, allow_pickle=True)

    print("Keys in the data file:")
    for key in data.keys():
        print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")

    # Look for frame data
    if 'observations' in data:
        obs = data['observations']
        print(f"\nObservation data:")
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Min value: {obs.min()}")
        print(f"  Max value: {obs.max()}")

        if len(obs.shape) >= 3:
            print(f"  Individual frame shape: {obs.shape[1:]}")

            # Show some sample frames
            print(f"\nFirst few frame shapes:")
            for i in range(min(5, len(obs))):
                print(f"  Frame {i}: {obs[i].shape}")

    # Check if there's a frames key
    if 'frames' in data:
        frames = data['frames']
        print(f"\nFrame data:")
        print(f"  Shape: {frames.shape}")
        print(f"  Dtype: {frames.dtype}")
        print(f"  Min value: {frames.min()}")
        print(f"  Max value: {frames.max()}")

    data.close()

if __name__ == "__main__":
    analyze_collected_data()
