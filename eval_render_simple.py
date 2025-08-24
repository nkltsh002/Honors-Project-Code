#!/usr/bin/env python3
"""
Minimal eval_render script to test functionality
"""

import argparse
import sys

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Visual Environment Evaluation with Real-Time Rendering"
    )

    # Environment selection
    parser.add_argument('--env', type=str, default=None,
                       help='Environment name (default: auto-select from curriculum)')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to run (default: 2)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for rendering (default: 30)')

    args = parser.parse_args()

    print("Visual Environment Evaluation - WORKING!")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"FPS: {args.fps}")

if __name__ == "__main__":
    main()
