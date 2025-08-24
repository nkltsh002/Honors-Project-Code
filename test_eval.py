#!/usr/bin/env python3
"""
Test script for eval_render functionality
"""

import argparse

def main():
    print("Starting main function...")

    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--env', type=str, default=None, help='Environment name')

    print("Parser created, parsing arguments...")
    args = parser.parse_args()

    print(f"Arguments parsed successfully: {args}")
    print("Test completed!")

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script completed.")
