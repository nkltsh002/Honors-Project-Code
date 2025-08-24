"""
Repository root directory management for World Models project.
Ensures all scripts run from the correct repository root directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def find_repo_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the repository root by looking upward for common indicators.

    Args:
        start_path: Path to start searching from (defaults to current file's directory)

    Returns:
        Path to repository root or None if not found
    """
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current = start_path

    # Look for common repository root indicators
    indicators = ['.git', 'pyproject.toml', 'run_pipeline.py', 'world_models']

    # Search upward through directory tree
    for parent in [current] + list(current.parents):
        for indicator in indicators:
            if (parent / indicator).exists():
                return parent

    return None


def chdir_repo_root():
    """
    Change to repository root directory.
    Prints current directory for confirmation.
    Issues warning if repository root cannot be found.
    """
    repo_root = find_repo_root()

    if repo_root is None:
        print("Warning: Could not find repository root. Continuing from current directory.")
        print(f"Current directory: {os.getcwd()}")
        return

    # Change to repository root
    original_cwd = os.getcwd()
    os.chdir(repo_root)

    if original_cwd != str(repo_root):
        print(f"Changed directory to repository root: {repo_root}")
    else:
        print(f"Already in repository root: {repo_root}")


def get_repo_root() -> Optional[Path]:
    """
    Get repository root path without changing directory.

    Returns:
        Path to repository root or None if not found
    """
    return find_repo_root()


if __name__ == "__main__":
    # Test functionality
    print("Testing repo root detection...")
    root = find_repo_root()
    if root:
        print(f"Repository root found: {root}")
        print(f"Current directory: {os.getcwd()}")
        chdir_repo_root()
        print(f"After chdir: {os.getcwd()}")
    else:
        print("Repository root not found!")
