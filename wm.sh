#!/usr/bin/env bash
# World Models Shell Wrapper (Bash/macOS/Linux)
# Ensures scripts always run from repository root

set -e

# Get the directory where this script is located (repo root)
REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to repository root
cd "$REPO_ROOT"
echo "📁 Repository root: $REPO_ROOT"

# Pass all arguments to Python 3.12
python3 "$@"
