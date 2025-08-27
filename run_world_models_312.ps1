# World Models Training Script for Python 3.12 with Box2D Support
# This script sets up the environment and runs World Models training

# Set environment variables
$env:PYTHONPATH = "C:\Users\User\OneDrive - University of Cape Town\Honors\New folder"
$env:PATH += ";$env:LOCALAPPDATA\Microsoft\WinGet\Packages\SWIG.SWIG_Microsoft.Winget.Source_8wekyb3d8bbwe\swigwin-4.3.1"

# Python 3.12 executable
$PYTHON_EXE = ".\.venv312\Scripts\python.exe"

Write-Host "🚀 World Models Training with Python 3.12 + Box2D Support" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# Verify environment
Write-Host "🔍 Verifying environment..." -ForegroundColor Yellow
& $PYTHON_EXE -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
& $PYTHON_EXE -c "import Box2D; print(f'Box2D: {Box2D.__version__}')"
& $PYTHON_EXE -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
Write-Host ""

# Parse command line arguments
$env_name = if ($args[0]) { $args[0] } else { "LunarLander-v3" }
$logdir = if ($args[1]) { $args[1] } else { "runs_312_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }

Write-Host "🎯 Environment: $env_name" -ForegroundColor Green
Write-Host "📁 Log Directory: $logdir" -ForegroundColor Green
Write-Host ""

# Run training
Write-Host "🏋️ Starting World Models training..." -ForegroundColor Magenta
& $PYTHON_EXE world_models/curriculum_trainer_visual.py --logdir $logdir --env-name $env_name --max-generations 5 --episodes-per-eval 3 --device cuda
