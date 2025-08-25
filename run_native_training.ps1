# Native Windows World Models Training Pipeline
# No Docker required - uses existing virtual environment

Write-Host "🚀 World Models Native Training Pipeline" -ForegroundColor Cyan
Write-Host "Free space: $((Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq 'C:' }).FreeSpace/1GB) GB" -ForegroundColor Green

# Set up paths
$REPO_ROOT = Get-Location
$VENV_PATH = "$REPO_ROOT\.venv\Scripts\python.exe"
$WM_DIR = "$REPO_ROOT\world_models"

# Verify environment
if (-not (Test-Path $VENV_PATH)) {
    Write-Error "Virtual environment not found at $VENV_PATH"
    exit 1
}

if (-not (Test-Path $WM_DIR)) {
    Write-Error "World models directory not found at $WM_DIR"
    exit 1
}

# Configure paths and settings
$env:PYTHONPATH = $REPO_ROOT
Set-Location $WM_DIR

# Generate timestamp for unique run directories
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$FULL_DIR = "$REPO_ROOT\runs\full_$TS"
$CLASSIC_DIR = "$REPO_ROOT\runs\classic_$TS"
$RESULTS_DIR = "$REPO_ROOT\runs\results_$TS"

# Create directories
New-Item -ItemType Directory -Force -Path $FULL_DIR, $CLASSIC_DIR, $RESULTS_DIR | Out-Null

Write-Host "📁 Created run directories:" -ForegroundColor Yellow
Write-Host "  FULL: $FULL_DIR" -ForegroundColor Gray
Write-Host "  CLASSIC: $CLASSIC_DIR" -ForegroundColor Gray
Write-Host "  RESULTS: $RESULTS_DIR" -ForegroundColor Gray

Write-Host "🎯 Starting FULL Curriculum (Box2D environments)..." -ForegroundColor Green
& $VENV_PATH curriculum_trainer_visual.py `
    --device cuda `
    --max-generations 50 `
    --episodes-per-eval 5 `
    --visualize False `
    --record-video False `
    --checkpoint-dir $FULL_DIR `
    --amp True `
    --tf32 True `
    --vae-img-size 32 `
    --vae-batch 4 `
    --grad-accum 8 `
    --prefer-box2d true `
    --num-workers 2 `
    --collect-episodes 10 `
    --collect-steps 1000 `
    --frames-for-vae 15000 `
    --frames-for-rnn 15000 `
    --mdnrnn-hidden 256 `
    --mdnrnn-mixtures 5 `
    --controller-pop 48

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ FULL curriculum completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ FULL curriculum failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "🎲 Starting CLASSIC Curriculum (Classic Control)..." -ForegroundColor Green
& $VENV_PATH curriculum_trainer_visual.py `
    --device cuda `
    --max-generations 50 `
    --episodes-per-eval 5 `
    --visualize False `
    --record-video False `
    --checkpoint-dir $CLASSIC_DIR `
    --amp True `
    --tf32 True `
    --vae-img-size 32 `
    --vae-batch 4 `
    --grad-accum 8 `
    --prefer-box2d false `
    --num-workers 2 `
    --collect-episodes 10 `
    --collect-steps 1000 `
    --frames-for-vae 15000 `
    --frames-for-rnn 15000 `
    --mdnrnn-hidden 256 `
    --mdnrnn-mixtures 5 `
    --controller-pop 48

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ CLASSIC curriculum completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ CLASSIC curriculum failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

# Run analysis if both completed
Write-Host "📊 Running comprehensive analysis..." -ForegroundColor Blue
Set-Location $REPO_ROOT
& $VENV_PATH scripts/analyze_results.py --full $FULL_DIR --classic $CLASSIC_DIR --outdir $RESULTS_DIR

Write-Host "🎉 Training pipeline completed!" -ForegroundColor Cyan
Write-Host "Results available in: $RESULTS_DIR" -ForegroundColor Yellow

Set-Location $REPO_ROOT
