# World Models Complete Training Pipeline - WORKING VERSION
Write-Host "🚀 World Models Complete Training Pipeline" -ForegroundColor Cyan

# Configuration
$REPO_ROOT = "C:\Users\User\OneDrive - University of Cape Town\Honors\New folder"
$VENV_PYTHON = "$REPO_ROOT\.venv\Scripts\python.exe"
$WM_DIR = "$REPO_ROOT\world_models"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Free space: $([math]::Round((Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq 'C:' }).FreeSpace/1GB,2)) GB" -ForegroundColor Green

# Create run directories
$FULL_DIR = "$REPO_ROOT\runs\full_$TS"
$CLASSIC_DIR = "$REPO_ROOT\runs\classic_$TS"
New-Item -ItemType Directory -Force -Path $FULL_DIR, $CLASSIC_DIR | Out-Null

Write-Host "📁 Training directories created:" -ForegroundColor Yellow
Write-Host "  FULL: $FULL_DIR" -ForegroundColor Gray
Write-Host "  CLASSIC: $CLASSIC_DIR" -ForegroundColor Gray

# Change to world_models directory for training
Push-Location $WM_DIR

try {
    Write-Host "`n🎯 Starting FULL Curriculum Training (Box2D Environments)" -ForegroundColor Green
    Write-Host "Environments: Pong → LunarLander → Breakout → CarRacing" -ForegroundColor Yellow

    & $VENV_PYTHON curriculum_trainer_visual.py `
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
        --prefer-box2d true

    $fullResult = $LASTEXITCODE

    Write-Host "`n🎲 Starting CLASSIC Curriculum Training (Classic Control)" -ForegroundColor Green
    Write-Host "Environments: CartPole → MountainCar → Acrobot → Pendulum" -ForegroundColor Yellow

    & $VENV_PYTHON curriculum_trainer_visual.py `
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
        --prefer-box2d false

    $classicResult = $LASTEXITCODE

    # Results summary
    Write-Host "`n📊 Training Results Summary:" -ForegroundColor Cyan
    if ($fullResult -eq 0) {
        Write-Host "✅ FULL Curriculum: Completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ FULL Curriculum: Failed (Exit code: $fullResult)" -ForegroundColor Red
    }

    if ($classicResult -eq 0) {
        Write-Host "✅ CLASSIC Curriculum: Completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ CLASSIC Curriculum: Failed (Exit code: $classicResult)" -ForegroundColor Red
    }

    Write-Host "`n🎉 Training Pipeline Completed!" -ForegroundColor Cyan
    Write-Host "Results saved to:" -ForegroundColor Yellow
    Write-Host "  FULL: $FULL_DIR" -ForegroundColor Gray
    Write-Host "  CLASSIC: $CLASSIC_DIR" -ForegroundColor Gray

} finally {
    Pop-Location
}
