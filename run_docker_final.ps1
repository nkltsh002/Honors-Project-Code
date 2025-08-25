# World Models Docker Training - Fixed for Windows
$DATA="C:\wm_docker"
$RUNS="$DATA\runs"
$CACHE="$DATA\pipcache"
$VENV="$DATA\venv"

Write-Host "🐳 Starting World Models Docker Training" -ForegroundColor Cyan
Write-Host "Data directory: $DATA" -ForegroundColor Yellow
Write-Host "Free space: $((Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }).FreeSpace/1GB) GB" -ForegroundColor Green

# Ensure directories exist
New-Item -ItemType Directory -Force -Path $RUNS,$CACHE,$VENV | Out-Null

# Run Docker with corrected paths and proper escaping
docker run --rm --gpus all `
  -v "${PWD}:/app" `
  -v "${RUNS}:/runs" `
  -v "${CACHE}:/root/.cache/pip" `
  -v "${VENV}:/venv" `
  -w /app `
  python:3.12-slim bash -c @"
set -e
echo '📦 Installing system dependencies...'
apt-get update && apt-get install -y --no-install-recommends \
  git swig ffmpeg libgl1 libglib2.0-0 build-essential && \
  rm -rf /var/lib/apt/lists/*

echo '🐍 Setting up Python environment...'
python -m venv /venv
. /venv/bin/activate

echo '📚 Installing Python packages...'
python -m pip install -U pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install gymnasium 'gymnasium[classic-control]' 'gymnasium[box2d]' 'gymnasium[atari,accept-roms]' \
  ale-py autorom matplotlib pandas numpy scipy tqdm tensorboard opencv-python cma

echo '🎮 Setting up Atari ROMs...'
AutoROM --accept-license || true

echo '🏁 Starting training pipeline...'
python tools/ensure_cwd.py

# Generate timestamp and directories
TS=$$(date +%Y%m%d_%H%M%S)
FULL=/runs/full_$$TS
CLASSIC=/runs/classic_$$TS
mkdir -p $$FULL $$CLASSIC

echo '🎯 Starting FULL curriculum (Box2D environments)...'
python world_models/curriculum_trainer_visual.py \
  --device cuda \
  --max-generations 50 \
  --episodes-per-eval 5 \
  --visualize false \
  --record-video false \
  --checkpoint-dir $$FULL \
  --amp true \
  --tf32 true \
  --vae-img-size 32 \
  --vae-latent 32 \
  --vae-batch 4 \
  --grad-accum 8 \
  --num-workers 2 \
  --mdnrnn-hidden 256 \
  --mdnrnn-mixtures 5 \
  --controller-pop 48 \
  --collect-episodes 8 \
  --collect-steps 800 \
  --frames-for-vae 12000 \
  --frames-for-rnn 12000 \
  --no-save-episodes \
  --store-latents \
  --latents-chunk 1024 \
  --grayscale \
  --prefer-box2d true

echo '🎲 Starting CLASSIC curriculum (Classic Control)...'
python world_models/curriculum_trainer_visual.py \
  --device cuda \
  --max-generations 50 \
  --episodes-per-eval 5 \
  --visualize false \
  --record-video false \
  --checkpoint-dir $$CLASSIC \
  --amp true \
  --tf32 true \
  --vae-img-size 32 \
  --vae-latent 32 \
  --vae-batch 4 \
  --grad-accum 8 \
  --num-workers 2 \
  --mdnrnn-hidden 256 \
  --mdnrnn-mixtures 5 \
  --controller-pop 48 \
  --collect-episodes 8 \
  --collect-steps 800 \
  --frames-for-vae 12000 \
  --frames-for-rnn 12000 \
  --no-save-episodes \
  --store-latents \
  --latents-chunk 1024 \
  --grayscale \
  --prefer-box2d false

echo '📊 Running comprehensive analysis...'
python scripts/analyze_results.py --full $$FULL --classic $$CLASSIC --outdir /runs/results

echo '✅ Training pipeline completed!'
echo 'Results saved to: /runs/results'
"@

Write-Host "🎉 Docker training completed!" -ForegroundColor Green
Write-Host "Results available in: $RUNS" -ForegroundColor Cyan
