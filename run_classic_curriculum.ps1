# CLASSIC Curriculum Training Command (Ready to Run)
# Execute this after FULL curriculum completes

$TS = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "🎲 Starting CLASSIC Curriculum Training..." -ForegroundColor Green

& "C:\Users\User\OneDrive - University of Cape Town\Honors\New folder\.venv\Scripts\python.exe" `
  "C:\Users\User\OneDrive - University of Cape Town\Honors\New folder\world_models\curriculum_trainer_visual.py" `
  --device cuda `
  --max-generations 50 `
  --episodes-per-eval 5 `
  --visualize False `
  --record-video False `
  --checkpoint-dir "C:\Users\User\OneDrive - University of Cape Town\Honors\New folder\runs\classic_$TS" `
  --amp True `
  --tf32 True `
  --vae-img-size 32 `
  --vae-batch 4 `
  --grad-accum 8 `
  --prefer-box2d false

Write-Host "✅ CLASSIC Curriculum completed!" -ForegroundColor Green
