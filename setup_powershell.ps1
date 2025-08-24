# World Models PowerShell Setup Script
# Run this once to configure PowerShell execution policy for this project

Write-Host "World Models PowerShell Setup" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Check current execution policy
$currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
Write-Host "Current execution policy: $currentPolicy" -ForegroundColor Yellow

if ($currentPolicy -eq "Undefined" -or $currentPolicy -eq "Restricted") {
    Write-Host ""
    Write-Host "PowerShell execution is restricted. This will prevent wm.ps1 from running." -ForegroundColor Red
    Write-Host ""
    Write-Host "RECOMMENDED SOLUTION (no admin required):" -ForegroundColor Green
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
    Write-Host ""

    $response = Read-Host "Apply this setting now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        Write-Host "Execution policy updated successfully!" -ForegroundColor Green

        # Test the wrapper
        Write-Host ""
        Write-Host "Testing PowerShell wrapper..." -ForegroundColor Cyan
        try {
            & ".\wm.ps1" "--version"
            Write-Host "PowerShell wrapper is now working!" -ForegroundColor Green
        }
        catch {
            Write-Host "Test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "Setup cancelled. Alternative options:" -ForegroundColor Yellow
        Write-Host "One-time: powershell -ExecutionPolicy Bypass -File .\wm.ps1 <args>" -ForegroundColor White
    }
}
else {
    Write-Host "PowerShell execution policy is already configured correctly!" -ForegroundColor Green

    # Test the wrapper
    Write-Host ""
    Write-Host "Testing PowerShell wrapper..." -ForegroundColor Cyan
    try {
        & ".\wm.ps1" "--version"
        Write-Host "PowerShell wrapper is working perfectly!" -ForegroundColor Green
    }
    catch {
        Write-Host "Test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host ".\wm.ps1 world_models\curriculum_trainer_visual.py --quick True" -ForegroundColor White
