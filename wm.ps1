# World Models PowerShell Wrapper (Windows)
# Ensures scripts always run from repository root with correct environment
Param([Parameter(ValueFromRemainingArguments=$true)] [string[]]$Args)

# Get the directory where this script is located (repo root)
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to repository root
Set-Location $Here
Write-Host "Repository root: $Here" -ForegroundColor Cyan

# Use the virtual environment Python if available, otherwise py -3.12
$PythonExe = "$Here\.venv\Scripts\python.exe"
if (Test-Path $PythonExe) {
    Write-Host "Using virtual environment Python" -ForegroundColor Green
    & $PythonExe $Args
} else {
    Write-Host "Using system Python 3.12" -ForegroundColor Yellow
    & py -3.12 $Args
}
