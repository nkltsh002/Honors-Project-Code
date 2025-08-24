@echo off
rem World Models Batch Wrapper (Windows)
rem Ensures scripts always run from repository root with correct environment

rem Get the directory where this batch file is located (repo root)
set "REPO_ROOT=%~dp0"
set "REPO_ROOT=%REPO_ROOT:~0,-1%"

rem Change to repository root
cd /d "%REPO_ROOT%"
echo Repository root: %REPO_ROOT%

rem Use the virtual environment Python if available, otherwise py -3.12
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    echo Using virtual environment Python
    "%REPO_ROOT%\.venv\Scripts\python.exe" %*
) else (
    echo Using system Python 3.12
    py -3.12 %*
)
