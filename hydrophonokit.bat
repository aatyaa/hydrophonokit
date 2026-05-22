@echo off
REM HydroPhonoKit Launcher - Sets proper encoding before running
REM This fixes Windows console encoding issues with VASP files

REM Set Python to use UTF-8 encoding
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Run HydroPhonoKit
python -m hydrophonokit %*

REM If no arguments were passed (interactive mode ran and exited), pause
if "%~1"=="" (
    echo.
    echo Press any key to exit ...
    pause >nul
)
