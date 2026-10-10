@echo off
rem ============================================================
rem  Guess Silhouette build script
rem  (ASCII wrapper only; real build logic lives in build.py)
rem    build.bat       single-file exe (default)
rem    build.bat dir   folder mode (faster startup)
rem ============================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)

python build.py %1
if errorlevel 1 (
    echo [ERROR] Build failed. See the log above.
    pause
    exit /b 1
)
pause