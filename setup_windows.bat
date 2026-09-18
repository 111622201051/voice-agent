@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Shree Voice Agent - Setup

echo.
echo ================================================================
echo  Shree Voice Agent - one-click setup
echo  Creates .venv, installs all pinned dependencies, checks Ollama.
echo  Requires Python 3.10 - 3.12 (64-bit) already installed.
echo ================================================================
echo.

rem --- 1. Find a supported Python (3.12 / 3.11 / 3.10) ----------
set "PY="
for %%V in (3.12 3.11 3.10) do (
    py -%%V --version >nul 2>&1
    if not errorlevel 1 if not defined PY set "PY=%%V"
)
if not defined PY (
    echo Python 3.10, 3.11 or 3.12 (64-bit) was not found.
    echo.
    echo   1. Download Python 3.12 from:  https://www.python.org/downloads/
    echo      (tick "Add Python to PATH" during install)
    echo      or run:  winget install Python.Python.3.12
    echo   2. Then run setup_windows.bat again.
    echo.
    pause
    exit /b 1
)
echo [1/5] Creating virtual environment with Python %PY% ...
py -%PY% -m venv .venv
if errorlevel 1 exit /b 1
call ".venv\Scripts\activate.bat"

echo [2/5] Upgrading pip ...
python -m pip install --upgrade pip --quiet

echo [3/5] Installing pinned dependencies from requirements-lock.txt ...
echo       (this downloads several hundred MB - please be patient)
python -m pip install -r requirements-lock.txt
if errorlevel 1 (
    echo.
    echo FAILED to install dependencies. Read the error above and retry.
    pause
    exit /b 1
)

echo [4/5] Installing resemblyzer (no-deps - avoids MSVC source build) ...
python -m pip install resemblyzer==0.1.4 --no-deps --no-build-isolation
if errorlevel 1 (
    echo   Warning: resemblyzer failed - voice verification may not work.
)

echo [5/5] Optional local voice cloning engine (F5-TTS, additional download)
set /p CLONE="  Install voice cloning engine now? [y/N] "
if /i "%CLONE%"=="y" (
    python -m pip install -r requirements-voice-clone.txt
)

echo.
echo Checking Ollama LLM model ...
ollama list 2>nul | findstr /C:"qwen2.5:3b" >nul
if errorlevel 1 (
    echo   Warning: Ollama model 'qwen2.5:3b' not found.
    echo   Start Ollama, then run:  ollama pull qwen2.5:3b
) else (
    echo   Ollama model found. Good to go!
)

echo.
echo ================================================================
echo  SETUP COMPLETE. How to run:
echo.
echo   CLI agent :  .venv\Scripts\python main.py
echo   Web UI    :  .venv\Scripts\python -m streamlit run ui/app.py
echo   Test      :  .venv\Scripts\python test_run.py
echo.
echo  First run will ask you to enroll your voice (read 3 sentences).
echo ================================================================
echo.
pause