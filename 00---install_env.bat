@echo off
setlocal enabledelayedexpansion

REM 设置环境变量确保实时输出
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

set "PYTHON_VERSION=3.12.10"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_DIR=%~dp0python_embed"
set "PYTHON_ZIP=%~dp0python-%PYTHON_VERSION%-embed-amd64.zip"
set "SCRIPT_DIR=%~dp0"

echo ===========================================
echo  musubi-tuner LTX2 - Portable Installer
echo  Python %PYTHON_VERSION% + Auto CUDA Detection
echo ===========================================

if not exist "%PYTHON_DIR%" (
    echo [1/7] Downloading Python %PYTHON_VERSION%...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"
    if errorlevel 1 goto :fail

    echo [2/7] Extracting Python...
    powershell -NoProfile -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
    if errorlevel 1 goto :fail
    del "%PYTHON_ZIP%"
) else (
    echo [!] python_embed already exists, skipping download.
)

echo [3/7] Enabling site-packages...
powershell -NoProfile -Command "(Get-Content '%PYTHON_DIR%\python312._pth') -replace '#import site','import site' | Set-Content '%PYTHON_DIR%\python312._pth'"

if not exist "%PYTHON_DIR%\Scripts\pip.exe" (
    echo [4/7] Installing pip...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%~dp0get-pip.py'"
    if errorlevel 1 goto :fail
    "%PYTHON_DIR%\python.exe" "%~dp0get-pip.py"
    if errorlevel 1 goto :fail
    del "%~dp0get-pip.py"
) else (
    echo [!] pip already installed.
)

echo [5/7] Installing build tools...
"%PYTHON_DIR%\python.exe" -m pip install --upgrade pip hatchling editables
if errorlevel 1 goto :fail

echo [6/7] Detecting CUDA version (using Python)...

set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_SUFFIX=cu128"
set "TXTTMP=%TEMP%\cuda_result_%RANDOM%.txt"

"%PYTHON_DIR%\python.exe" -c "import subprocess,re,sys; r=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10); m=re.search(r'CUDA Version: *(\d+)\.(\d+)',r.stdout); maj=int(m.group(1)) if m else 0; minor=int(m.group(2)) if m else 0; s='cu130' if maj>=13 else ('cu128' if maj>=11 else 'NOGPU'); open(sys.argv[1],'w').write(s)" "%TXTTMP%" 2>nul

if errorlevel 1 (
    echo [!] Python detection failed. Defaulting to cu128.
    goto :install_torch
)

set "DETECTED="
for /f "usebackq tokens=*" %%A in ("%TXTTMP%") do set "DETECTED=%%A"
del "%TXTTMP%" 2>nul

if "%DETECTED%"=="NOGPU" (
    echo [!] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    set "TORCH_INDEX="
    goto :install_torch
)
if not "%DETECTED%"=="" (
    set "TORCH_SUFFIX=%DETECTED%"
    set "TORCH_INDEX=https://download.pytorch.org/whl/%DETECTED%"
)
echo Detected PyTorch target: %TORCH_SUFFIX%

:install_torch
echo [7/7] Installing PyTorch and dependencies...

set "TORCH_VER=2.8.0"
set "VISION_VER=0.23.0"
set "AUDIO_VER=2.8.0"

if "%TORCH_SUFFIX%"=="cu130" (
    set "TORCH_VER=2.9.0"
    set "VISION_VER=0.24.0"
    set "AUDIO_VER=2.9.0"
)

if defined TORCH_INDEX (
    echo Installing torch with %TORCH_SUFFIX% ...
    "%PYTHON_DIR%\python.exe" -m pip install torch==%TORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER% --force-reinstall --index-url %TORCH_INDEX%
    if errorlevel 1 goto :fail
) else (
    "%PYTHON_DIR%\python.exe" -m pip install torch==%TORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER%
    if errorlevel 1 goto :fail
)

echo Installing musubi-tuner [gui] dependencies...
"%PYTHON_DIR%\python.exe" -m pip install "%SCRIPT_DIR%.[gui]"
if errorlevel 1 goto :fail

echo Fixing pillow for Gradio compatibility...
"%PYTHON_DIR%\python.exe" -m pip install "pillow>=8.0,<12.0"

echo Installing TensorBoard...
"%PYTHON_DIR%\python.exe" -m pip install tensorboard

echo.
echo ===========================================
echo  Installation Complete!
if defined TORCH_SUFFIX echo  PyTorch: %TORCH_SUFFIX%
echo  Run: run_ui_portable.bat to start the UI
echo ===========================================
pause
exit /b

:fail
echo.
echo ===========================================
echo  ERROR: Step failed. See output above.
echo ===========================================
pause
exit /b
