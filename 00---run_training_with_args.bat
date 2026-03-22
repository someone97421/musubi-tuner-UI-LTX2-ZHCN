@echo off
setlocal enabledelayedexpansion

set "PYTHON_DIR=python_embed"

if not exist "%PYTHON_DIR%\python.exe" (
    echo [ERROR] Portable Python environment not found!
    echo Please double click install_env.bat to install the environment first.
    pause
    exit /b 1
)

REM 设置环境变量确保实时输出
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

echo.
echo ===========================================
echo    LTX2 Training (Advanced)
echo ===========================================
echo.

REM 默认参数
set "CONFIG_FILE=ltx2_train_config.toml"
set "CHECKPOINT="
set "MODE=video"

REM 解析参数
:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="--config" set "CONFIG_FILE=%~2" & shift & shift & goto :parse_args
if /i "%~1"=="-c" set "CONFIG_FILE=%~2" & shift & shift & goto :parse_args
if /i "%~1"=="--checkpoint" set "CHECKPOINT=%~2" & shift & shift & goto :parse_args
if /i "%~1"=="--mode" set "MODE=%~2" & shift & shift & goto :parse_args
if /i "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   -c, --config FILE      Config file path (default: ltx2_train_config.toml)
echo   --checkpoint PATH      LTX2 checkpoint path
echo   --mode MODE            Training mode: video, av, audio (default: video)
echo   --help                 Show this help message
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 -c my_config.toml
echo   %~nx0 --checkpoint path\to\model.safetensors --mode av
echo.
pause
exit /b 0

:run
echo [INFO] Config file: %CONFIG_FILE%
echo [INFO] Mode: %MODE%
if not "%CHECKPOINT%"=="" echo [INFO] Checkpoint: %CHECKPOINT%
echo.
echo [INFO] Starting training...
echo [INFO] Press Ctrl+C to stop
echo.

REM 构建命令
set "CMD="%PYTHON_DIR%\python.exe" -u ltx2_train_network.py --config_file "%CONFIG_FILE%" --ltx2_mode %MODE%"

if not "%CHECKPOINT%"=="" (
    set "CMD=!CMD! --ltx2_checkpoint "%CHECKPOINT%""
)

REM 执行并实时显示输出
echo [CMD] !CMD!
echo.
!CMD!

if errorlevel 1 (
    echo.
    echo ===========================================
    echo [ERROR] Training failed!
    echo ===========================================
) else (
    echo.
    echo ===========================================
    echo [SUCCESS] Training completed!
    echo ===========================================
)

echo.
pause
