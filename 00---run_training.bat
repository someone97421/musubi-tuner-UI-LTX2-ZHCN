@echo off
setlocal

set "PYTHON_DIR=python_embed"

if not exist "%PYTHON_DIR%\python.exe" (
    echo [ERROR] Portable Python environment not found!
    echo Please double click install_env.bat to install the environment first.
    pause
    exit /b
)

echo.
echo ===========================================
echo    LTX2 Training Console (Real-time)
echo ===========================================
echo.
echo [INFO] This window will show real-time training logs
echo [INFO] Press Ctrl+C to stop training
echo.

REM 设置环境变量确保实时输出
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM 默认配置文件路径
set "CONFIG_FILE=ltx2_train_config.toml"

REM 检查是否有传入的配置文件参数
if not "%~1"=="" set "CONFIG_FILE=%~1"

echo [INFO] Using config file: %CONFIG_FILE%
echo.

REM 使用 -u 参数强制无缓冲输出，实时显示训练进度
"%PYTHON_DIR%\python.exe" -u ltx2_train_network.py --config_file "%CONFIG_FILE%"

if errorlevel 1 (
    echo.
    echo ===========================================
    echo [ERROR] Training failed! Check logs above.
    echo ===========================================
) else (
    echo.
    echo ===========================================
    echo [SUCCESS] Training completed!
    echo ===========================================
)

echo.
pause
