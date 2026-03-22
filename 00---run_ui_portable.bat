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
echo Starting LTX2 Training WebUI...
echo ===========================================
echo.

REM 设置环境变量确保实时输出
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM 使用 -u 参数强制无缓冲输出，实时显示
"%PYTHON_DIR%\python.exe" -u ltx2_ui_zh.py

echo.
echo ===========================================
echo WebUI closed.
echo ===========================================
pause
