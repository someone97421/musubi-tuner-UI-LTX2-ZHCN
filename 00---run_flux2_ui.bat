@echo off
chcp 65001 >nul
echo ==========================================
echo   FLUX.2 训练 WebUI 启动器
echo ==========================================
echo.

set PYTHON=python_embed\python.exe
if not exist %PYTHON% (
    echo 正在使用系统 Python...
    set PYTHON=python
)

echo 正在启动 FLUX.2 训练界面...
echo 请稍候...
echo.

%PYTHON% flux2_ui_zh.py

pause
