@echo off
REM ===========================================================================
REM  start.bat —— Windows 一键启动脚本
REM  1) 优先使用项目目录下的 .venv 虚拟环境（如果存在）
REM  2) 否则使用系统 Python（需已 pip install -r requirements.txt）
REM  3) 以 UTF-8 代码页运行，保证控制台中文不乱码
REM ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    set "PY=..\.venv\Scripts\python.exe"
    echo [启动] 使用虚拟环境: .venv
) else (
    set "PY=python"
    echo [提示] 未找到 .venv，使用系统 Python。若缺少依赖请先运行:
    echo        python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
)

set PYTHONIOENCODING=utf-8
"%PY%" app.py --open-browser
endlocal
