@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if defined CONDA_PYTHON_EXE (
    "%CONDA_PYTHON_EXE%" -X utf8 "%~dp0setup_envs.py" %*
) else (
    python -X utf8 "%~dp0setup_envs.py" %*
)
exit /b %errorlevel%
