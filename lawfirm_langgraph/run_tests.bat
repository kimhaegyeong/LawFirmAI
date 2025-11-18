@echo off
setlocal
cd /d "%~dp0"

REM Check if venv exists in api directory (preferred)
if exist "..\api\venv\Scripts\python.exe" (
    call "..\api\venv\Scripts\activate.bat"
    echo [INFO] Activated api venv
) else if exist "venv\Scripts\python.exe" (
    call "venv\Scripts\activate.bat"
    echo [INFO] Activated local venv
) else (
    echo [WARNING] No virtual environment found. Using system Python.
)

REM Set PYTHONPATH
set "PYTHONPATH=%CD%;%CD%\..;%PYTHONPATH%"

REM Run tests with provided arguments
if "%1"=="" (
    echo [INFO] Running all tests in tests/langgraph_core
    python -m pytest tests/langgraph_core -v
) else (
    python -m pytest %*
)

endlocal

