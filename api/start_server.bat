@echo off
setlocal
chcp 65001 >nul 2>&1

echo ========================================
echo LawFirmAI API Server - Simple Test
echo ========================================

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo [INFO] Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate venv
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check Python
python --version
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

REM Check uvicorn
python -c "import uvicorn" 2>nul
if errorlevel 1 (
    echo [ERROR] uvicorn not installed
    echo [INFO] Installing uvicorn...
    python -m pip install uvicorn[standard]
    if errorlevel 1 (
        echo [ERROR] Failed to install uvicorn
        pause
        exit /b 1
    )
)

REM Change to project root
cd /d "%~dp0.."

REM Set PYTHONPATH
set "PYTHONPATH=%CD%;%CD%\lawfirm_langgraph;%PYTHONPATH%"

REM Environment variables (defaults)
if not defined API_HOST set "API_HOST=0.0.0.0"
if not defined API_PORT set "API_PORT=8000"
if not defined LOG_LEVEL set "LOG_LEVEL=debug"
if not defined DEBUG_STREAM set "DEBUG_STREAM=true"

REM Start server
echo [INFO] Starting server...
echo [INFO] Host: %API_HOST%
echo [INFO] Port: %API_PORT%
echo [INFO] Log Level: %LOG_LEVEL%
echo [INFO] Press Ctrl+C to stop
echo ========================================
echo.

python -m uvicorn api.main:app --host %API_HOST% --port %API_PORT% --reload --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] Server failed to start
    pause
    exit /b 1
)
