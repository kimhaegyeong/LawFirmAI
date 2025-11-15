@echo off
REM Start server and run streaming test
cd /d "%~dp0\.."
cd /d "%~dp0\..\.."

echo [INFO] Checking if server is running...
python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=2); exit(0 if r.status_code == 200 else 1)" 2>nul
if %errorlevel% == 0 (
    echo [INFO] Server is already running
    goto :run_test
)

echo [INFO] Starting API server in background...
start /min cmd /c "cd /d %~dp0\.. && python main.py"

echo [INFO] Waiting for server to start...
timeout /t 10 /nobreak >nul

:check_server
python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=2); exit(0 if r.status_code == 200 else 1)" 2>nul
if %errorlevel% == 0 (
    echo [INFO] Server is ready
    goto :run_test
)

echo [INFO] Server not ready yet, waiting...
timeout /t 3 /nobreak >nul
goto :check_server

:run_test
echo [INFO] Running streaming test...
python api/test/test_stream_simple.py

pause

