@echo off
echo Killing process on port 8000...

REM Find process using port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    set PID=%%a
    echo Found process with PID: %%a
    taskkill /PID %%a /F
    echo Process %%a killed
)

REM Also check for Python processes that might be using port 8000
for /f "tokens=2" %%a in ('tasklist ^| findstr python.exe') do (
    echo Found Python process: %%a
    REM You can uncomment the line below to kill all Python processes
    REM taskkill /PID %%a /F
)

echo Done.
pause

