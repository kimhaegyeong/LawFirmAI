@echo off
REM 로컬 개발용 PostgreSQL + pgAdmin Docker 재시작 스크립트

echo ========================================
echo Restarting LawFirmAI Local Docker Services
echo ========================================
echo.

REM 이 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

echo Restarting services...
docker-compose -f docker-compose.local.yml restart

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Services restarted successfully!
    echo ========================================
    echo.
) else (
    echo.
    echo ========================================
    echo Failed to restart services!
    echo ========================================
    echo.
    exit /b 1
)

pause

