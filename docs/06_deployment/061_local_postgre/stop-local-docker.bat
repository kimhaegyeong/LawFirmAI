@echo off
REM 로컬 개발용 PostgreSQL + pgAdmin Docker 중지 스크립트

echo ========================================
echo Stopping LawFirmAI Local Docker Services
echo ========================================
echo.

REM 이 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

echo Stopping services...
docker-compose -f docker-compose.local.yml stop

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Services stopped successfully!
    echo ========================================
    echo.
    echo To remove volumes (delete all data):
    echo docker-compose -f docker-compose.local.yml down -v
    echo.
) else (
    echo.
    echo ========================================
    echo Failed to stop services!
    echo ========================================
    echo.
    exit /b 1
)

pause

