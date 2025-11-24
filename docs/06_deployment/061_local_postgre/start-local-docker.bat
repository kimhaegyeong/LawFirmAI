@echo off
REM 로컬 개발용 PostgreSQL + pgAdmin Docker 시작 스크립트

echo ========================================
echo LawFirmAI Local Docker Services
echo ========================================
echo.

REM 이 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

echo Starting PostgreSQL and pgAdmin...
docker-compose -f docker-compose.local.yml up -d

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Services started successfully!
    echo ========================================
    echo.
    echo PostgreSQL: localhost:5432
    echo   Database: lawfirmai_local
    echo   User: lawfirmai
    echo   Password: local_password
    echo.
    echo pgAdmin: http://localhost:5050
    echo   Email: admin@lawfirmai.local
    echo   Password: admin
    echo.
    echo Connection string:
    echo postgresql://lawfirmai:local_password@localhost:5432/lawfirmai_local
    echo.
    echo To view logs: docker-compose -f docker-compose.local.yml logs -f
    echo To stop: stop-local-docker.bat
    echo.
) else (
    echo.
    echo ========================================
    echo Failed to start services!
    echo ========================================
    echo.
    exit /b 1
)

pause

