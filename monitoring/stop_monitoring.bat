@echo off
REM stop_monitoring.bat
REM Windows 배치 파일

echo 🛑 Stopping Grafana + Prometheus monitoring stack...

REM 현재 디렉토리 확인
if not exist "docker-compose.yml" (
    echo ❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory.
    pause
    exit /b 1
)

REM Docker Compose로 모니터링 스택 중지
echo 📦 Stopping Docker containers...
docker-compose down

echo ✅ Monitoring stack stopped successfully!
echo.
echo 💡 To start the monitoring stack again, run: start_monitoring.bat
pause
