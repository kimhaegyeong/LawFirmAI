@echo off
REM start_monitoring.bat
REM Windows 배치 파일

echo 🚀 Starting Grafana + Prometheus monitoring stack...

REM 현재 디렉토리 확인
if not exist "docker-compose.yml" (
    echo ❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory.
    pause
    exit /b 1
)

REM Docker Compose로 모니터링 스택 시작
echo 📦 Starting Docker containers...
docker-compose up -d

REM 서비스가 시작될 때까지 대기
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak > nul

REM 서비스 상태 확인
echo 🔍 Checking service status...

REM Prometheus 상태 확인
curl -s http://localhost:9090/-/healthy > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus is running on http://localhost:9090
) else (
    echo ❌ Prometheus failed to start
)

REM Grafana 상태 확인
curl -s http://localhost:3001/api/health > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Grafana is running on http://localhost:3001
) else (
    echo ❌ Grafana failed to start
)

REM Node Exporter 상태 확인
curl -s http://localhost:9100/metrics > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node Exporter is running on http://localhost:9100
) else (
    echo ❌ Node Exporter failed to start
)

echo.
echo 🎉 Monitoring stack started successfully!
echo.
echo 📊 Access URLs:
echo    Grafana: http://localhost:3001 (admin/admin123)
echo    Prometheus: http://localhost:9090
echo    Node Exporter: http://localhost:9100
echo.
echo 📈 Metrics endpoint: http://localhost:8000/metrics
echo.
echo 💡 To stop the monitoring stack, run: stop_monitoring.bat
pause
