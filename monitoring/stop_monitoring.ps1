# stop_monitoring.ps1
# Windows PowerShell 스크립트

Write-Host "🛑 Stopping Grafana + Prometheus monitoring stack..." -ForegroundColor Red

# 현재 디렉토리 확인
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory." -ForegroundColor Red
    exit 1
}

# Docker Compose로 모니터링 스택 중지
Write-Host "📦 Stopping Docker containers..." -ForegroundColor Yellow
docker-compose down

Write-Host "✅ Monitoring stack stopped successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 To start the monitoring stack again, run: .\start_monitoring.ps1" -ForegroundColor Yellow
