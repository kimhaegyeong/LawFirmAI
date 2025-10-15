# start_monitoring.ps1
# Windows PowerShell 스크립트

Write-Host "🚀 Starting Grafana + Prometheus monitoring stack..." -ForegroundColor Green

# 현재 디렉토리 확인
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory." -ForegroundColor Red
    exit 1
}

# Docker Compose로 모니터링 스택 시작
Write-Host "📦 Starting Docker containers..." -ForegroundColor Yellow
docker-compose up -d

# 서비스가 시작될 때까지 대기
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 서비스 상태 확인
Write-Host "🔍 Checking service status..." -ForegroundColor Yellow

# Prometheus 상태 확인
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Prometheus is running on http://localhost:9090" -ForegroundColor Green
} catch {
    Write-Host "❌ Prometheus failed to start" -ForegroundColor Red
}

# Grafana 상태 확인
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Grafana is running on http://localhost:3000" -ForegroundColor Green
} catch {
    Write-Host "❌ Grafana failed to start" -ForegroundColor Red
}

# Node Exporter 상태 확인
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9100/metrics" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Node Exporter is running on http://localhost:9100" -ForegroundColor Green
} catch {
    Write-Host "❌ Node Exporter failed to start" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Monitoring stack started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Access URLs:" -ForegroundColor Cyan
Write-Host "   Grafana: http://localhost:3000 (admin/admin123)" -ForegroundColor White
Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "   Node Exporter: http://localhost:9100" -ForegroundColor White
Write-Host ""
Write-Host "📈 Metrics endpoint: http://localhost:8000/metrics" -ForegroundColor White
Write-Host ""
Write-Host "💡 To stop the monitoring stack, run: .\stop_monitoring.ps1" -ForegroundColor Yellow
