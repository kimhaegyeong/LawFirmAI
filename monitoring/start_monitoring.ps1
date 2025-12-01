# start_monitoring.ps1
# Windows PowerShell script

Write-Host "🚀 Starting Grafana + Prometheus monitoring stack..." -ForegroundColor Green

# Check current directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory." -ForegroundColor Red
    exit 1
}

# Start monitoring stack with Docker Compose
Write-Host "📦 Starting Docker containers..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "🔍 Checking service status..." -ForegroundColor Yellow

# Check Prometheus status
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Prometheus is running on http://localhost:9090" -ForegroundColor Green
} catch {
    Write-Host "❌ Prometheus failed to start" -ForegroundColor Red
}

# Check Grafana status
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3001/api/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Grafana is running on http://localhost:3001" -ForegroundColor Green
} catch {
    Write-Host "❌ Grafana failed to start" -ForegroundColor Red
}

# Check Node Exporter status
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
Write-Host "   Grafana: http://localhost:3001 (admin/admin123)" -ForegroundColor White
Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "   Node Exporter: http://localhost:9100" -ForegroundColor White
Write-Host ""
Write-Host "📈 Metrics endpoint: http://localhost:8000/metrics" -ForegroundColor White
Write-Host ""
Write-Host "💡 To stop the monitoring stack, run: .\stop_monitoring.ps1" -ForegroundColor Yellow
