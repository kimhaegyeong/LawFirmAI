# stop_monitoring.ps1
# Windows PowerShell script

Write-Host "🛑 Stopping Grafana + Prometheus monitoring stack..." -ForegroundColor Red

# Check current directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory." -ForegroundColor Red
    exit 1
}

# Stop monitoring stack with Docker Compose
Write-Host "📦 Stopping Docker containers..." -ForegroundColor Yellow
docker-compose down

Write-Host "✅ Monitoring stack stopped successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 To start the monitoring stack again, run: .\start_monitoring.ps1" -ForegroundColor Yellow
