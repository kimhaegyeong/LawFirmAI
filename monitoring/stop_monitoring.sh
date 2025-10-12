#!/bin/bash
# stop_monitoring.sh

echo "🛑 Stopping Grafana + Prometheus monitoring stack..."

# 현재 디렉토리 확인
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory."
    exit 1
fi

# Docker Compose로 모니터링 스택 중지
echo "📦 Stopping Docker containers..."
docker-compose down

echo "✅ Monitoring stack stopped successfully!"
echo ""
echo "💡 To start the monitoring stack again, run: ./start_monitoring.sh"
