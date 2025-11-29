#!/bin/bash
# start_monitoring.sh

echo "🚀 Starting Grafana + Prometheus monitoring stack..."

# 현재 디렉토리 확인
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this script from the monitoring directory."
    exit 1
fi

# Docker Compose로 모니터링 스택 시작
echo "📦 Starting Docker containers..."
docker-compose up -d

# 서비스가 시작될 때까지 대기
echo "⏳ Waiting for services to start..."
sleep 10

# 서비스 상태 확인
echo "🔍 Checking service status..."

# Prometheus 상태 확인
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is running on http://localhost:9090"
else
    echo "❌ Prometheus failed to start"
fi

# Grafana 상태 확인
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ Grafana is running on http://localhost:3001"
else
    echo "❌ Grafana failed to start"
fi

# Node Exporter 상태 확인
if curl -s http://localhost:9100/metrics > /dev/null; then
    echo "✅ Node Exporter is running on http://localhost:9100"
else
    echo "❌ Node Exporter failed to start"
fi

echo ""
echo "🎉 Monitoring stack started successfully!"
echo ""
echo "📊 Access URLs:"
echo "   Grafana: http://localhost:3001 (admin/admin123)"
echo "   Prometheus: http://localhost:9090"
echo "   Node Exporter: http://localhost:9100"
echo ""
echo "📈 Metrics endpoint: http://localhost:8000/metrics"
echo ""
echo "💡 To stop the monitoring stack, run: ./stop_monitoring.sh"
