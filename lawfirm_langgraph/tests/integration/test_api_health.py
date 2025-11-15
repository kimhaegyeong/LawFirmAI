# -*- coding: utf-8 -*-
"""
Health API 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from api.main import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from fastapi.testclient import TestClient


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestHealthAPI:
    """Health API 테스트 클래스"""
    
    @pytest.fixture
    def client(self):
        """TestClient 픽스처"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """헬스체크 테스트"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "chat_service_available" in data
    
    def test_health_check_service_status(self, client):
        """서비스 상태 확인 테스트"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data.get("chat_service_available"), bool)
        assert data.get("status") == "healthy"


