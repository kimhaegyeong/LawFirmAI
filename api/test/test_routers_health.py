"""
헬스체크 라우터 테스트
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.routers import health


class TestHealthRouter:
    """헬스체크 라우터 테스트"""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """헬스체크 엔드포인트 테스트"""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(health.router)
        
        with patch('api.routers.health.get_chat_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_available.return_value = True
            mock_get_service.return_value = mock_service
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "chat_service_available" in data
            assert data["status"] == "healthy"
            assert data["chat_service_available"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_service_unavailable(self):
        """ChatService 사용 불가 시 헬스체크 테스트"""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(health.router)
        
        with patch('api.routers.health.get_chat_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_available.return_value = False
            mock_get_service.return_value = mock_service
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["chat_service_available"] is False

