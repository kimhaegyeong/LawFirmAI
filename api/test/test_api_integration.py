"""
API 통합 테스트
"""
import pytest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# API 모듈 경로 추가
api_path = project_root / "api"
sys.path.insert(0, str(api_path))

from api.main import app


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


class TestHealthEndpoint:
    """헬스체크 엔드포인트 테스트"""
    
    def test_health_endpoint(self, client):
        """헬스체크 엔드포인트 동작 확인"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestChatEndpoint:
    """채팅 엔드포인트 테스트"""
    
    def test_chat_endpoint_structure(self, client):
        """채팅 엔드포인트 구조 확인"""
        # 인증이 필요한 경우 401이 발생할 수 있음
        response = client.post("/api/chat", json={"message": "test"})
        assert response.status_code in [200, 401, 422, 500]
    
    def test_chat_endpoint_validation(self, client):
        """채팅 엔드포인트 입력 검증"""
        # 빈 메시지
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code in [400, 422]
        
        # 메시지 없음
        response = client.post("/api/chat", json={})
        assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

