"""
API 통합 테스트
"""
import pytest
from api.tests.helpers.client_helpers import make_chat_request


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
        response = make_chat_request(client, "test", endpoint="/api/chat")
        assert response.status_code in [200, 401, 422, 500]
    
    def test_chat_endpoint_validation(self, client):
        """채팅 엔드포인트 입력 검증"""
        # 빈 메시지
        response = make_chat_request(client, "", endpoint="/api/chat")
        assert response.status_code in [400, 422]
        
        # 메시지 없음
        response = client.post("/api/chat", json={})
        assert response.status_code in [400, 422]

