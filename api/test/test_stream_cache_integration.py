"""
스트리밍 캐시 통합 테스트
"""
import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.main import app
from api.routers.chat import StreamCache, get_stream_cache
from api.config import get_api_config


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
def mock_chat_service():
    """ChatService 모킹"""
    with patch('api.routers.chat.get_chat_service') as mock_get_service:
        mock_service = MagicMock()
        
        # stream_final_answer 모킹
        async def mock_stream():
            # quota 이벤트
            yield 'data: {"type":"quota","remaining":999,"limit":1000}\n\n'
            
            # stream 이벤트들
            for chunk in ["테스트", " 답변", "입니다"]:
                yield f'data: {{"type":"stream","content":"{chunk}","timestamp":"2024-01-01T00:00:00"}}\n\n'
            
            # final 이벤트
            yield 'data: {"type":"final","content":"테스트 답변입니다","metadata":{"sources":["소스1"],"message_id":"msg123"},"timestamp":"2024-01-01T00:00:00"}\n\n'
            
            # done 이벤트
            yield 'data: {"type":"done","content":"테스트 답변입니다","metadata":{"sources":["소스1"],"message_id":"msg123"},"timestamp":"2024-01-01T00:00:00"}\n\n'
        
        mock_service.stream_final_answer = AsyncMock(return_value=mock_stream())
        mock_get_service.return_value = mock_service
        
        yield mock_service


@pytest.fixture
def enable_cache():
    """캐시 활성화"""
    with patch('api.routers.chat.get_api_config') as mock_config:
        config = mock_config.return_value
        config.enable_stream_cache = True
        config.stream_cache_max_size = 100
        config.stream_cache_ttl_seconds = 3600
        
        # 전역 인스턴스 초기화
        import api.routers.chat as chat_module
        chat_module._stream_cache_instance = None
        
        yield


@pytest.fixture
def disable_cache():
    """캐시 비활성화"""
    with patch('api.routers.chat.get_api_config') as mock_config:
        config = mock_config.return_value
        config.enable_stream_cache = False
        
        # 전역 인스턴스 초기화
        import api.routers.chat as chat_module
        chat_module._stream_cache_instance = None
        
        yield


@pytest.fixture
def mock_auth():
    """인증 모킹"""
    with patch('api.routers.chat.require_auth') as mock_auth:
        mock_auth.return_value = {
            "authenticated": False,
            "user_id": None,
            "quota_remaining": 3
        }
        yield mock_auth


@pytest.fixture
def mock_session_service():
    """SessionService 모킹"""
    with patch('api.routers.chat.session_service') as mock_service:
        mock_service.create_session.return_value = "test_session_123"
        mock_service.add_message.return_value = "msg_123"
        yield mock_service


class TestStreamCacheIntegration:
    """스트리밍 캐시 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_on_second_request(
        self, 
        client, 
        mock_chat_service, 
        enable_cache, 
        mock_auth,
        mock_session_service
    ):
        """두 번째 요청에서 캐시 히트 테스트"""
        message = "캐시 테스트 질문"
        request_data = {
            "message": message,
            "session_id": "test_session"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Anonymous-Session-Id": "test_anon_123"
        }
        
        # 첫 번째 요청 (캐시 미스, LLM 호출)
        response1 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response1.status_code == 200
        assert response1.headers.get("X-Cache") == "MISS"
        
        # 응답 내용 확인
        content1 = response1.text
        assert "테스트 답변입니다" in content1
        
        # 두 번째 요청 (캐시 히트, LLM 호출 없음)
        response2 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response2.status_code == 200
        assert response2.headers.get("X-Cache") == "HIT"
        
        # 응답 내용 확인 (캐시된 내용)
        content2 = response2.text
        assert "테스트 답변입니다" in content2
        
        # ChatService가 한 번만 호출되었는지 확인
        # (첫 번째 요청에서만 호출, 두 번째는 캐시 사용)
        assert mock_chat_service.stream_final_answer.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss_different_sessions(
        self,
        client,
        mock_chat_service,
        enable_cache,
        mock_auth,
        mock_session_service
    ):
        """다른 세션에서는 캐시 미스 테스트"""
        message = "세션별 캐시 테스트"
        
        # 세션1로 요청
        request1 = {
            "message": message,
            "session_id": "session1"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Anonymous-Session-Id": "test_anon_123"
        }
        
        response1 = client.post(
            "/api/v1/chat/stream",
            json=request1,
            headers=headers
        )
        
        assert response1.status_code == 200
        assert response1.headers.get("X-Cache") == "MISS"
        
        # 세션2로 같은 메시지 요청 (캐시 미스 - 세션이 다름)
        request2 = {
            "message": message,
            "session_id": "session2"
        }
        
        response2 = client.post(
            "/api/v1/chat/stream",
            json=request2,
            headers=headers
        )
        
        assert response2.status_code == 200
        # 세션이 다르면 캐시 키가 달라서 미스
        assert response2.headers.get("X-Cache") == "MISS"
    
    @pytest.mark.asyncio
    async def test_cache_disabled(
        self,
        client,
        mock_chat_service,
        disable_cache,
        mock_auth,
        mock_session_service
    ):
        """캐시 비활성화 시 항상 LLM 호출 테스트"""
        message = "캐시 비활성화 테스트"
        request_data = {
            "message": message,
            "session_id": "test_session"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Anonymous-Session-Id": "test_anon_123"
        }
        
        # 첫 번째 요청
        response1 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response1.status_code == 200
        assert response1.headers.get("X-Cache") == "MISS"
        
        # 두 번째 요청 (캐시 비활성화이므로 여전히 LLM 호출)
        response2 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response2.status_code == 200
        assert response2.headers.get("X-Cache") == "MISS"
        
        # 두 번 모두 LLM 호출
        assert mock_chat_service.stream_final_answer.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_with_metadata(
        self,
        client,
        mock_chat_service,
        enable_cache,
        mock_auth,
        mock_session_service
    ):
        """메타데이터가 포함된 캐시 테스트"""
        message = "메타데이터 캐시 테스트"
        request_data = {
            "message": message,
            "session_id": "test_session"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Anonymous-Session-Id": "test_anon_123"
        }
        
        # 첫 번째 요청
        response1 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response1.status_code == 200
        
        # 두 번째 요청 (캐시 히트)
        response2 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response2.status_code == 200
        assert response2.headers.get("X-Cache") == "HIT"
        
        # 캐시된 응답에 메타데이터가 포함되어 있는지 확인
        content2 = response2.text
        # sources 이벤트가 포함되어야 함
        assert "sources" in content2.lower() or "소스1" in content2
    
    @pytest.mark.asyncio
    async def test_cache_quota_event(
        self,
        client,
        mock_chat_service,
        enable_cache,
        mock_auth,
        mock_session_service
    ):
        """캐시된 응답에 quota 이벤트 포함 테스트"""
        message = "쿼터 이벤트 테스트"
        request_data = {
            "message": message,
            "session_id": "test_session"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Anonymous-Session-Id": "test_anon_123"
        }
        
        # 첫 번째 요청
        response1 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response1.status_code == 200
        
        # 두 번째 요청 (캐시 히트)
        response2 = client.post(
            "/api/v1/chat/stream",
            json=request_data,
            headers=headers
        )
        
        assert response2.status_code == 200
        content2 = response2.text
        
        # 캐시된 응답에 quota 이벤트가 포함되어야 함
        assert "quota" in content2.lower() or '"type":"quota"' in content2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

