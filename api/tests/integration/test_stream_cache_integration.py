"""
스트리밍 캐시 통합 테스트
"""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from api.routers.chat import StreamCache, get_stream_cache


@pytest.mark.integration
class TestStreamCacheIntegration:
    """스트리밍 캐시 통합 테스트"""
    
    @pytest.fixture
    def client(self):
        """테스트 클라이언트"""
        from api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    @pytest.fixture
    def mock_chat_service(self):
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
    
    def test_stream_cache_integration(self, client, mock_chat_service):
        """스트리밍 캐시 통합 테스트"""
        # 첫 번째 요청
        response1 = client.post(
            "/api/v1/chat/stream",
            json={"message": "테스트 질문", "session_id": "test-session-1"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response1.status_code == 200
        
        # 두 번째 요청 (같은 메시지, 캐시 히트)
        response2 = client.post(
            "/api/v1/chat/stream",
            json={"message": "테스트 질문", "session_id": "test-session-2"},
            headers={"Accept": "text/event-stream"}
        )
        
        assert response2.status_code == 200

