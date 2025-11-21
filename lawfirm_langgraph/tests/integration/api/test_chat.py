# -*- coding: utf-8 -*-
"""
Chat API 테스트
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

try:
    from api.main import app
    from api.services.session_service import session_service
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from fastapi.testclient import TestClient


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestChatAPI:
    """Chat API 테스트 클래스"""
    
    @pytest.fixture
    def client(self):
        """TestClient 픽스처"""
        return TestClient(app)
    
    @pytest.fixture
    def test_session_id(self):
        """테스트 세션 ID 생성"""
        session_id = session_service.create_session(
            title="테스트 세션",
            category="test"
        )
        yield session_id
        try:
            session_service.delete_session(session_id)
        except Exception:
            pass
    
    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService 픽스처"""
        with patch('api.services.chat_service.get_chat_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_available.return_value = True
            mock_service.process_message = AsyncMock(return_value={
                "answer": "테스트 답변입니다",
                "confidence": 0.9,
                "sources": ["민법 제1조"],
                "metadata": {}
            })
            
            async def mock_stream():
                yield '{"type": "stream", "content": "테스트"}'
                yield '{"type": "final", "content": "테스트 답변입니다", "metadata": {}}'
            
            mock_service.stream_message.return_value = mock_stream()
            mock_service.get_sources_from_session = AsyncMock(return_value={
                "sources": ["민법 제1조"],
                "legal_references": ["민법 제1조"],
                "sources_detail": []
            })
            mock_get_service.return_value = mock_service
            yield mock_service
    
    def test_chat_basic(self, client, mock_chat_service):
        """기본 채팅 테스트"""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "테스트 질문입니다",
                "session_id": None,
                "enable_checkpoint": True
            },
            headers={"X-Anonymous-Session-Id": "test_anonymous_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "confidence" in data
        assert isinstance(data.get("sources"), list)
    
    def test_chat_with_session(self, client, mock_chat_service, test_session_id):
        """세션 포함 채팅 테스트"""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "테스트 질문입니다",
                "session_id": test_session_id,
                "enable_checkpoint": True
            },
            headers={"X-Anonymous-Session-Id": "test_anonymous_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert data.get("answer") == "테스트 답변입니다"
    
    def test_chat_validation(self, client):
        """입력 검증 테스트"""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "",
                "session_id": None
            },
            headers={"X-Anonymous-Session-Id": "test_anonymous_123"}
        )
        
        assert response.status_code in [400, 422]
    
    def test_chat_stream_basic(self, client, mock_chat_service):
        """기본 스트리밍 테스트"""
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "message": "테스트 질문입니다",
                "session_id": None
            },
            headers={
                "X-Anonymous-Session-Id": "test_anonymous_123",
                "Accept": "text/event-stream"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"
        
        content = response.text
        assert "data:" in content
    
    def test_chat_stream_events(self, client, mock_chat_service):
        """스트리밍 이벤트 테스트"""
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "message": "테스트 질문입니다",
                "session_id": None
            },
            headers={
                "X-Anonymous-Session-Id": "test_anonymous_123",
                "Accept": "text/event-stream"
            }
        )
        
        assert response.status_code == 200
        
        content = response.text
        lines = content.split("\n")
        
        data_lines = [line for line in lines if line.startswith("data:")]
        assert len(data_lines) > 0
        
        for line in data_lines:
            if line.startswith("data:"):
                json_str = line[5:].strip()
                try:
                    event_data = json.loads(json_str)
                    assert "type" in event_data
                except json.JSONDecodeError:
                    pass
    
    def test_get_sources(self, client, mock_chat_service, test_session_id):
        """sources 조회 테스트"""
        response = client.get(
            f"/api/v1/chat/{test_session_id}/sources",
            headers={"X-Anonymous-Session-Id": "test_anonymous_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sources" in data
        assert "legal_references" in data
        assert "sources_detail" in data
        assert isinstance(data.get("sources"), list)
    
    def test_get_sources_not_found(self, client, mock_chat_service):
        """세션 없음 테스트"""
        response = client.get(
            "/api/v1/chat/non_existent_session/sources",
            headers={"X-Anonymous-Session-Id": "test_anonymous_123"}
        )
        
        assert response.status_code in [200, 404, 503]


