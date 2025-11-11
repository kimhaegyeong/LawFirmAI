# -*- coding: utf-8 -*-
"""
API 테스트용 공통 Fixture 및 유틸리티
"""

import os
import sys
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

try:
    from api.main import app
    API_AVAILABLE = True
except ImportError as e:
    API_AVAILABLE = False
    print(f"API not available: {e}")

try:
    from api.services.session_service import session_service
    SESSION_SERVICE_AVAILABLE = True
except ImportError:
    SESSION_SERVICE_AVAILABLE = False


@pytest.fixture
def test_client():
    """FastAPI TestClient 픽스처"""
    if not API_AVAILABLE:
        pytest.skip("API not available")
    return TestClient(app)


@pytest.fixture
def test_session_id():
    """테스트용 세션 ID 생성"""
    if not SESSION_SERVICE_AVAILABLE:
        return "test_session_123"
    
    session_id = session_service.create_session(
        title="테스트 세션",
        category="test"
    )
    return session_id


@pytest.fixture
def auth_headers():
    """인증 헤더 픽스처 (익명 사용자)"""
    return {
        "X-Anonymous-Session-Id": "test_anonymous_session_123"
    }


@pytest.fixture
def authenticated_headers():
    """인증된 사용자 헤더 픽스처"""
    return {
        "Authorization": "Bearer test_token_123"
    }


@pytest.fixture
def cleanup_test_sessions():
    """테스트 세션 정리 픽스처"""
    test_session_ids = []
    
    def add_session(session_id: str):
        test_session_ids.append(session_id)
    
    yield add_session
    
    # 테스트 후 정리
    if SESSION_SERVICE_AVAILABLE:
        for session_id in test_session_ids:
            try:
                session_service.delete_session(session_id)
            except Exception:
                pass


@pytest.fixture
def mock_chat_service():
    """Mock ChatService 픽스처"""
    with patch('api.services.chat_service.get_chat_service') as mock_get_service:
        mock_service = MagicMock()
        mock_service.is_available.return_value = True
        mock_service.process_message = AsyncMock(return_value={
            "answer": "테스트 답변",
            "confidence": 0.9,
            "sources": ["민법 제1조"],
            "metadata": {}
        })
        mock_service.stream_message = AsyncMock()
        
        async def mock_stream():
            yield '{"type": "stream", "content": "테스트"}'
            yield '{"type": "final", "content": "테스트 답변", "metadata": {}}'
        
        mock_service.stream_message.return_value = mock_stream()
        mock_service.get_sources_from_session = AsyncMock(return_value={
            "sources": ["민법 제1조"],
            "legal_references": ["민법 제1조"],
            "sources_detail": []
        })
        mock_get_service.return_value = mock_service
        yield mock_service


@pytest.fixture
def sample_chat_request():
    """샘플 채팅 요청 데이터"""
    return {
        "message": "테스트 질문입니다",
        "session_id": None,
        "enable_checkpoint": True
    }


@pytest.fixture
def sample_session_data():
    """샘플 세션 데이터"""
    return {
        "title": "테스트 세션",
        "category": "test"
    }


