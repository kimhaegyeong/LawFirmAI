"""
세션 스키마 테스트
"""
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse
)


class TestSessionSchemas:
    """세션 스키마 테스트"""
    
    def test_session_create_empty_title(self):
        """빈 제목으로 세션 생성 테스트"""
        session = SessionCreate(title=None)
        assert session.title is None
    
    def test_session_create_with_title(self):
        """제목이 있는 세션 생성 테스트"""
        session = SessionCreate(title="테스트 세션")
        assert session.title == "테스트 세션"
    
    def test_session_create_title_validation_empty_string(self):
        """빈 문자열 제목 검증 테스트"""
        with pytest.raises(Exception):
            SessionCreate(title="   ")
    
    def test_session_create_title_validation_too_long(self):
        """너무 긴 제목 검증 테스트"""
        long_title = "A" * 256
        with pytest.raises(Exception):
            SessionCreate(title=long_title)
    
    def test_session_update(self):
        """세션 업데이트 테스트"""
        session = SessionUpdate(title="업데이트된 제목")
        assert session.title == "업데이트된 제목"
    
    def test_session_response(self):
        """세션 응답 스키마 테스트"""
        response = SessionResponse(
            session_id="test-session-id",
            title="테스트 세션",
            message_count=5
        )
        assert response.session_id == "test-session-id"
        assert response.title == "테스트 세션"
        assert response.message_count == 5
    
    def test_session_list_response(self):
        """세션 목록 응답 스키마 테스트"""
        sessions = [
            SessionResponse(session_id="1", title="세션 1"),
            SessionResponse(session_id="2", title="세션 2")
        ]
        list_response = SessionListResponse(
            sessions=sessions,
            total=2,
            page=1,
            page_size=10
        )
        assert len(list_response.sessions) == 2
        assert list_response.total == 2
        assert list_response.page == 1
        assert list_response.page_size == 10

