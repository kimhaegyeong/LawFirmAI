# -*- coding: utf-8 -*-
"""
Feedback API 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from api.main import app
    from api.services.session_service import session_service
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from fastapi.testclient import TestClient


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestFeedbackAPI:
    """Feedback API 테스트 클래스"""
    
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
        
        message_id = session_service.add_message(
            session_id=session_id,
            role="user",
            content="테스트 질문"
        )
        
        yield session_id, message_id
        
        try:
            session_service.delete_session(session_id)
        except Exception:
            pass
    
    @pytest.fixture
    def auth_headers(self):
        """인증 헤더"""
        return {"X-Anonymous-Session-Id": "test_anonymous_123"}
    
    def test_submit_feedback(self, client, auth_headers, test_session_id):
        """피드백 제출 테스트"""
        session_id, message_id = test_session_id
        
        response = client.post(
            "/api/v1/feedback",
            json={
                "session_id": session_id,
                "message_id": message_id,
                "rating": 5,
                "comment": "좋은 답변입니다",
                "feedback_type": "positive"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "feedback_id" in data
        assert "session_id" in data
        assert "message_id" in data
        assert "rating" in data
        assert data.get("rating") == 5
        assert data.get("comment") == "좋은 답변입니다"
    
    def test_submit_feedback_validation(self, client, auth_headers):
        """검증 테스트"""
        response = client.post(
            "/api/v1/feedback",
            json={
                "session_id": "invalid_session",
                "message_id": "invalid_message",
                "rating": 10,
                "comment": "",
                "feedback_type": "invalid"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [400, 404, 422]
    
    def test_submit_feedback_not_found(self, client, auth_headers):
        """세션 없음 테스트"""
        response = client.post(
            "/api/v1/feedback",
            json={
                "session_id": "non_existent_session",
                "message_id": "non_existent_message",
                "rating": 5,
                "comment": "테스트",
                "feedback_type": "positive"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 404


