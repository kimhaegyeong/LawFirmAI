# -*- coding: utf-8 -*-
"""
History API 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from api.main import app
    from api.services.session_service import session_service
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from fastapi.testclient import TestClient


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestHistoryAPI:
    """History API 테스트 클래스"""
    
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
        
        session_service.add_message(
            session_id=session_id,
            role="user",
            content="테스트 질문"
        )
        session_service.add_message(
            session_id=session_id,
            role="assistant",
            content="테스트 답변"
        )
        
        yield session_id
        
        try:
            session_service.delete_session(session_id)
        except Exception:
            pass
    
    @pytest.fixture
    def auth_headers(self):
        """인증 헤더"""
        return {"X-Anonymous-Session-Id": "test_anonymous_123"}
    
    def test_get_history(self, client, auth_headers):
        """히스토리 조회 테스트"""
        response = client.get(
            "/api/v1/history",
            headers=auth_headers,
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "messages" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data.get("messages"), list)
    
    def test_get_history_with_filters(self, client, auth_headers, test_session_id):
        """필터링 테스트"""
        response = client.get(
            "/api/v1/history",
            headers=auth_headers,
            params={
                "session_id": test_session_id,
                "category": "test",
                "search": "테스트",
                "page": 1,
                "page_size": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "messages" in data
        assert isinstance(data.get("messages"), list)
    
    def test_get_history_pagination(self, client, auth_headers):
        """페이지네이션 테스트"""
        response = client.get(
            "/api/v1/history",
            headers=auth_headers,
            params={"page": 1, "page_size": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("page") == 1
        assert data.get("page_size") == 5
        assert len(data.get("messages", [])) <= 5
    
    def test_export_history_json(self, client, auth_headers, test_session_id):
        """JSON 내보내기 테스트"""
        response = client.post(
            "/api/v1/history/export",
            json={
                "session_ids": [test_session_id],
                "format": "json"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
        assert "Content-Disposition" in response.headers
    
    def test_export_history_txt(self, client, auth_headers, test_session_id):
        """TXT 내보내기 테스트"""
        response = client.post(
            "/api/v1/history/export",
            json={
                "session_ids": [test_session_id],
                "format": "txt"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/plain"
        assert "Content-Disposition" in response.headers
    
    def test_export_history_validation(self, client, auth_headers):
        """검증 테스트"""
        response = client.post(
            "/api/v1/history/export",
            json={
                "session_ids": [],
                "format": "invalid_format"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [400, 422]


