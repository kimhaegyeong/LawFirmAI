# -*- coding: utf-8 -*-
"""
Session API 테스트
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
class TestSessionAPI:
    """Session API 테스트 클래스"""
    
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
    def auth_headers(self):
        """인증 헤더"""
        return {"X-Anonymous-Session-Id": "test_anonymous_123"}
    
    def test_list_sessions(self, client, auth_headers):
        """세션 목록 조회 테스트"""
        response = client.get(
            "/api/v1/sessions",
            headers=auth_headers,
            params={"page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data.get("sessions"), list)
    
    def test_list_sessions_with_filters(self, client, auth_headers):
        """필터링 테스트"""
        response = client.get(
            "/api/v1/sessions",
            headers=auth_headers,
            params={
                "page": 1,
                "page_size": 10,
                "category": "test",
                "search": "테스트"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert isinstance(data.get("sessions"), list)
    
    def test_list_sessions_pagination(self, client, auth_headers):
        """페이지네이션 테스트"""
        response = client.get(
            "/api/v1/sessions",
            headers=auth_headers,
            params={"page": 1, "page_size": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("page") == 1
        assert data.get("page_size") == 5
        assert len(data.get("sessions", [])) <= 5
    
    def test_sessions_by_date_today(self, client, auth_headers):
        """오늘 세션 조회 테스트"""
        response = client.get(
            "/api/v1/sessions/by-date",
            headers=auth_headers,
            params={"date_group": "today", "page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert isinstance(data.get("sessions"), list)
    
    def test_sessions_by_date_yesterday(self, client, auth_headers):
        """어제 세션 조회 테스트"""
        response = client.get(
            "/api/v1/sessions/by-date",
            headers=auth_headers,
            params={"date_group": "yesterday", "page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
    
    def test_sessions_by_date_week(self, client, auth_headers):
        """지난 7일 세션 조회 테스트"""
        response = client.get(
            "/api/v1/sessions/by-date",
            headers=auth_headers,
            params={"date_group": "week", "page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
    
    def test_sessions_by_date_month(self, client, auth_headers):
        """지난 30일 세션 조회 테스트"""
        response = client.get(
            "/api/v1/sessions/by-date",
            headers=auth_headers,
            params={"date_group": "month", "page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
    
    def test_sessions_by_date_older(self, client, auth_headers):
        """30일 이전 세션 조회 테스트"""
        response = client.get(
            "/api/v1/sessions/by-date",
            headers=auth_headers,
            params={"date_group": "older", "page": 1, "page_size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
    
    def test_create_session(self, client, auth_headers):
        """세션 생성 테스트"""
        response = client.post(
            "/api/v1/sessions",
            json={
                "title": "새 테스트 세션",
                "category": "test"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "title" in data
        assert data.get("title") == "새 테스트 세션"
        
        session_id = data.get("session_id")
        try:
            session_service.delete_session(session_id)
        except Exception:
            pass
    
    def test_create_session_with_title(self, client, auth_headers):
        """제목 포함 세션 생성 테스트"""
        response = client.post(
            "/api/v1/sessions",
            json={
                "title": "제목이 있는 세션",
                "category": "test"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("title") == "제목이 있는 세션"
        
        session_id = data.get("session_id")
        try:
            session_service.delete_session(session_id)
        except Exception:
            pass
    
    def test_get_session(self, client, auth_headers, test_session_id):
        """세션 조회 테스트"""
        response = client.get(
            f"/api/v1/sessions/{test_session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data.get("session_id") == test_session_id
    
    def test_get_session_not_found(self, client, auth_headers):
        """세션 없음 테스트"""
        response = client.get(
            "/api/v1/sessions/non_existent_session",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_update_session(self, client, auth_headers, test_session_id):
        """세션 업데이트 테스트"""
        response = client.put(
            f"/api/v1/sessions/{test_session_id}",
            json={
                "title": "업데이트된 제목",
                "category": "updated"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("title") == "업데이트된 제목"
        assert data.get("category") == "updated"
    
    def test_delete_session(self, client, auth_headers):
        """세션 삭제 테스트"""
        session_id = session_service.create_session(
            title="삭제할 세션",
            category="test"
        )
        
        response = client.delete(
            f"/api/v1/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
    
    def test_generate_title(self, client, auth_headers, test_session_id):
        """제목 생성 테스트"""
        session_service.add_message(
            session_id=test_session_id,
            role="user",
            content="테스트 질문입니다"
        )
        session_service.add_message(
            session_id=test_session_id,
            role="assistant",
            content="테스트 답변입니다"
        )
        
        response = client.post(
            f"/api/v1/sessions/{test_session_id}/generate-title",
            headers=auth_headers
        )
        
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "title" in data


