# -*- coding: utf-8 -*-
"""
Auth API 테스트
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from api.main import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

from fastapi.testclient import TestClient


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestAuthAPI:
    """Auth API 테스트 클래스"""
    
    @pytest.fixture
    def client(self):
        """TestClient 픽스처"""
        return TestClient(app)
    
    def test_oauth2_authorize(self, client):
        """OAuth2 인증 시작 테스트"""
        with patch('api.routers.auth.oauth2_google_service.is_enabled', return_value=True):
            with patch('api.routers.auth.oauth2_google_service.get_authorization_url') as mock_get_url:
                mock_get_url.return_value = "https://accounts.google.com/authorize?test=123"
                
                response = client.get("/api/v1/oauth2/google/authorize", follow_redirects=False)
                
                assert response.status_code in [302, 307]
                assert "Location" in response.headers
    
    def test_oauth2_authorize_not_enabled(self, client):
        """OAuth2 비활성화 테스트"""
        with patch('api.routers.auth.oauth2_google_service.is_enabled', return_value=False):
            response = client.get("/api/v1/oauth2/google/authorize")
            
            assert response.status_code == 503
    
    def test_oauth2_callback(self, client):
        """OAuth2 콜백 테스트"""
        with patch('api.routers.auth.oauth2_google_service.is_enabled', return_value=True):
            with patch('api.routers.auth.oauth2_states', {"test_state": True}):
                with patch('api.routers.auth.oauth2_google_service.get_token') as mock_get_token:
                    with patch('api.routers.auth.oauth2_google_service.get_user_info') as mock_get_user:
                        with patch('api.routers.auth.user_service.create_or_update_user'):
                            with patch('api.routers.auth.auth_service.create_access_token') as mock_create_token:
                                with patch('api.routers.auth.auth_service.create_refresh_token'):
                                    mock_get_token.return_value = {"access_token": "test_token"}
                                    mock_get_user.return_value = {
                                        "id": "test_user_id",
                                        "email": "test@example.com",
                                        "name": "Test User",
                                        "picture": "https://example.com/picture.jpg"
                                    }
                                    mock_create_token.return_value = "test_access_token"
                                    
                                    response = client.get(
                                        "/api/v1/oauth2/google/callback",
                                        params={"code": "test_code", "state": "test_state"},
                                        follow_redirects=False
                                    )
                                    
                                    assert response.status_code in [302, 307]
    
    def test_oauth2_callback_invalid_state(self, client):
        """잘못된 state 테스트"""
        with patch('api.routers.auth.oauth2_google_service.is_enabled', return_value=True):
            with patch('api.routers.auth.oauth2_states', {}):
                response = client.get(
                    "/api/v1/oauth2/google/callback",
                    params={"code": "test_code", "state": "invalid_state"}
                )
                
                assert response.status_code == 400
    
    def test_refresh_token(self, client):
        """토큰 갱신 테스트"""
        with patch('api.routers.auth.auth_service.secret_key', "test_secret_key"):
            with patch('api.routers.auth.auth_service.verify_token') as mock_verify:
                with patch('api.routers.auth.auth_service.create_access_token') as mock_create_access:
                    with patch('api.routers.auth.auth_service.create_refresh_token') as mock_create_refresh:
                        mock_verify.return_value = {
                            "sub": "test_user_id",
                            "email": "test@example.com",
                            "name": "Test User",
                            "picture": "https://example.com/picture.jpg",
                            "provider": "google"
                        }
                        mock_create_access.return_value = "new_access_token"
                        mock_create_refresh.return_value = "new_refresh_token"
                        
                        response = client.post(
                            "/api/v1/auth/refresh",
                            json={"refresh_token": "valid_refresh_token"}
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        
                        assert "access_token" in data
                        assert "refresh_token" in data
                        assert data.get("access_token") == "new_access_token"
    
    def test_refresh_token_invalid(self, client):
        """잘못된 토큰 테스트"""
        with patch('api.routers.auth.auth_service.secret_key', "test_secret_key"):
            with patch('api.routers.auth.auth_service.verify_token', return_value=None):
                response = client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": "invalid_token"}
                )
                
                assert response.status_code == 401
    
    def test_get_current_user_anonymous(self, client):
        """익명 사용자 조회 테스트"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert data.get("user_id") == "anonymous"
        assert data.get("authenticated") == False
    
    def test_get_current_user_authenticated(self, client):
        """인증된 사용자 조회 테스트"""
        with patch('api.routers.auth.get_current_user') as mock_get_user:
            mock_get_user.return_value = {
                "user_id": "test_user_id",
                "email": "test@example.com",
                "name": "Test User",
                "picture": "https://example.com/picture.jpg",
                "provider": "google",
                "authenticated": True
            }
            
            with patch('api.routers.auth.user_service.get_user') as mock_get_user_data:
                mock_get_user_data.return_value = {
                    "user_id": "test_user_id",
                    "email": "test@example.com",
                    "name": "Test User",
                    "picture": "https://example.com/picture.jpg",
                    "provider": "google"
                }
                
                response = client.get(
                    "/api/v1/auth/me",
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data.get("authenticated") == True
                assert data.get("user_id") == "test_user_id"
    
    def test_delete_account(self, client):
        """회원탈퇴 테스트"""
        with patch('api.routers.auth.get_current_user') as mock_get_user:
            mock_get_user.return_value = {
                "user_id": "test_user_id",
                "email": "test@example.com",
                "provider": "google",
                "authenticated": True
            }
            
            with patch('api.routers.auth.oauth2_google_service.is_enabled', return_value=False):
                with patch('api.routers.auth.user_service.delete_user'):
                    with patch('api.routers.auth.session_service.delete_user_sessions', return_value=5):
                        response = client.delete(
                            "/api/v1/auth/account",
                            headers={"Authorization": "Bearer test_token"}
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        
                        assert "message" in data
                        assert "deleted_sessions" in data
    
    def test_delete_account_not_authenticated(self, client):
        """비인증 회원탈퇴 테스트"""
        with patch('api.routers.auth.get_current_user') as mock_get_user:
            mock_get_user.return_value = {
                "authenticated": False
            }
            
            response = client.delete(
                "/api/v1/auth/account",
                headers={"Authorization": "Bearer invalid_token"}
            )
            
            assert response.status_code == 401


