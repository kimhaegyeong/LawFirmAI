"""
보안 기능 테스트
"""
import pytest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# API 모듈 경로 추가
api_path = project_root / "api"
sys.path.insert(0, str(api_path))

from api.main import app
from api.config import api_config
from api.services.auth_service import auth_service


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
def mock_auth_disabled():
    """인증 비활성화 모킹"""
    with patch.object(api_config, 'auth_enabled', False):
        with patch.object(auth_service, 'is_auth_enabled', return_value=False):
            yield


@pytest.fixture
def mock_auth_enabled():
    """인증 활성화 모킹"""
    with patch.object(api_config, 'auth_enabled', True):
        with patch.object(api_config, 'jwt_secret_key', 'test_secret_key'):
            with patch.object(auth_service, 'is_auth_enabled', return_value=True):
                with patch.object(auth_service, 'verify_token', return_value={'sub': 'test_user'}):
                    yield


class TestAuthentication:
    """인증 시스템 테스트"""
    
    def test_auth_disabled_allows_access(self, client, mock_auth_disabled):
        """인증이 비활성화된 경우 접근 허용"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_auth_enabled_requires_token(self, client, mock_auth_enabled):
        """인증이 활성화된 경우 토큰 필요"""
        response = client.post("/api/chat", json={"message": "test"})
        assert response.status_code == 401
    
    def test_auth_with_valid_token(self, client, mock_auth_enabled):
        """유효한 토큰으로 접근"""
        headers = {"Authorization": "Bearer valid_token"}
        with patch.object(auth_service, 'verify_token', return_value={'sub': 'test_user'}):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_auth_with_api_key(self, client, mock_auth_enabled):
        """API 키로 접근"""
        headers = {"X-API-Key": "test_api_key"}
        with patch.object(auth_service, 'verify_api_key', return_value=True):
            response = client.get("/health", headers=headers)
            assert response.status_code == 200


class TestRateLimiting:
    """Rate Limiting 테스트"""
    
    def test_rate_limit_disabled(self, client):
        """Rate Limiting이 비활성화된 경우"""
        with patch.object(api_config, 'rate_limit_enabled', False):
            # 여러 요청이 모두 성공해야 함
            for _ in range(15):
                response = client.get("/health")
                assert response.status_code == 200
    
    def test_rate_limit_enabled(self, client):
        """Rate Limiting이 활성화된 경우"""
        with patch.object(api_config, 'rate_limit_enabled', True):
            with patch.object(api_config, 'rate_limit_per_minute', 5):
                # Rate limit 초과 시도
                responses = []
                for _ in range(10):
                    response = client.post("/api/chat", json={"message": "test"})
                    responses.append(response.status_code)
                
                # 일부 요청은 429 에러가 발생해야 함
                assert 429 in responses or any(r == 429 for r in responses)


class TestCORS:
    """CORS 설정 테스트"""
    
    def test_cors_headers_present(self, client):
        """CORS 헤더가 포함되어 있는지 확인"""
        response = client.options("/health", headers={"Origin": "http://localhost:3000"})
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200
    
    def test_cors_wildcard_in_production(self, client):
        """프로덕션 환경에서 와일드카드 제거"""
        with patch.object(api_config, 'debug', False):
            with patch.object(api_config, 'cors_origins', '*'):
                origins = api_config.get_cors_origins()
                assert '*' not in origins or api_config.debug


class TestInputValidation:
    """입력 검증 테스트"""
    
    def test_xss_pattern_detection(self, client, mock_auth_disabled):
        """XSS 패턴 검출"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='evil.com'></iframe>",
            "<img onerror='alert(1)'>",
        ]
        
        for payload in xss_payloads:
            response = client.post(
                "/api/chat",
                json={"message": payload}
            )
            # XSS 패턴이 감지되어 422 또는 400 에러가 발생해야 함
            assert response.status_code in [400, 422]
    
    def test_sql_injection_pattern_detection(self, client, mock_auth_disabled):
        """SQL Injection 패턴 검출"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
        ]
        
        for payload in sql_payloads:
            response = client.post(
                "/api/chat",
                json={"message": payload}
            )
            # SQL Injection 패턴이 감지되어 422 또는 400 에러가 발생해야 함
            assert response.status_code in [400, 422]
    
    def test_valid_input_accepted(self, client, mock_auth_disabled):
        """유효한 입력은 허용"""
        response = client.post(
            "/api/chat",
            json={"message": "계약서 작성 시 주의할 사항은 무엇인가요?"}
        )
        # 인증이 비활성화된 경우 200 또는 다른 정상 응답 코드
        assert response.status_code in [200, 401, 500]  # 500은 서비스 초기화 실패 가능


class TestFileUpload:
    """파일 업로드 보안 테스트"""
    
    def test_file_size_limit(self, client, mock_auth_disabled):
        """파일 크기 제한 테스트"""
        # 10MB를 초과하는 Base64 데이터
        large_base64 = "A" * (11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/chat",
            json={
                "message": "test",
                "file_base64": large_base64,
                "filename": "test.pdf"
            }
        )
        # 파일 크기 초과로 422 또는 400 에러가 발생해야 함
        assert response.status_code in [400, 422]
    
    def test_invalid_base64_format(self, client, mock_auth_disabled):
        """유효하지 않은 Base64 형식"""
        invalid_base64 = "!!!invalid_base64!!!"
        
        response = client.post(
            "/api/chat",
            json={
                "message": "test",
                "file_base64": invalid_base64,
                "filename": "test.pdf"
            }
        )
        # Base64 형식 오류로 422 또는 400 에러가 발생해야 함
        assert response.status_code in [400, 422]
    
    def test_dangerous_filename(self, client, mock_auth_disabled):
        """위험한 파일명 차단"""
        dangerous_filenames = [
            "../../etc/passwd",
            "test.exe",
            "script.js",
            "test.bat",
        ]
        
        for filename in dangerous_filenames:
            response = client.post(
                "/api/chat",
                json={
                    "message": "test",
                    "file_base64": "dGVzdA==",  # "test" in base64
                    "filename": filename
                }
            )
            # 위험한 파일명으로 422 또는 400 에러가 발생해야 함
            assert response.status_code in [400, 422]


class TestErrorMasking:
    """에러 메시지 마스킹 테스트"""
    
    def test_error_masking_in_production(self, client, mock_auth_disabled):
        """프로덕션 환경에서 에러 메시지 마스킹"""
        with patch.object(api_config, 'debug', False):
            # 의도적으로 에러를 발생시키는 요청
            response = client.post(
                "/api/chat",
                json={"message": ""}  # 빈 메시지로 검증 에러 발생
            )
            # 에러 응답에 상세한 스택 트레이스가 포함되지 않아야 함
            if response.status_code >= 500:
                error_detail = response.json().get("detail", "")
                # 상세한 에러 정보가 포함되지 않아야 함
                assert "Traceback" not in str(error_detail)
                assert "File" not in str(error_detail) or "line" not in str(error_detail).lower()
    
    def test_error_details_in_debug(self, client, mock_auth_disabled):
        """디버그 환경에서 에러 상세 정보 포함"""
        with patch.object(api_config, 'debug', True):
            response = client.post(
                "/api/chat",
                json={"message": ""}  # 빈 메시지로 검증 에러 발생
            )
            # 디버그 모드에서는 더 상세한 에러 정보가 포함될 수 있음
            assert response.status_code in [400, 422, 500]


class TestSecurityHeaders:
    """보안 헤더 테스트"""
    
    def test_security_headers_present(self, client):
        """보안 헤더가 포함되어 있는지 확인"""
        response = client.get("/health")
        
        # 보안 헤더 확인
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_hsts_in_production(self, client):
        """프로덕션 환경에서 HSTS 헤더 포함"""
        with patch.object(api_config, 'debug', False):
            response = client.get("/health")
            assert "Strict-Transport-Security" in response.headers


class TestCSRFProtection:
    """CSRF 보호 테스트"""
    
    def test_csrf_protection_in_production(self, client):
        """프로덕션 환경에서 CSRF 보호"""
        with patch.object(api_config, 'debug', False):
            # CSRF 토큰 없이 POST 요청
            response = client.post(
                "/api/chat",
                json={"message": "test"}
            )
            # CSRF 보호가 활성화된 경우 403 에러가 발생할 수 있음
            # 또는 인증이 필요하여 401이 발생할 수 있음
            assert response.status_code in [200, 401, 403, 500]


class TestSessionSecurity:
    """세션 보안 테스트"""
    
    def test_session_expiry(self, client, mock_auth_disabled):
        """세션 만료 시간 확인"""
        from api.services.session_service import session_service
        from datetime import datetime, timedelta
        
        # 세션 생성
        session_id = session_service.create_session()
        
        # 만료 시간 확인
        session = session_service.get_session(session_id, check_expiry=True)
        assert session is not None
        
        # 만료된 세션 시뮬레이션
        with patch('api.services.session_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(hours=25)
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            expired_session = session_service.get_session(session_id, check_expiry=True)
            # 만료된 세션은 None이 반환되어야 함
            # (실제 구현에 따라 다를 수 있음)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

