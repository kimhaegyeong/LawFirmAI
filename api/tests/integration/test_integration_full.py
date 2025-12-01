"""
전체 통합 테스트 스크립트
실제 서버 실행 후 HTTP 요청으로 테스트
"""
import pytest
import requests
from api.tests.helpers.server_helpers import wait_for_server, check_server_health

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


@pytest.mark.integration
@pytest.mark.slow
class TestIntegrationFull:
    """전체 통합 테스트"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self):
        """서버가 실행 중인지 확인"""
        if not check_server_health():
            pytest.skip("서버가 실행 중이지 않습니다. 서버를 시작한 후 테스트를 실행하세요.")
    
    def test_health_endpoint(self):
        """Health Check 엔드포인트 테스트"""
        response = requests.get(f"{API_BASE}/health", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "chat_service_available" in data
        assert data["status"] == "healthy"
    
    def test_oauth2_google_endpoints(self):
        """OAuth2 Google 엔드포인트 테스트"""
        # OAuth2 Google 인증 엔드포인트
        response = requests.get(
            f"{API_BASE}/oauth2/google/authorize",
            allow_redirects=False,
            timeout=5
        )
        assert response.status_code in [302, 307, 503]
        
        # OAuth2 Google 콜백 엔드포인트
        response = requests.get(
            f"{API_BASE}/oauth2/google/callback?code=test_code",
            timeout=5
        )
        assert response.status_code in [400, 503]
    
    def test_api_docs_disabled(self):
        """API 문서 비활성화 테스트"""
        # /docs 엔드포인트
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        assert response.status_code == 404
        
        # /redoc 엔드포인트
        response = requests.get(f"{BASE_URL}/redoc", timeout=5)
        assert response.status_code == 404
    
    def test_pydantic_validation(self):
        """Pydantic 스키마 검증 테스트"""
        # Chat 엔드포인트 검증 - 빈 메시지
        response = requests.post(
            f"{API_BASE}/chat",
            json={"message": ""},
            timeout=5
        )
        assert response.status_code in [422, 401]
        
        # Session 엔드포인트 검증 - 잘못된 카테고리 형식
        response = requests.post(
            f"{API_BASE}/sessions",
            json={"title": "테스트", "category": "test@category"},
            timeout=5
        )
        assert response.status_code in [422, 401]
        
        # Feedback 엔드포인트 검증 - 잘못된 세션 ID 형식
        response = requests.post(
            f"{API_BASE}/feedback",
            json={"session_id": "invalid-uuid", "rating": 5},
            timeout=5
        )
        assert response.status_code in [422, 401]

