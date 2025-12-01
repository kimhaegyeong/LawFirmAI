"""
보안 검증 테스트
OAuth2 Google 인증 및 엔드포인트 검증 테스트
"""
import pytest
from api.schemas.chat import ChatRequest
from api.schemas.session import SessionCreate


@pytest.mark.integration
class TestOAuth2Google:
    """OAuth2 Google 인증 테스트"""
    
    def test_oauth2_google_authorize_endpoint_exists(self, client):
        """OAuth2 Google 인증 엔드포인트 존재 확인"""
        response = client.get("/api/v1/oauth2/google/authorize")
        # OAuth2가 활성화되지 않았으면 503, 활성화되었으면 리다이렉트
        assert response.status_code in [302, 307, 503]
    
    def test_oauth2_google_callback_endpoint_exists(self, client):
        """OAuth2 Google 콜백 엔드포인트 존재 확인"""
        response = client.get("/api/v1/oauth2/google/callback?code=test")
        # OAuth2가 활성화되지 않았으면 503, 활성화되었으면 400 (잘못된 코드)
        assert response.status_code in [400, 503]
    
    def test_oauth2_google_service_initialization(self):
        """OAuth2 Google 서비스 초기화 확인"""
        from api.services.oauth2_service import oauth2_google_service
        assert oauth2_google_service is not None
        # 설정이 없으면 비활성화되어야 함
        if not oauth2_google_service.is_enabled():
            assert oauth2_google_service.client_id == ""
            assert oauth2_google_service.client_secret == ""


@pytest.mark.integration
class TestPydanticValidation:
    """Pydantic 스키마 검증 테스트"""
    
    def test_chat_request_validation(self):
        """ChatRequest 스키마 검증 테스트"""
        # 정상 요청
        valid_request = ChatRequest(message="테스트 메시지")
        assert valid_request.message == "테스트 메시지"
        
        # 빈 메시지 검증
        with pytest.raises(ValueError):
            ChatRequest(message="")
        
        # 너무 긴 메시지 검증
        with pytest.raises(ValueError):
            ChatRequest(message="a" * 10001)
        
        # XSS 패턴 검증
        with pytest.raises(ValueError):
            ChatRequest(message="<script>alert('xss')</script>")
        
        # SQL 인젝션 패턴 검증
        with pytest.raises(ValueError):
            ChatRequest(message="'; DROP TABLE users; --")
    
    def test_session_create_validation(self):
        """SessionCreate 스키마 검증 테스트"""
        # 정상 요청
        valid_request = SessionCreate(title="테스트 세션", category="test")
        assert valid_request.title == "테스트 세션"
        assert valid_request.category == "test"
        
        # 잘못된 카테고리 형식 검증
        with pytest.raises(ValueError):
            SessionCreate(title="테스트", category="test@category")

