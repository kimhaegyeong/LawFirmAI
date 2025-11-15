"""
보안 검증 테스트
OAuth2 Google 인증 및 엔드포인트 검증 테스트
"""
import pytest
import httpx
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

client = TestClient(app)


class TestOAuth2Google:
    """OAuth2 Google 인증 테스트"""
    
    def test_oauth2_google_authorize_endpoint_exists(self):
        """OAuth2 Google 인증 엔드포인트 존재 확인"""
        response = client.get("/api/v1/oauth2/google/authorize")
        # OAuth2가 활성화되지 않았으면 503, 활성화되었으면 리다이렉트
        assert response.status_code in [302, 307, 503]
    
    def test_oauth2_google_callback_endpoint_exists(self):
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


class TestPydanticValidation:
    """Pydantic 스키마 검증 테스트"""
    
    def test_chat_request_validation(self):
        """ChatRequest 스키마 검증 테스트"""
        from api.schemas.chat import ChatRequest
        
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
        from api.schemas.session import SessionCreate
        
        # 정상 요청
        valid_request = SessionCreate(title="테스트 세션", category="test")
        assert valid_request.title == "테스트 세션"
        assert valid_request.category == "test"
        
        # 빈 제목 검증
        with pytest.raises(ValueError):
            SessionCreate(title="   ", category="test")
        
        # 너무 긴 제목 검증
        with pytest.raises(ValueError):
            SessionCreate(title="a" * 256, category="test")
        
        # 잘못된 카테고리 형식 검증
        with pytest.raises(ValueError):
            SessionCreate(title="테스트", category="test@category")
    
    def test_feedback_request_validation(self):
        """FeedbackRequest 스키마 검증 테스트"""
        from api.schemas.feedback import FeedbackRequest
        import uuid
        
        session_id = str(uuid.uuid4())
        
        # 정상 요청
        valid_request = FeedbackRequest(
            session_id=session_id,
            rating=5,
            feedback_type="general"
        )
        assert valid_request.rating == 5
        
        # 잘못된 세션 ID 형식 검증
        with pytest.raises(ValueError):
            FeedbackRequest(
                session_id="invalid-uuid",
                rating=5
            )
        
        # 잘못된 평점 검증
        with pytest.raises(ValueError):
            FeedbackRequest(
                session_id=session_id,
                rating=6
            )
        
        # 잘못된 피드백 유형 검증
        with pytest.raises(ValueError):
            FeedbackRequest(
                session_id=session_id,
                rating=5,
                feedback_type="invalid_type"
            )
    
    def test_history_query_validation(self):
        """HistoryQuery 스키마 검증 테스트"""
        from api.schemas.history import HistoryQuery
        
        # 정상 요청
        valid_query = HistoryQuery(page=1, page_size=10)
        assert valid_query.page == 1
        assert valid_query.page_size == 10
        
        # 잘못된 페이지 번호 검증
        with pytest.raises(ValueError):
            HistoryQuery(page=0, page_size=10)
        
        # 잘못된 페이지 크기 검증
        with pytest.raises(ValueError):
            HistoryQuery(page=1, page_size=101)
        
        # 잘못된 정렬 기준 검증
        with pytest.raises(ValueError):
            HistoryQuery(page=1, page_size=10, sort_by="invalid_field")
        
        # 잘못된 정렬 순서 검증
        with pytest.raises(ValueError):
            HistoryQuery(page=1, page_size=10, sort_order="invalid")
    
    def test_export_request_validation(self):
        """ExportRequest 스키마 검증 테스트"""
        from api.schemas.history import ExportRequest
        import uuid
        
        session_ids = [str(uuid.uuid4())]
        
        # 정상 요청
        valid_request = ExportRequest(session_ids=session_ids, format="json")
        assert valid_request.format == "json"
        
        # 빈 세션 ID 목록 검증
        with pytest.raises(ValueError):
            ExportRequest(session_ids=[], format="json")
        
        # 너무 많은 세션 ID 검증
        with pytest.raises(ValueError):
            ExportRequest(session_ids=[str(uuid.uuid4())] * 101, format="json")
        
        # 잘못된 형식 검증
        with pytest.raises(ValueError):
            ExportRequest(session_ids=session_ids, format="xml")


class TestEndpointValidation:
    """엔드포인트 검증 테스트"""
    
    def test_health_endpoint_validation(self):
        """Health Check 엔드포인트 검증"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "chat_service_available" in data
    
    def test_chat_endpoint_validation(self):
        """Chat 엔드포인트 검증"""
        # 빈 메시지 검증
        response = client.post(
            "/api/v1/chat",
            json={"message": ""}
        )
        assert response.status_code == 422
        
        # XSS 패턴 검증
        response = client.post(
            "/api/v1/chat",
            json={"message": "<script>alert('xss')</script>"}
        )
        assert response.status_code == 422
    
    def test_session_endpoint_validation(self):
        """Session 엔드포인트 검증"""
        # 잘못된 카테고리 형식 검증
        response = client.post(
            "/api/v1/sessions",
            json={"title": "테스트", "category": "test@category"}
        )
        assert response.status_code == 422
    
    def test_feedback_endpoint_validation(self):
        """Feedback 엔드포인트 검증"""
        import uuid
        
        # 잘못된 세션 ID 형식 검증
        response = client.post(
            "/api/v1/feedback",
            json={
                "session_id": "invalid-uuid",
                "rating": 5
            }
        )
        assert response.status_code == 422
        
        # 잘못된 평점 검증
        response = client.post(
            "/api/v1/feedback",
            json={
                "session_id": str(uuid.uuid4()),
                "rating": 6
            }
        )
        assert response.status_code == 422


class TestJWTTokenValidation:
    """JWT 토큰 검증 테스트"""
    
    def test_jwt_token_creation(self):
        """JWT 토큰 생성 테스트"""
        from api.services.auth_service import auth_service
        
        # JWT_SECRET_KEY가 설정되지 않았으면 에러 발생
        if not auth_service.secret_key:
            with pytest.raises(ValueError):
                auth_service.create_access_token({"sub": "test"})
        else:
            token = auth_service.create_access_token({"sub": "test"})
            assert token is not None
            assert len(token) > 0
    
    def test_refresh_token_creation(self):
        """Refresh token 생성 테스트"""
        from api.services.auth_service import auth_service
        
        if auth_service.secret_key:
            token = auth_service.create_refresh_token({"sub": "test"})
            assert token is not None
            assert len(token) > 0
    
    def test_token_verification(self):
        """토큰 검증 테스트"""
        from api.services.auth_service import auth_service
        
        if auth_service.secret_key:
            # Access token 생성 및 검증
            access_token = auth_service.create_access_token({"sub": "test"})
            payload = auth_service.verify_token(access_token, token_type="access")
            assert payload is not None
            assert payload.get("sub") == "test"
            assert payload.get("type") == "access"
            
            # Refresh token 생성 및 검증
            refresh_token = auth_service.create_refresh_token({"sub": "test"})
            payload = auth_service.verify_token(refresh_token, token_type="refresh")
            assert payload is not None
            assert payload.get("sub") == "test"
            assert payload.get("type") == "refresh"
            
            # 잘못된 토큰 타입 검증
            payload = auth_service.verify_token(access_token, token_type="refresh")
            assert payload is None


class TestAPIProductionSettings:
    """프로덕션 환경 설정 테스트"""
    
    def test_api_docs_disabled_in_production(self):
        """프로덕션 환경에서 API 문서 비활성화 확인"""
        # debug=False일 때 docs_url이 None인지 확인
        from api.config import api_config
        
        if not api_config.debug:
            # 프로덕션 환경에서는 /docs 접근 불가
            response = client.get("/docs")
            assert response.status_code == 404
            
            response = client.get("/redoc")
            assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

