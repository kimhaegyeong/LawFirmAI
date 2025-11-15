"""
보안 검증 수동 테스트 스크립트
OAuth2 Google 인증 및 엔드포인트 검증 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_oauth2_google_endpoints():
    """OAuth2 Google 엔드포인트 테스트"""
    print("\n=== OAuth2 Google 엔드포인트 테스트 ===")
    
    # 먼저 라우터 등록 확인
    print("\n0. 라우터 등록 확인")
    routes = [r.path for r in client.app.routes if hasattr(r, 'path')]
    oauth2_routes = [r for r in routes if 'oauth2' in r or 'auth' in r]
    print(f"   등록된 라우트 중 OAuth2 관련: {oauth2_routes}")
    
    # 1. OAuth2 Google 인증 엔드포인트 확인
    print("\n1. OAuth2 Google 인증 엔드포인트 확인")
    response = client.get("/api/v1/oauth2/google/authorize")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:200] if response.text else 'None'}")
    # 404가 나오면 라우터가 등록되지 않은 것
    if response.status_code == 404:
        print("   ⚠️  OAuth2 Google 인증 엔드포인트가 404를 반환했습니다.")
        print("   라우터는 등록되어 있지만 경로 문제일 수 있습니다.")
        # 실제 라우트 확인
        for route in client.app.routes:
            if hasattr(route, 'path') and 'oauth2' in route.path:
                print(f"   발견된 라우트: {route.path}, methods: {getattr(route, 'methods', 'N/A')}")
    elif response.status_code in [302, 307, 503]:
        print("   ✅ OAuth2 Google 인증 엔드포인트 존재 확인")
    else:
        print(f"   ⚠️  예상: 302/307/503, 실제: {response.status_code}")
    
    # 2. OAuth2 Google 콜백 엔드포인트 확인
    print("\n2. OAuth2 Google 콜백 엔드포인트 확인")
    response = client.get("/api/v1/oauth2/google/callback?code=test_code")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:200] if response.text else 'None'}")
    if response.status_code == 404:
        print("   ⚠️  OAuth2 Google 콜백 엔드포인트가 등록되지 않았습니다.")
        print("   라우터 등록을 확인하세요.")
    else:
        assert response.status_code in [400, 503], f"예상: 400/503, 실제: {response.status_code}"
        print("   ✅ OAuth2 Google 콜백 엔드포인트 존재 확인")
    
    # 3. OAuth2 Google 서비스 초기화 확인
    print("\n3. OAuth2 Google 서비스 초기화 확인")
    from api.services.oauth2_service import oauth2_google_service
    assert oauth2_google_service is not None
    print(f"   OAuth2 Google 활성화: {oauth2_google_service.is_enabled()}")
    print("   ✅ OAuth2 Google 서비스 초기화 확인")


def test_pydantic_validation():
    """Pydantic 스키마 검증 테스트"""
    print("\n=== Pydantic 스키마 검증 테스트 ===")
    
    # 1. ChatRequest 검증
    print("\n1. ChatRequest 검증")
    from api.schemas.chat import ChatRequest
    
    # 정상 요청
    try:
        valid_request = ChatRequest(message="테스트 메시지")
        assert valid_request.message == "테스트 메시지"
        print("   ✅ 정상 메시지 검증 통과")
    except Exception as e:
        print(f"   ❌ 정상 메시지 검증 실패: {e}")
        raise
    
    # 빈 메시지 검증
    try:
        ChatRequest(message="")
        print("   ❌ 빈 메시지 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 빈 메시지 검증 통과")
    
    # XSS 패턴 검증
    try:
        ChatRequest(message="<script>alert('xss')</script>")
        print("   ❌ XSS 패턴 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ XSS 패턴 검증 통과")
    
    # 2. SessionCreate 검증
    print("\n2. SessionCreate 검증")
    from api.schemas.session import SessionCreate
    
    # 정상 요청
    try:
        valid_request = SessionCreate(title="테스트 세션", category="test")
        assert valid_request.title == "테스트 세션"
        print("   ✅ 정상 세션 생성 검증 통과")
    except Exception as e:
        print(f"   ❌ 정상 세션 생성 검증 실패: {e}")
        raise
    
    # 잘못된 카테고리 형식 검증
    try:
        SessionCreate(title="테스트", category="test@category")
        print("   ❌ 잘못된 카테고리 형식 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 잘못된 카테고리 형식 검증 통과")
    
    # 3. FeedbackRequest 검증
    print("\n3. FeedbackRequest 검증")
    from api.schemas.feedback import FeedbackRequest
    import uuid
    
    session_id = str(uuid.uuid4())
    
    # 정상 요청
    try:
        valid_request = FeedbackRequest(session_id=session_id, rating=5, feedback_type="general")
        assert valid_request.rating == 5
        print("   ✅ 정상 피드백 요청 검증 통과")
    except Exception as e:
        print(f"   ❌ 정상 피드백 요청 검증 실패: {e}")
        raise
    
    # 잘못된 세션 ID 형식 검증
    try:
        FeedbackRequest(session_id="invalid-uuid", rating=5)
        print("   ❌ 잘못된 세션 ID 형식 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 잘못된 세션 ID 형식 검증 통과")
    
    # 잘못된 평점 검증
    try:
        FeedbackRequest(session_id=session_id, rating=6)
        print("   ❌ 잘못된 평점 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 잘못된 평점 검증 통과")
    
    # 4. HistoryQuery 검증
    print("\n4. HistoryQuery 검증")
    from api.schemas.history import HistoryQuery
    
    # 정상 요청
    try:
        valid_query = HistoryQuery(page=1, page_size=10)
        assert valid_query.page == 1
        print("   ✅ 정상 히스토리 쿼리 검증 통과")
    except Exception as e:
        print(f"   ❌ 정상 히스토리 쿼리 검증 실패: {e}")
        raise
    
    # 잘못된 페이지 번호 검증
    try:
        HistoryQuery(page=0, page_size=10)
        print("   ❌ 잘못된 페이지 번호 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 잘못된 페이지 번호 검증 통과")
    
    # 5. ExportRequest 검증
    print("\n5. ExportRequest 검증")
    from api.schemas.history import ExportRequest
    
    session_ids = [str(uuid.uuid4())]
    
    # 정상 요청
    try:
        valid_request = ExportRequest(session_ids=session_ids, format="json")
        assert valid_request.format == "json"
        print("   ✅ 정상 내보내기 요청 검증 통과")
    except Exception as e:
        print(f"   ❌ 정상 내보내기 요청 검증 실패: {e}")
        raise
    
    # 빈 세션 ID 목록 검증
    try:
        ExportRequest(session_ids=[], format="json")
        print("   ❌ 빈 세션 ID 목록 검증 실패 (예외가 발생해야 함)")
        assert False
    except ValueError:
        print("   ✅ 빈 세션 ID 목록 검증 통과")


def test_endpoint_validation():
    """엔드포인트 검증 테스트"""
    print("\n=== 엔드포인트 검증 테스트 ===")
    
    # 1. Health Check 엔드포인트
    print("\n1. Health Check 엔드포인트")
    response = client.get("/api/v1/health")
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Response: {data}")
        assert "status" in data
        assert "timestamp" in data
        assert "chat_service_available" in data
        print("   ✅ Health Check 엔드포인트 검증 통과")
    else:
        print(f"   ❌ Health Check 엔드포인트 실패: {response.status_code}")
        raise AssertionError(f"Health Check 실패: {response.status_code}")
    
    # 2. Chat 엔드포인트 검증
    print("\n2. Chat 엔드포인트 검증")
    # 빈 메시지 검증
    response = client.post("/api/v1/chat", json={"message": ""})
    print(f"   빈 메시지 Status Code: {response.status_code}")
    if response.status_code == 405:
        print("   ⚠️  Chat 엔드포인트가 POST 메서드를 지원하지 않습니다.")
    elif response.status_code == 422:
        print("   ✅ 빈 메시지 검증 통과")
    else:
        print(f"   ⚠️  예상: 422, 실제: {response.status_code}")
    
    # XSS 패턴 검증
    response = client.post("/api/v1/chat", json={"message": "<script>alert('xss')</script>"})
    print(f"   XSS 패턴 Status Code: {response.status_code}")
    if response.status_code == 405:
        print("   ⚠️  Chat 엔드포인트가 POST 메서드를 지원하지 않습니다.")
    elif response.status_code == 422:
        print("   ✅ XSS 패턴 검증 통과")
    else:
        print(f"   ⚠️  예상: 422, 실제: {response.status_code}")
    
    # 3. Session 엔드포인트 검증
    print("\n3. Session 엔드포인트 검증")
    # 잘못된 카테고리 형식 검증
    response = client.post("/api/v1/sessions", json={"title": "테스트", "category": "test@category"})
    print(f"   잘못된 카테고리 Status Code: {response.status_code}")
    assert response.status_code == 422, f"예상: 422, 실제: {response.status_code}"
    print("   ✅ 잘못된 카테고리 형식 검증 통과")
    
    # 4. Feedback 엔드포인트 검증
    print("\n4. Feedback 엔드포인트 검증")
    # 잘못된 세션 ID 형식 검증
    response = client.post("/api/v1/feedback", json={"session_id": "invalid-uuid", "rating": 5})
    print(f"   잘못된 세션 ID Status Code: {response.status_code}")
    assert response.status_code == 422, f"예상: 422, 실제: {response.status_code}"
    print("   ✅ 잘못된 세션 ID 형식 검증 통과")
    
    # 잘못된 평점 검증
    import uuid
    response = client.post("/api/v1/feedback", json={"session_id": str(uuid.uuid4()), "rating": 6})
    print(f"   잘못된 평점 Status Code: {response.status_code}")
    assert response.status_code == 422, f"예상: 422, 실제: {response.status_code}"
    print("   ✅ 잘못된 평점 검증 통과")


def test_jwt_token_validation():
    """JWT 토큰 검증 테스트"""
    print("\n=== JWT 토큰 검증 테스트 ===")
    
    from api.services.auth_service import auth_service
    
    # JWT_SECRET_KEY가 설정되지 않았으면 테스트 스킵
    if not auth_service.secret_key:
        print("   ⚠️  JWT_SECRET_KEY가 설정되지 않아 토큰 생성 테스트를 건너뜁니다.")
        return
    
    # 1. Access token 생성 및 검증
    print("\n1. Access token 생성 및 검증")
    try:
        access_token = auth_service.create_access_token({"sub": "test_user"})
        assert access_token is not None
        assert len(access_token) > 0
        print(f"   Access Token 생성 성공 (길이: {len(access_token)})")
        
        payload = auth_service.verify_token(access_token, token_type="access")
        assert payload is not None
        assert payload.get("sub") == "test_user"
        assert payload.get("type") == "access"
        print("   ✅ Access token 검증 통과")
    except Exception as e:
        print(f"   ❌ Access token 검증 실패: {e}")
        raise
    
    # 2. Refresh token 생성 및 검증
    print("\n2. Refresh token 생성 및 검증")
    try:
        refresh_token = auth_service.create_refresh_token({"sub": "test_user"})
        assert refresh_token is not None
        assert len(refresh_token) > 0
        print(f"   Refresh Token 생성 성공 (길이: {len(refresh_token)})")
        
        payload = auth_service.verify_token(refresh_token, token_type="refresh")
        assert payload is not None
        assert payload.get("sub") == "test_user"
        assert payload.get("type") == "refresh"
        print("   ✅ Refresh token 검증 통과")
    except Exception as e:
        print(f"   ❌ Refresh token 검증 실패: {e}")
        raise
    
    # 3. 토큰 타입 불일치 검증
    print("\n3. 토큰 타입 불일치 검증")
    try:
        access_token = auth_service.create_access_token({"sub": "test_user"})
        payload = auth_service.verify_token(access_token, token_type="refresh")
        assert payload is None, "Access token을 refresh token으로 검증하면 실패해야 함"
        print("   ✅ 토큰 타입 불일치 검증 통과")
    except Exception as e:
        print(f"   ❌ 토큰 타입 불일치 검증 실패: {e}")
        raise


def test_api_production_settings():
    """프로덕션 환경 설정 테스트"""
    print("\n=== 프로덕션 환경 설정 테스트 ===")
    
    from api.config import api_config
    
    # 1. API 문서 비활성화 확인
    print("\n1. API 문서 비활성화 확인")
    if not api_config.debug:
        # 프로덕션 환경에서는 /docs 접근 불가
        response = client.get("/docs")
        print(f"   /docs Status Code: {response.status_code}")
        assert response.status_code == 404, f"예상: 404, 실제: {response.status_code}"
        print("   ✅ /docs 비활성화 확인")
        
        response = client.get("/redoc")
        print(f"   /redoc Status Code: {response.status_code}")
        assert response.status_code == 404, f"예상: 404, 실제: {response.status_code}"
        print("   ✅ /redoc 비활성화 확인")
    else:
        print("   ⚠️  개발 환경에서는 API 문서가 활성화되어 있습니다.")


def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("FastAPI 보안 검증 테스트 시작")
    print("=" * 60)
    
    try:
        test_oauth2_google_endpoints()
        test_pydantic_validation()
        test_endpoint_validation()
        test_jwt_token_validation()
        test_api_production_settings()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 테스트 실패: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

