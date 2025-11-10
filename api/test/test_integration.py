"""
통합 테스트 스크립트
실제 서버 실행 후 HTTP 요청으로 테스트
"""
import requests
import time
import subprocess
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def wait_for_server(url: str, max_attempts: int = 30, delay: float = 1.0):
    """서버가 시작될 때까지 대기"""
    print(f"서버 시작 대기 중: {url}")
    for i in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in [200, 404]:
                print(f"✅ 서버가 시작되었습니다. (시도 {i+1}/{max_attempts})")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    print(f"❌ 서버 시작 실패 (최대 시도: {max_attempts})")
    return False

def test_health_endpoint():
    """Health Check 엔드포인트 테스트"""
    print("\n=== Health Check 엔드포인트 테스트 ===")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            assert "status" in data
            assert "timestamp" in data
            assert "chat_service_available" in data
            print("✅ Health Check 엔드포인트 테스트 통과")
            return True
        else:
            print(f"❌ Health Check 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check 오류: {e}")
        return False

def test_oauth2_google_endpoints():
    """OAuth2 Google 엔드포인트 테스트"""
    print("\n=== OAuth2 Google 엔드포인트 테스트 ===")
    
    # 1. OAuth2 Google 인증 엔드포인트
    print("\n1. OAuth2 Google 인증 엔드포인트")
    try:
        response = requests.get(f"{API_BASE}/oauth2/google/authorize", allow_redirects=False, timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code in [302, 307]:
            print(f"Redirect Location: {response.headers.get('Location', 'N/A')}")
            print("✅ OAuth2 Google 인증 엔드포인트 정상 작동")
            return True
        elif response.status_code == 503:
            print("⚠️  OAuth2 Google이 비활성화되어 있습니다.")
            print("   GOOGLE_CLIENT_ID와 GOOGLE_CLIENT_SECRET을 설정하세요.")
            return True
        else:
            print(f"❌ 예상: 302/307/503, 실제: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ OAuth2 Google 인증 엔드포인트 오류: {e}")
        return False
    
    # 2. OAuth2 Google 콜백 엔드포인트
    print("\n2. OAuth2 Google 콜백 엔드포인트")
    try:
        response = requests.get(f"{API_BASE}/oauth2/google/callback?code=test_code", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code in [400, 503]:
            print("✅ OAuth2 Google 콜백 엔드포인트 정상 작동")
            return True
        else:
            print(f"❌ 예상: 400/503, 실제: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ OAuth2 Google 콜백 엔드포인트 오류: {e}")
        return False

def test_api_docs_disabled():
    """API 문서 비활성화 테스트"""
    print("\n=== API 문서 비활성화 테스트 ===")
    
    # 1. /docs 엔드포인트
    print("\n1. /docs 엔드포인트")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 404:
            print("✅ /docs 엔드포인트 비활성화 확인")
            return True
        elif response.status_code == 200:
            print("⚠️  /docs 엔드포인트가 활성화되어 있습니다.")
            print("   프로덕션 환경에서는 DEBUG=false로 설정하세요.")
            return False
        else:
            print(f"⚠️  예상: 404, 실제: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ /docs 엔드포인트 오류: {e}")
        return False
    
    # 2. /redoc 엔드포인트
    print("\n2. /redoc 엔드포인트")
    try:
        response = requests.get(f"{BASE_URL}/redoc", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 404:
            print("✅ /redoc 엔드포인트 비활성화 확인")
            return True
        elif response.status_code == 200:
            print("⚠️  /redoc 엔드포인트가 활성화되어 있습니다.")
            print("   프로덕션 환경에서는 DEBUG=false로 설정하세요.")
            return False
        else:
            print(f"⚠️  예상: 404, 실제: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ /redoc 엔드포인트 오류: {e}")
        return False

def test_pydantic_validation():
    """Pydantic 스키마 검증 테스트 (HTTP 요청)"""
    print("\n=== Pydantic 스키마 검증 테스트 (HTTP) ===")
    
    # 1. Chat 엔드포인트 검증
    print("\n1. Chat 엔드포인트 검증")
    try:
        # 빈 메시지 검증
        response = requests.post(
            f"{API_BASE}/chat",
            json={"message": ""},
            timeout=5
        )
        print(f"빈 메시지 Status Code: {response.status_code}")
        if response.status_code == 422:
            print("✅ 빈 메시지 검증 통과")
        elif response.status_code == 401:
            print("⚠️  인증이 필요합니다. (정상 동작)")
        else:
            print(f"⚠️  예상: 422, 실제: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat 엔드포인트 오류: {e}")
    
    # 2. Session 엔드포인트 검증
    print("\n2. Session 엔드포인트 검증")
    try:
        # 잘못된 카테고리 형식 검증
        response = requests.post(
            f"{API_BASE}/sessions",
            json={"title": "테스트", "category": "test@category"},
            timeout=5
        )
        print(f"잘못된 카테고리 Status Code: {response.status_code}")
        if response.status_code == 422:
            print("✅ 잘못된 카테고리 형식 검증 통과")
        elif response.status_code == 401:
            print("⚠️  인증이 필요합니다. (정상 동작)")
        else:
            print(f"⚠️  예상: 422, 실제: {response.status_code}")
    except Exception as e:
        print(f"❌ Session 엔드포인트 오류: {e}")
    
    # 3. Feedback 엔드포인트 검증
    print("\n3. Feedback 엔드포인트 검증")
    try:
        import uuid
        # 잘못된 세션 ID 형식 검증
        response = requests.post(
            f"{API_BASE}/feedback",
            json={"session_id": "invalid-uuid", "rating": 5},
            timeout=5
        )
        print(f"잘못된 세션 ID Status Code: {response.status_code}")
        if response.status_code == 422:
            print("✅ 잘못된 세션 ID 형식 검증 통과")
        elif response.status_code == 401:
            print("⚠️  인증이 필요합니다. (정상 동작)")
        else:
            print(f"⚠️  예상: 422, 실제: {response.status_code}")
    except Exception as e:
        print(f"❌ Feedback 엔드포인트 오류: {e}")

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("FastAPI 통합 테스트 시작")
    print("=" * 60)
    print(f"\n서버 URL: {BASE_URL}")
    print("서버가 실행 중인지 확인 중...")
    
    # 서버가 실행 중인지 확인
    if not wait_for_server(BASE_URL):
        print("\n⚠️  서버가 실행되지 않았습니다.")
        print("다음 명령어로 서버를 시작하세요:")
        print("  python api/main.py")
        print("\n또는 다음 명령어로 서버를 시작하고 테스트를 실행하세요:")
        print("  python api/test/test_integration.py --start-server")
        return 1
    
    # 테스트 실행
    results = []
    
    results.append(("Health Check", test_health_endpoint()))
    results.append(("OAuth2 Google", test_oauth2_google_endpoints()))
    results.append(("API 문서 비활성화", test_api_docs_disabled()))
    test_pydantic_validation()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name}: {status}")
    
    print(f"\n통과: {passed}/{total}")
    
    if passed == total:
        print("✅ 모든 테스트 통과!")
        return 0
    else:
        print("⚠️  일부 테스트 실패")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI 통합 테스트")
    parser.add_argument("--start-server", action="store_true", help="서버를 시작하고 테스트 실행")
    args = parser.parse_args()
    
    if args.start_server:
        print("서버를 시작합니다...")
        # 서버를 백그라운드로 시작
        server_process = subprocess.Popen(
            [sys.executable, "api/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # 서버 시작 대기
            if wait_for_server(BASE_URL):
                # 테스트 실행
                exit_code = main()
            else:
                print("❌ 서버 시작 실패")
                exit_code = 1
        finally:
            # 서버 종료
            server_process.terminate()
            server_process.wait()
            print("\n서버를 종료했습니다.")
        
        sys.exit(exit_code)
    else:
        sys.exit(main())

