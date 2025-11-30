"""
세션 생성 및 조회 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    print("⚠️  utils.env_loader를 찾을 수 없습니다. 환경 변수 로드를 건너뜁니다.")

import logging
from datetime import datetime
from api.services.session_service import session_service
from api.database.connection import get_session
from api.database.models import Session as SessionModel

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_session_creation():
    """세션 생성 테스트"""
    print("\n" + "="*60)
    print("세션 생성 테스트 시작")
    print("="*60)
    
    # 테스트 세션 생성
    test_title = f"테스트 세션 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    test_user_id = "test_user_123"
    test_ip = "127.0.0.1"
    
    print(f"\n1. 세션 생성 시도:")
    print(f"   - title: {test_title}")
    print(f"   - user_id: {test_user_id}")
    print(f"   - ip_address: {test_ip}")
    
    try:
        session_id = session_service.create_session(
            title=test_title,
            user_id=test_user_id,
            ip_address=test_ip
        )
        print(f"\n✅ 세션 생성 성공!")
        print(f"   - session_id: {session_id}")
        
        # 즉시 조회 테스트
        print(f"\n2. 생성된 세션 즉시 조회:")
        session = session_service.get_session(session_id, check_expiry=False)
        if session:
            print(f"   ✅ 세션 조회 성공!")
            print(f"   - session_id: {session.get('session_id')}")
            print(f"   - title: {session.get('title')}")
            print(f"   - user_id: {session.get('user_id')}")
            print(f"   - ip_address: {session.get('ip_address')}")
            print(f"   - created_at: {session.get('created_at')}")
            print(f"   - updated_at: {session.get('updated_at')}")
        else:
            print(f"   ❌ 세션 조회 실패! (None 반환)")
        
        # 데이터베이스 직접 조회
        print(f"\n3. 데이터베이스 직접 조회:")
        db = get_session()
        try:
            db_session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            
            if db_session:
                print(f"   ✅ 데이터베이스에서 세션 발견!")
                print(f"   - session_id: {db_session.session_id}")
                print(f"   - title: {db_session.title}")
                print(f"   - user_id: {db_session.user_id}")
                print(f"   - ip_address: {db_session.ip_address}")
                print(f"   - created_at: {db_session.created_at}")
                print(f"   - updated_at: {db_session.updated_at}")
                print(f"   - message_count: {db_session.message_count}")
            else:
                print(f"   ❌ 데이터베이스에서 세션을 찾을 수 없습니다!")
        finally:
            db.close()
        
        # 전체 세션 목록 확인
        print(f"\n4. 전체 세션 목록 조회:")
        sessions, total = session_service.list_sessions(page=1, page_size=5)
        print(f"   - 총 세션 수: {total}")
        print(f"   - 조회된 세션 수: {len(sessions)}")
        if sessions:
            print(f"   - 최근 세션:")
            for s in sessions[:3]:
                print(f"     * {s.get('session_id')[:20]}... - {s.get('title')}")
        
        return session_id
        
    except Exception as e:
        print(f"\n❌ 세션 생성 실패!")
        print(f"   오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_specific_session(session_id: str):
    """특정 세션 조회 테스트"""
    print("\n" + "="*60)
    print(f"특정 세션 조회 테스트: {session_id}")
    print("="*60)
    
    # session_service를 통한 조회
    print(f"\n1. session_service.get_session() 사용:")
    session = session_service.get_session(session_id, check_expiry=False)
    if session:
        print(f"   ✅ 세션 조회 성공!")
        print(f"   - title: {session.get('title')}")
        print(f"   - user_id: {session.get('user_id')}")
    else:
        print(f"   ❌ 세션 조회 실패!")
    
    # 데이터베이스 직접 조회
    print(f"\n2. 데이터베이스 직접 조회:")
    db = get_session()
    try:
        db_session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if db_session:
            print(f"   ✅ 데이터베이스에서 세션 발견!")
            print(f"   - session_id: {db_session.session_id}")
            print(f"   - title: {db_session.title}")
            print(f"   - user_id: {db_session.user_id}")
        else:
            print(f"   ❌ 데이터베이스에서 세션을 찾을 수 없습니다!")
    finally:
        db.close()


if __name__ == "__main__":
    # 테스트 실행
    created_session_id = test_session_creation()
    
    # 특정 세션 ID로 테스트 (명령줄 인자로 제공된 경우)
    if len(sys.argv) > 1:
        test_session_id = sys.argv[1]
        test_specific_session(test_session_id)
    elif created_session_id:
        # 생성된 세션으로 테스트
        test_specific_session(created_session_id)
    
    print("\n" + "="*60)
    print("테스트 완료")
    print("="*60)

