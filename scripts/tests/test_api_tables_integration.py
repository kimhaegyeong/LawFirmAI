#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 서버용 테이블 통합 테스트
- users, sessions, messages 테이블 CRUD 테스트
- OAuth2 시나리오 시뮬레이션
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
        langgraph_env = _PROJECT_ROOT / "lawfirm_langgraph" / ".env"
        if langgraph_env.exists():
            load_dotenv(dotenv_path=str(langgraph_env), override=True)
    except ImportError:
        pass

from api.services.user_service import UserService
from api.services.session_service import SessionService
from api.database.models import Message
from api.database.connection import get_session

# 테스트 결과 추적
test_results = {
    "passed": [],
    "failed": [],
    "total": 0
}

def print_test(name: str):
    """테스트 시작 출력"""
    print(f"\n{'='*80}")
    print(f"🧪 테스트: {name}")
    print(f"{'='*80}")

def print_success(message: str):
    """성공 메시지 출력"""
    print(f"✅ {message}")
    test_results["passed"].append(message)
    test_results["total"] += 1

def print_error(message: str, error: Exception = None):
    """에러 메시지 출력"""
    print(f"❌ {message}")
    if error:
        print(f"   오류: {error}")
        import traceback
        traceback.print_exc()
    test_results["failed"].append(message)
    test_results["total"] += 1

def test_user_service():
    """UserService 테스트"""
    print_test("UserService - 사용자 생성 및 조회")
    
    user_service = UserService()
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    
    try:
        # 1. 사용자 생성
        success = user_service.create_or_update_user(
            user_id=test_user_id,
            email=test_email,
            name="테스트 사용자",
            picture="https://example.com/picture.jpg",
            provider="google",
            google_access_token="test_access_token_123",
            google_refresh_token="test_refresh_token_456"
        )
        
        if not success:
            print_error("사용자 생성 실패")
            return False
        
        print_success("사용자 생성 성공")
        
        # 2. 사용자 조회
        user = user_service.get_user(test_user_id)
        if not user:
            print_error("사용자 조회 실패 (None 반환)")
            return False
        
        if user["user_id"] != test_user_id:
            print_error(f"사용자 ID 불일치: {user['user_id']} != {test_user_id}")
            return False
        
        if user["email"] != test_email:
            print_error(f"이메일 불일치: {user['email']} != {test_email}")
            return False
        
        print_success(f"사용자 조회 성공: {user['user_id']}")
        
        # 3. 사용자 업데이트
        new_name = "업데이트된 사용자"
        success = user_service.create_or_update_user(
            user_id=test_user_id,
            name=new_name
        )
        
        if not success:
            print_error("사용자 업데이트 실패")
            return False
        
        updated_user = user_service.get_user(test_user_id)
        if updated_user["name"] != new_name:
            print_error(f"이름 업데이트 실패: {updated_user['name']} != {new_name}")
            return False
        
        print_success(f"사용자 업데이트 성공: {updated_user['name']}")
        
        # 4. Google 토큰 조회
        tokens = user_service.get_google_tokens(test_user_id)
        if not tokens:
            print_error("Google 토큰 조회 실패")
            return False
        
        if tokens["access_token"] != "test_access_token_123":
            print_error(f"Access Token 불일치: {tokens['access_token']}")
            return False
        
        print_success("Google 토큰 조회 성공")
        
        # 5. 정리: 사용자 삭제
        deleted = user_service.delete_user(test_user_id)
        if not deleted:
            print_error("사용자 삭제 실패")
            return False
        
        print_success("사용자 삭제 성공")
        
        return True
        
    except Exception as e:
        print_error("UserService 테스트 중 예외 발생", e)
        return False

def test_session_service():
    """SessionService 테스트"""
    print_test("SessionService - 세션 생성 및 조회")
    
    session_service = SessionService()
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    
    try:
        # 1. 세션 생성
        test_session_id = session_service.create_session(
            user_id=test_user_id,
            title="테스트 세션",
            ip_address="127.0.0.1"
        )
        
        if not test_session_id:
            print_error("세션 생성 실패")
            return False
        
        print_success(f"세션 생성 성공: {test_session_id}")
        
        # 2. 세션 조회
        retrieved_session = session_service.get_session(test_session_id)
        if not retrieved_session:
            print_error("세션 조회 실패 (None 반환)")
            return False
        
        if retrieved_session["user_id"] != test_user_id:
            print_error(f"사용자 ID 불일치: {retrieved_session['user_id']} != {test_user_id}")
            return False
        
        print_success("세션 조회 성공")
        
        # 3. 세션 목록 조회
        sessions = session_service.get_user_sessions(test_user_id)
        if not sessions:
            print_error("세션 목록 조회 실패 (빈 리스트)")
            return False
        
        found = any(s.get("session_id") == test_session_id for s in sessions)
        if not found:
            print_error("생성한 세션이 목록에 없음")
            return False
        
        print_success(f"세션 목록 조회 성공: {len(sessions)}개")
        
        # 4. 세션 삭제
        deleted = session_service.delete_session(test_session_id)
        if not deleted:
            print_error("세션 삭제 실패")
            return False
        
        print_success("세션 삭제 성공")
        
        return True
        
    except Exception as e:
        print_error("SessionService 테스트 중 예외 발생", e)
        return False

def test_message_storage():
    """Message 저장 및 조회 테스트"""
    print_test("Message - 메시지 저장 및 조회")
    
    session_service = SessionService()
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    test_session_id = str(uuid.uuid4())
    
    try:
        # 1. 세션 생성 (메시지 저장을 위해 필요)
        test_session_id = session_service.create_session(
            user_id=test_user_id,
            title="메시지 테스트 세션"
        )
        
        if not test_session_id:
            print_error("세션 생성 실패 (메시지 테스트 전제조건)")
            return False
        
        # 2. 메시지 저장
        db = get_session()
        try:
            test_message_id = str(uuid.uuid4())
            test_metadata = {
                "sources_by_type": {
                    "statutes_articles": [
                        {
                            "type": "statute_article",
                            "law_name": "민법",
                            "article_no": "750"
                        }
                    ],
                    "precedent_contents": [],
                    "precedent_chunks": []
                },
                "sources_detail": [
                    {
                        "type": "statute_article",
                        "law_name": "민법",
                        "article_no": "750"
                    }
                ]
            }
            
            message = Message(
                message_id=test_message_id,
                session_id=test_session_id,
                role="assistant",
                content="테스트 메시지 내용입니다.",
                timestamp=datetime.now(),
                message_metadata=test_metadata
            )
            
            db.add(message)
            db.commit()
            
            print_success(f"메시지 저장 성공: {test_message_id}")
            
            # 3. 메시지 조회
            retrieved_message = db.query(Message).filter(
                Message.message_id == test_message_id
            ).first()
            
            if not retrieved_message:
                print_error("메시지 조회 실패 (None 반환)")
                return False
            
            if retrieved_message.content != "테스트 메시지 내용입니다.":
                print_error(f"메시지 내용 불일치: {retrieved_message.content}")
                return False
            
            if not retrieved_message.message_metadata:
                print_error("메시지 메타데이터 없음")
                return False
            
            metadata = retrieved_message.message_metadata
            if "sources_by_type" not in metadata:
                print_error("sources_by_type이 메타데이터에 없음")
                return False
            
            if "statutes_articles" not in metadata["sources_by_type"]:
                print_error("statutes_articles가 sources_by_type에 없음")
                return False
            
            print_success("메시지 조회 성공")
            print_success(f"메타데이터 검증 성공: {len(metadata['sources_by_type']['statutes_articles'])}개 출처")
            
            # 4. 세션별 메시지 조회
            session_messages = db.query(Message).filter(
                Message.session_id == test_session_id
            ).all()
            
            if len(session_messages) == 0:
                print_error("세션별 메시지 조회 실패 (빈 리스트)")
                return False
            
            print_success(f"세션별 메시지 조회 성공: {len(session_messages)}개")
            
            # 5. 정리
            db.delete(retrieved_message)
            db.commit()
            
            print_success("메시지 삭제 성공")
            
        finally:
            db.close()
        
        # 세션도 삭제
        session_service.delete_session(test_session_id)
        
        return True
        
    except Exception as e:
        print_error("Message 테스트 중 예외 발생", e)
        return False

def test_oauth2_scenario():
    """OAuth2 시나리오 시뮬레이션"""
    print_test("OAuth2 시나리오 - 전체 플로우 시뮬레이션")
    
    user_service = UserService()
    session_service = SessionService()
    
    # OAuth2에서 받은 정보 시뮬레이션
    google_user_id = f"google_{uuid.uuid4().hex[:8]}"
    google_email = f"user_{uuid.uuid4().hex[:8]}@gmail.com"
    google_name = "Google 사용자"
    google_picture = "https://lh3.googleusercontent.com/..."
    access_token = f"ya29.{uuid.uuid4().hex}"
    refresh_token = f"1//{uuid.uuid4().hex}"
    
    try:
        # 1. OAuth2 콜백: 사용자 생성/업데이트
        print("\n📝 Step 1: OAuth2 콜백 - 사용자 생성/업데이트")
        success = user_service.create_or_update_user(
            user_id=google_user_id,
            email=google_email,
            name=google_name,
            picture=google_picture,
            provider="google",
            google_access_token=access_token,
            google_refresh_token=refresh_token
        )
        
        if not success:
            print_error("OAuth2 사용자 생성 실패")
            return False
        
        print_success(f"OAuth2 사용자 생성 성공: {google_email}")
        
        # 2. 세션 생성 (로그인 후 첫 세션)
        print("\n📝 Step 2: 로그인 후 세션 생성")
        session_id = session_service.create_session(
            user_id=google_user_id,
            title="새로운 대화",
            ip_address="192.168.1.100"
        )
        
        if not session_id:
            print_error("세션 생성 실패")
            return False
        
        print_success(f"세션 생성 성공: {session_id}")
        
        # 3. 메시지 저장 (사용자 질문)
        print("\n📝 Step 3: 사용자 메시지 저장")
        db = get_session()
        try:
            user_message = Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content="계약 해지 사유에 대해 알려주세요",
                timestamp=datetime.now(),
                message_metadata=None
            )
            db.add(user_message)
            db.commit()
            print_success("사용자 메시지 저장 성공")
        finally:
            db.close()
        
        # 4. 메시지 저장 (AI 응답)
        print("\n📝 Step 4: AI 응답 메시지 저장")
        db = get_session()
        try:
            ai_message_id = str(uuid.uuid4())
            ai_metadata = {
                "sources_by_type": {
                    "statutes_articles": [
                        {
                            "type": "statute_article",
                            "law_name": "민법",
                            "article_no": "750",
                            "content": "불법행위로 인한 손해배상..."
                        }
                    ],
                    "precedent_contents": [
                        {
                            "type": "case_paragraph",
                            "case_name": "대법원 2023다12345",
                            "content": "계약 해지와 관련하여..."
                        }
                    ],
                    "precedent_chunks": []
                }
            }
            
            ai_message = Message(
                message_id=ai_message_id,
                session_id=session_id,
                role="assistant",
                content="계약 해지는 다음과 같은 사유로 가능합니다...",
                timestamp=datetime.now(),
                message_metadata=ai_metadata
            )
            db.add(ai_message)
            db.commit()
            print_success("AI 응답 메시지 저장 성공")
            print_success(f"메타데이터 저장 성공: {len(ai_metadata['sources_by_type']['statutes_articles'])}개 법령, {len(ai_metadata['sources_by_type']['precedent_contents'])}개 판례")
        finally:
            db.close()
        
        # 5. 세션 조회 및 검증
        print("\n📝 Step 5: 세션 및 메시지 조회 검증")
        retrieved_session = session_service.get_session(session_id)
        if not retrieved_session:
            print_error("세션 조회 실패")
            return False
        
        db = get_session()
        try:
            messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.timestamp).all()
            
            if len(messages) != 2:
                print_error(f"메시지 개수 불일치: {len(messages)} != 2")
                return False
            
            print_success(f"세션 메시지 조회 성공: {len(messages)}개")
            
            # 메타데이터 검증
            ai_msg = next((m for m in messages if m.role == "assistant"), None)
            if not ai_msg or not ai_msg.message_metadata:
                print_error("AI 메시지 또는 메타데이터 없음")
                return False
            
            metadata = ai_msg.message_metadata
            if "sources_by_type" not in metadata:
                print_error("sources_by_type 없음")
                return False
            
            print_success("OAuth2 시나리오 전체 플로우 검증 성공")
            
        finally:
            db.close()
        
        # 6. 정리
        print("\n📝 Step 6: 테스트 데이터 정리")
        session_service.delete_session(session_id)
        user_service.delete_user(google_user_id)
        print_success("테스트 데이터 정리 완료")
        
        return True
        
    except Exception as e:
        print_error("OAuth2 시나리오 테스트 중 예외 발생", e)
        return False

def main():
    """메인 함수"""
    print("=" * 80)
    print("API 서버용 테이블 통합 테스트 시작")
    print("=" * 80)
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 테스트 실행
    tests = [
        ("UserService", test_user_service),
        ("SessionService", test_session_service),
        ("Message Storage", test_message_storage),
        ("OAuth2 Scenario", test_oauth2_scenario),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} 테스트 실행 중 예외 발생", e)
            results[test_name] = False
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{status} {test_name}")
    
    print()
    print(f"총 테스트: {test_results['total']}개")
    print(f"성공: {len(test_results['passed'])}개")
    print(f"실패: {len(test_results['failed'])}개")
    
    if test_results['failed']:
        print("\n실패한 테스트:")
        for failed in test_results['failed']:
            print(f"  - {failed}")
    
    print("=" * 80)
    
    # 모든 테스트 통과 여부
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

