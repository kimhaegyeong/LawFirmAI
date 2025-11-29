#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
세션 404 오류 디버깅 스크립트
"""

import os
import sys
from pathlib import Path
import logging

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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

from api.services.session_service import SessionService
from api.database.connection import get_session
from api.database.models import Session as SessionModel

def main():
    session_id = "mxn2DHgkfWgtB5owRfYqbd21JjnUfvW8JGASqJPgjD4"
    
    print("=" * 80)
    print(f"세션 404 오류 디버깅: {session_id}")
    print("=" * 80)
    
    # 1. 직접 데이터베이스 조회
    print("\n1. 직접 데이터베이스 조회:")
    db = get_session()
    try:
        session_model = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if session_model:
            print(f"   ✅ 세션 발견!")
            print(f"   - session_id: {session_model.session_id}")
            print(f"   - title: {session_model.title}")
            print(f"   - user_id: {session_model.user_id}")
            print(f"   - created_at: {session_model.created_at} (tzinfo: {session_model.created_at.tzinfo if session_model.created_at else None})")
            print(f"   - updated_at: {session_model.updated_at} (tzinfo: {session_model.updated_at.tzinfo if session_model.updated_at else None})")
            print(f"   - message_count: {session_model.message_count}")
            print(f"   - ip_address: {session_model.ip_address}")
            
            # to_dict() 결과 확인
            session_dict = session_model.to_dict()
            print(f"\n   to_dict() 결과:")
            for key, value in session_dict.items():
                print(f"   - {key}: {value} (type: {type(value).__name__})")
        else:
            print("   ❌ 세션을 찾을 수 없습니다")
            
            # 전체 세션 개수 및 최근 세션 확인
            total_sessions = db.query(SessionModel).count()
            print(f"\n   전체 세션 개수: {total_sessions}")
            
            if total_sessions > 0:
                recent_sessions = db.query(SessionModel).order_by(
                    SessionModel.updated_at.desc()
                ).limit(10).all()
                print(f"\n   최근 세션 10개:")
                for s in recent_sessions:
                    print(f"   - {s.session_id} ({s.title or 'N/A'})")
    finally:
        db.close()
    
    # 2. SessionService.get_session() 테스트 (check_expiry=False)
    print("\n2. SessionService.get_session() 테스트 (check_expiry=False):")
    session_service = SessionService()
    try:
        session = session_service.get_session(session_id, check_expiry=False)
        if session:
            print(f"   ✅ 세션 발견 (check_expiry=False)")
            print(f"   - title: {session.get('title')}")
            print(f"   - user_id: {session.get('user_id')}")
        else:
            print("   ❌ 세션을 찾을 수 없습니다 (check_expiry=False)")
    except Exception as e:
        print(f"   ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. SessionService.get_session() 테스트 (check_expiry=True, 기본값)
    print("\n3. SessionService.get_session() 테스트 (check_expiry=True):")
    try:
        session = session_service.get_session(session_id, check_expiry=True)
        if session:
            print(f"   ✅ 세션 발견 (check_expiry=True)")
            print(f"   - title: {session.get('title')}")
            print(f"   - user_id: {session.get('user_id')}")
        else:
            print("   ❌ 세션을 찾을 수 없습니다 (check_expiry=True)")
            print("   ⚠️  만료 확인 로직에서 None이 반환되었을 수 있습니다")
    except Exception as e:
        print(f"   ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 만료 시간 계산 테스트
    if session_model:
        print("\n4. 만료 시간 계산 테스트:")
        from datetime import datetime, timezone, timedelta
        from api.config import api_config
        
        KST = timezone(timedelta(hours=9))
        
        updated_at = session_model.updated_at
        print(f"   - updated_at: {updated_at} (tzinfo: {updated_at.tzinfo if updated_at else None})")
        
        # timezone 변환 테스트
        if updated_at and updated_at.tzinfo is None:
            updated_at_aware = updated_at.replace(tzinfo=timezone.utc).astimezone(KST)
            print(f"   - updated_at (KST 변환 후): {updated_at_aware}")
        elif updated_at:
            updated_at_aware = updated_at.astimezone(KST)
            print(f"   - updated_at (KST 변환 후): {updated_at_aware}")
        else:
            updated_at_aware = None
        
        if updated_at_aware:
            expiry_hours = api_config.session_ttl_hours
            expiry_time = updated_at_aware + timedelta(hours=expiry_hours)
            now_kst = datetime.now(KST)
            
            print(f"   - session_ttl_hours: {expiry_hours}")
            print(f"   - expiry_time: {expiry_time}")
            print(f"   - now_kst: {now_kst}")
            print(f"   - 만료 여부: {now_kst > expiry_time}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

