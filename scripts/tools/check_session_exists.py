#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
특정 세션 ID가 데이터베이스에 존재하는지 확인
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

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

from api.services.session_service import SessionService
from api.database.connection import get_session
from api.database.models import Session as SessionModel

def main():
    session_id = "oVojD8VeydkbE1ZvFLncSOnlCVmtWjBstSSsFtfatO0"
    
    print("=" * 80)
    print(f"세션 존재 여부 확인: {session_id}")
    print("=" * 80)
    
    # 방법 1: SessionService 사용
    print("\n1. SessionService.get_session() 사용:")
    session_service = SessionService()
    session = session_service.get_session(session_id, check_expiry=False)
    if session:
        print(f"   ✅ 세션 발견: {session.get('title', 'N/A')}")
        print(f"   - user_id: {session.get('user_id')}")
        print(f"   - created_at: {session.get('created_at')}")
        print(f"   - updated_at: {session.get('updated_at')}")
    else:
        print("   ❌ 세션을 찾을 수 없습니다")
    
    # 방법 2: 직접 데이터베이스 조회
    print("\n2. 직접 데이터베이스 조회:")
    db = get_session()
    try:
        session_model = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if session_model:
            print(f"   ✅ 세션 발견 (직접 조회): {session_model.title}")
            print(f"   - user_id: {session_model.user_id}")
            print(f"   - created_at: {session_model.created_at}")
            print(f"   - updated_at: {session_model.updated_at}")
            print(f"   - updated_at.tzinfo: {session_model.updated_at.tzinfo if session_model.updated_at else None}")
        else:
            print("   ❌ 세션을 찾을 수 없습니다 (직접 조회)")
            
            # 전체 세션 개수 확인
            total_sessions = db.query(SessionModel).count()
            print(f"\n   전체 세션 개수: {total_sessions}")
            
            if total_sessions > 0:
                # 최근 세션 5개 출력
                recent_sessions = db.query(SessionModel).order_by(
                    SessionModel.updated_at.desc()
                ).limit(5).all()
                print(f"\n   최근 세션 5개:")
                for s in recent_sessions:
                    print(f"   - {s.session_id[:20]}... ({s.title or 'N/A'})")
    finally:
        db.close()
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

