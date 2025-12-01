#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
특정 세션 ID 확인 스크립트
"""

import os
import sys
from pathlib import Path

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

from api.database.connection import get_session
from api.database.models import Session as SessionModel

def main():
    session_id = "0Qsgy9pfW5RV9660OIkJ92R0bEAYe7e-l082tJnK17Y"
    
    print("=" * 80)
    print(f"세션 확인: {session_id}")
    print("=" * 80)
    
    db = get_session()
    try:
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if session:
            print(f"✅ 세션 발견!")
            print(f"   - session_id: {session.session_id}")
            print(f"   - title: {session.title}")
            print(f"   - user_id: {session.user_id}")
            print(f"   - created_at: {session.created_at}")
            print(f"   - updated_at: {session.updated_at}")
            print(f"   - message_count: {session.message_count}")
        else:
            print("❌ 세션을 찾을 수 없습니다")
            
            # 전체 세션 개수 및 최근 세션 확인
            total = db.query(SessionModel).count()
            print(f"\n   전체 세션 개수: {total}")
            
            if total > 0:
                recent = db.query(SessionModel).order_by(
                    SessionModel.updated_at.desc()
                ).limit(5).all()
                print(f"\n   최근 세션 5개:")
                for s in recent:
                    print(f"   - {s.session_id} ({s.title or 'N/A'})")
    finally:
        db.close()
    
    print("=" * 80)

if __name__ == '__main__':
    main()

