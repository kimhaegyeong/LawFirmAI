#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL 확장 설치 상태 확인 스크립트
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import text
from utils.env_loader import ensure_env_loaded
from scripts.migrations.utils.database import build_database_url, get_database_connection

ensure_env_loaded(_PROJECT_ROOT)

def main():
    db_url = build_database_url()
    if not db_url:
        print("❌ DATABASE_URL을 구성할 수 없습니다.")
        return
    
    # PostgreSQL인지 확인
    if not db_url.startswith('postgresql'):
        print("⚠️  이 스크립트는 PostgreSQL에서만 작동합니다.")
        print(f"현재 데이터베이스: {db_url.split('://')[0]}")
        return
    
    engine = get_database_connection(database_url=db_url)
    
    try:
        with engine.connect() as conn:
            # pg_trgm 확장 확인
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm'"))
            row = result.fetchone()
            
            if row:
                print("✅ pg_trgm 확장이 설치되어 있습니다.")
            else:
                print("⚠️  pg_trgm 확장이 설치되어 있지 않습니다.")
                print("다음 명령으로 설치하세요:")
                print("  python scripts/migrations/scripts/init/init_open_law_schema.py")
            
            # 모든 확장 목록
            result = conn.execute(text("SELECT extname, extversion FROM pg_extension ORDER BY extname"))
            print("\n📋 설치된 확장 목록:")
            for row in result:
                print(f"  - {row[0]}: {row[1]}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    main()

