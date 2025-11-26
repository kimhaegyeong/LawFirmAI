#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precedent_contents 테이블에 UNIQUE 인덱스 추가
"""

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.env_loader import ensure_env_loaded
ensure_env_loaded(_PROJECT_ROOT)

from scripts.migrations.utils.database import build_database_url, get_database_connection
from sqlalchemy import text

def main():
    db_url = build_database_url()
    if not db_url:
        print("데이터베이스 URL을 찾을 수 없습니다.")
        return
    
    engine = get_database_connection(database_url=db_url)
    
    with engine.connect() as conn:
        try:
            # 중복 데이터 확인
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT (precedent_id, section_type)) as unique_combos
                FROM precedent_contents
            """))
            row = result.fetchone()
            total = row[0]
            unique_combos = row[1]
            
            if total > unique_combos:
                print(f"경고: 중복 데이터가 있습니다. (전체: {total}, 고유: {unique_combos})")
                print("중복 데이터를 정리한 후 인덱스를 추가하세요.")
                return
            
            # UNIQUE 인덱스 추가
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_precedent_contents_precedent_section 
                ON precedent_contents(precedent_id, section_type)
            """))
            conn.commit()
            print("UNIQUE 인덱스 추가 완료")
            
        except Exception as e:
            print(f"인덱스 추가 실패: {e}")
            conn.rollback()

if __name__ == '__main__':
    main()

