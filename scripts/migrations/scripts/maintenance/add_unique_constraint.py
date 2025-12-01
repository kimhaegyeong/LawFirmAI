#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UNIQUE 제약조건 추가 스크립트"""

import os
import sys
from pathlib import Path
from sqlalchemy import text

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=True)
    scripts_env = _PROJECT_ROOT / "scripts" / ".env"
    if scripts_env.exists():
        load_dotenv(dotenv_path=str(scripts_env), override=True)
except ImportError:
    pass

from scripts.migrations.utils.database import build_database_url, get_database_connection

def main():
    db_url = build_database_url()
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        sys.exit(1)
    
    engine = get_database_connection(database_url=db_url)
    
    with engine.connect() as conn:
        # 기존 제약조건 확인 및 추가
        check_sql = """
            SELECT 1 FROM pg_constraint 
            WHERE conname = 'statute_embeddings_article_id_version_key'
        """
        result = conn.execute(text(check_sql))
        if result.fetchone():
            print("✅ UNIQUE 제약조건이 이미 존재합니다.")
        else:
            # 기존 (article_id) 제약조건 삭제 시도
            try:
                drop_sql = "ALTER TABLE statute_embeddings DROP CONSTRAINT IF EXISTS statute_embeddings_article_id_key"
                conn.execute(text(drop_sql))
                conn.commit()
                print("✅ 기존 제약조건 삭제 완료")
            except:
                pass
            
            # 새 제약조건 추가
            trans = conn.begin()
            try:
                add_sql = """
                    ALTER TABLE statute_embeddings 
                    ADD CONSTRAINT statute_embeddings_article_id_version_key 
                    UNIQUE(article_id, embedding_version)
                """
                conn.execute(text(add_sql))
                trans.commit()
                print("✅ UNIQUE 제약조건 추가 완료: (article_id, embedding_version)")
            except Exception as e:
                trans.rollback()
                print(f"❌ 제약조건 추가 실패: {e}")

if __name__ == '__main__':
    main()

