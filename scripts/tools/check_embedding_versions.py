#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""임베딩 버전 상태 확인 스크립트"""

import os
import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=True)
except ImportError:
    pass

from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
from scripts.ingest.open_law.utils import build_database_url

def main():
    """임베딩 버전 상태 확인"""
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return
    
    mgr = PgEmbeddingVersionManager(db_url)
    
    print("=" * 80)
    print("임베딩 버전 상태 확인")
    print("=" * 80)
    
    # 모든 버전 목록
    print("\n📋 모든 임베딩 버전:")
    all_versions = mgr.list_versions()
    for v in all_versions:
        status = "✅ 활성" if v['is_active'] else "❌ 비활성"
        print(f"  {status} | ID={v['id']:2d} | version={v['version']:2d} | data_type={v['data_type']:12s} | model={v['model_name']}")
    
    # 활성 버전별 확인
    print("\n✅ 활성 버전:")
    for data_type in ['statutes', 'precedents']:
        active = mgr.get_active_version(data_type)
        if active:
            print(f"  {data_type:12s}: version={active['version']}, ID={active['id']}, model={active['model_name']}")
        else:
            print(f"  {data_type:12s}: ❌ 활성 버전 없음")
    
    # precedent_chunks 테이블의 버전 분포 확인
    print("\n📊 precedent_chunks 테이블 버전 분포:")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT embedding_version, COUNT(*) as count
                FROM precedent_chunks
                WHERE embedding_version IS NOT NULL
                GROUP BY embedding_version
                ORDER BY embedding_version
            """))
            rows = result.fetchall()
            if rows:
                for row in rows:
                    print(f"  version {row[0]}: {row[1]:,}개 청크")
            else:
                print("  ⚠️  버전 정보가 없는 청크가 있습니다.")
    except Exception as e:
        print(f"  ❌ 확인 실패: {e}")

if __name__ == '__main__':
    main()

