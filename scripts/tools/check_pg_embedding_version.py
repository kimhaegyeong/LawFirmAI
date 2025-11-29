#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostgreSQL 임베딩 버전 확인 스크립트"""

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
    scripts_env = _PROJECT_ROOT / "scripts" / ".env"
    if scripts_env.exists():
        load_dotenv(dotenv_path=str(scripts_env), override=True)
except ImportError:
    pass

from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
from scripts.ingest.open_law.utils import build_database_url

def main():
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return
    
    mgr = PgEmbeddingVersionManager(db_url)
    
    print("=" * 80)
    print("PostgreSQL 임베딩 버전 정보")
    print("=" * 80)
    
    # 법령 활성 버전
    print("\n📜 법령 (statutes) 활성 버전:")
    active_statutes = mgr.get_active_version('statutes')
    if active_statutes:
        print(f"  버전: {active_statutes['version']}")
        print(f"  모델: {active_statutes['model_name']}")
        print(f"  차원: {active_statutes['dim']}")
        print(f"  청킹 전략: {active_statutes['chunking_strategy']}")
        print(f"  설명: {active_statutes['description']}")
        print(f"  생성일: {active_statutes['created_at']}")
    else:
        print("  활성 버전 없음")
    
    # 판례 활성 버전
    print("\n⚖️  판례 (precedents) 활성 버전:")
    active_precedents = mgr.get_active_version('precedents')
    if active_precedents:
        print(f"  버전: {active_precedents['version']}")
        print(f"  모델: {active_precedents['model_name']}")
        print(f"  차원: {active_precedents['dim']}")
        print(f"  청킹 전략: {active_precedents['chunking_strategy']}")
        print(f"  설명: {active_precedents['description']}")
        print(f"  생성일: {active_precedents['created_at']}")
    else:
        print("  활성 버전 없음")
    
    # 전체 버전 목록
    print("\n📋 전체 버전 목록:")
    all_versions = mgr.list_versions()
    for v in all_versions:
        active_mark = "✅ 활성" if v['is_active'] else "  "
        print(f"  {active_mark} [{v['data_type']}] 버전 {v['version']}: {v['model_name']} "
              f"({v['chunking_strategy']}) - {v['created_at']}")

if __name__ == '__main__':
    main()

