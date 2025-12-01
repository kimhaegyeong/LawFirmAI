#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""임베딩 버전 생성 스크립트"""

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
    
    # 버전 2 생성 (woong0322/ko-legal-sbert-finetuned)
    version_id = mgr.get_or_create_version(
        version=2,
        model_name="woong0322/ko-legal-sbert-finetuned",
        dim=768,
        data_type="statutes",
        chunking_strategy="article",
        description="법률 특화 모델로 업그레이드된 법령 조문 임베딩",
        metadata={
            "model_name": "woong0322/ko-legal-sbert-finetuned",
            "dimension": 768,
            "chunking_strategy": "article",
            "created_at": "2025-11-25T09:30:00"
        },
        set_active=True
    )
    
    print(f"✅ 버전 2 생성 완료 (ID: {version_id})")
    print(f"   모델: woong0322/ko-legal-sbert-finetuned")
    print(f"   활성 버전으로 설정됨")

if __name__ == '__main__':
    main()

