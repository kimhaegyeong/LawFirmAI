#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""precedent_content 테이블용 임베딩 버전 1 생성 스크립트"""

import os
import sys
from pathlib import Path
from datetime import datetime

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
    """precedent_content 테이블용 버전 1 등록"""
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        print("   DATABASE_URL 환경변수 또는 POSTGRES_* 환경변수를 설정하세요.")
        return
    
    print(f"📌 데이터베이스 연결: {db_url.split('@')[1] if '@' in db_url else '***'}")
    
    mgr = PgEmbeddingVersionManager(db_url)
    
    # 기존 버전 확인
    existing_version = mgr.get_version_info(version=1, data_type='precedents')
    if existing_version:
        print(f"ℹ️  기존 버전 발견: ID={existing_version['id']}, version=1, data_type=precedents")
        if existing_version['is_active']:
            print("   이미 활성 버전으로 설정되어 있습니다.")
        else:
            print("   활성 버전으로 설정합니다...")
            mgr.set_active_version(version=1, data_type='precedents')
            print("   ✅ 활성 버전으로 설정 완료")
        return
    
    # precedent_content용 버전 1 생성
    # precedent_chunks 테이블에 있는 벡터들이 사용하는 모델 정보 확인 필요
    # 일반적으로 기존 벡터들이 사용하는 모델을 확인해야 함
    # 여기서는 기본값으로 설정 (실제 모델은 precedent_chunks 테이블의 벡터를 확인해야 함)
    version_id = mgr.get_or_create_version(
        version=1,
        model_name="woong0322/ko-legal-sbert-finetuned",  # 실제 모델명으로 변경 필요할 수 있음
        dim=768,
        data_type="precedents",
        chunking_strategy="512-token",
        description="precedent_content 테이블용 버전 1 - 판례 청크 임베딩",
        metadata={
            "model_name": "woong0322/ko-legal-sbert-finetuned",
            "dimension": 768,
            "chunking_strategy": "512-token",
            "data_type": "precedents",
            "table_name": "precedent_chunks",
            "created_at": datetime.now().isoformat()
        },
        set_active=True
    )
    
    print(f"✅ precedent_content 버전 1 생성 완료 (ID: {version_id})")
    print(f"   모델: woong0322/ko-legal-sbert-finetuned")
    print(f"   차원: 768")
    print(f"   데이터 타입: precedents")
    print(f"   청킹 전략: 512-token")
    print(f"   활성 버전으로 설정됨")

if __name__ == '__main__':
    main()

