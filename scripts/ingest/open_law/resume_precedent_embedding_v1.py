#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""민사법 판례 임베딩 재시작 스크립트 (버전 1)"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError as e:
    print(f"환경 변수 로더를 불러올 수 없습니다: {e}")

# 모듈 임포트
try:
    from scripts.ingest.open_law.utils import build_database_url
    from scripts.ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.utils import build_database_url
    from ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    # 데이터베이스 URL 확인
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        logger.info(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
        logger.info(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
        return 1
    
    logger.info("=" * 80)
    logger.info("민사법 판례 임베딩 재시작 (버전 1)")
    logger.info("=" * 80)
    
    # 버전 1로 임베딩 생성기 초기화
    embedder = PgVectorEmbedder(
        db_url,
        model_name='jhgan/ko-sroberta-multitask',
        version=1,
        chunking_strategy='512-token'
    )
    
    logger.info("=" * 80)
    logger.info("임베딩 생성 시작...")
    logger.info("=" * 80)
    
    # 민사법 판례 임베딩 생성 (이미 임베딩된 항목은 자동으로 건너뜀)
    results = embedder.generate_precedent_embeddings(
        batch_size=100,
        limit=None,
        domain='civil_law',
        version=1,
        chunking_strategy='512-token'
    )
    
    logger.info("=" * 80)
    logger.info("민사법 판례 임베딩 완료")
    logger.info("=" * 80)
    logger.info(f"처리 결과:")
    logger.info(f"  총 처리: {results.get('total_processed', 0):,}개")
    logger.info(f"  임베딩 생성: {results.get('total_embedded', 0):,}개")
    logger.info(f"  건너뜀: {results.get('total_skipped', 0):,}개")
    logger.info(f"  실패: {results.get('total_failed', 0):,}개")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

