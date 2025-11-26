#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 임베딩 생성 스크립트
pgvector 및 FAISS 임베딩 생성
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 모듈 임포트
try:
    from scripts.ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder
    from scripts.ingest.open_law.embedding.faiss.faiss_embedder import FaissEmbedder
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder
    from ingest.open_law.embedding.faiss.faiss_embedder import FaissEmbedder

# 데이터베이스 URL 빌드
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    from urllib.parse import quote_plus
    def build_database_url():
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        if db and user and password:
            encoded_password = quote_plus(password)
            return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        return None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='통합 임베딩 생성')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--method',
        choices=['pgvector', 'faiss', 'both'],
        default='both',
        help='임베딩 생성 방법'
    )
    parser.add_argument(
        '--data-type',
        choices=['precedents', 'statutes', 'both'],
        default='precedents',
        help='임베딩 생성할 데이터 타입'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='최대 처리 개수'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law'],
        default=None,
        help='도메인 필터'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/embeddings/open_law_postgresql'),
        help='FAISS 출력 디렉토리'
    )
    parser.add_argument(
        '--model',
        default='jhgan/ko-sroberta-multitask',
        help='임베딩 모델 이름'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    data_types = []
    if args.data_type == 'both':
        data_types = ['precedents', 'statutes']
    else:
        data_types = [args.data_type]
    
    try:
        # pgvector 임베딩 생성
        if args.method in ['pgvector', 'both']:
            logger.info("pgvector 임베딩 생성 시작")
            pgvector_embedder = PgVectorEmbedder(args.db, model_name=args.model)
            
            for data_type in data_types:
                if data_type == 'precedents':
                    results = pgvector_embedder.generate_precedent_embeddings(
                        batch_size=args.batch_size,
                        limit=args.limit,
                        domain=args.domain
                    )
                    logger.info(f"판례 청크 pgvector 임베딩 생성 완료: {results}")
                elif data_type == 'statutes':
                    results = pgvector_embedder.generate_statute_embeddings(
                        batch_size=args.batch_size,
                        limit=args.limit,
                        domain=args.domain
                    )
                    logger.info(f"법령 조문 pgvector 임베딩 생성 완료: {results}")
        
        # FAISS 임베딩 생성
        if args.method in ['faiss', 'both']:
            logger.info("FAISS 임베딩 생성 시작")
            
            for data_type in data_types:
                faiss_embedder = FaissEmbedder(
                    args.db,
                    args.output_dir,
                    model_name=args.model
                )
                
                results = faiss_embedder.generate_embeddings(
                    data_type=data_type,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    domain=args.domain
                )
                logger.info(f"{data_type} FAISS 임베딩 생성 완료: {results}")
                
                # 저장
                success = faiss_embedder.save_embeddings(data_type)
                if success:
                    logger.info(f"{data_type} FAISS 임베딩 저장 완료")
                else:
                    logger.error(f"{data_type} FAISS 임베딩 저장 실패")
        
        logger.info("모든 임베딩 생성 완료")
    
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

