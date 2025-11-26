#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
증분 데이터 파이프라인 실행 스크립트
수집 -> 청킹 -> 임베딩을 순차적으로 실행 (증분 처리 지원)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
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
    from scripts.ingest.open_law.utils import build_database_url
    from scripts.ingest.open_law.chunk_precedents import chunk_precedents
    from scripts.ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder
    from scripts.ingest.open_law.embedding.faiss.faiss_embedder import FaissEmbedder
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.utils import build_database_url
    from ingest.open_law.chunk_precedents import chunk_precedents
    from ingest.open_law.embedding.pgvector.pgvector_embedder import PgVectorEmbedder
    from ingest.open_law.embedding.faiss.faiss_embedder import FaissEmbedder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_collection(domain: str, db_url: str):
    """판례 데이터 수집"""
    logger.info(f"{domain} 판례 데이터 수집 시작")
    
    if domain == 'civil_law':
        from scripts.ingest.open_law.scripts.collect_civil_precedents import main as collect_main
        # collect_main은 argparse를 사용하므로 직접 호출하기 어려움
        # 대신 subprocess로 실행
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'scripts.ingest.open_law.scripts.collect_civil_precedents',
            '--phase', 'content',
            '--input', 'data/raw/open_law/civil_precedents_list.json'
        ], cwd=str(_PROJECT_ROOT))
        return result.returncode == 0
    elif domain == 'criminal_law':
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'scripts.ingest.open_law.scripts.collect_criminal_precedents',
            '--phase', 'content',
            '--input', 'data/raw/open_law/criminal_precedents_list.json'
        ], cwd=str(_PROJECT_ROOT))
        return result.returncode == 0
    else:
        logger.error(f"알 수 없는 도메인: {domain}")
        return False


def run_chunking(db_url: str, domain: str = None, batch_size: int = 100, limit: int = None):
    """판례 데이터 청킹 (증분 처리)"""
    logger.info(f"판례 데이터 청킹 시작 (도메인: {domain or '전체'})")
    start_time = time.time()
    
    try:
        chunk_precedents(
            db_url=db_url,
            batch_size=batch_size,
            limit=limit,
            domain=domain
        )
        elapsed = time.time() - start_time
        logger.info(f"청킹 완료 (소요 시간: {elapsed:.1f}초)")
        return True
    except Exception as e:
        logger.error(f"청킹 실패: {e}")
        return False


def run_embedding(
    db_url: str,
    method: str = 'both',
    domain: str = None,
    batch_size: int = 100,
    limit: int = None,
    output_dir: Path = None,
    model: str = 'jhgan/ko-sroberta-multitask'
):
    """임베딩 생성 (증분 처리: 이미 임베딩된 데이터 제외)"""
    logger.info(f"임베딩 생성 시작 (방법: {method}, 도메인: {domain or '전체'})")
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path('data/embeddings/open_law_postgresql')
    
    try:
        # pgvector 임베딩 생성
        if method in ['pgvector', 'both']:
            logger.info("pgvector 임베딩 생성 시작 (증분 처리: 이미 임베딩된 데이터 제외)")
            pgvector_embedder = PgVectorEmbedder(db_url, model_name=model)
            
            results = pgvector_embedder.generate_precedent_embeddings(
                batch_size=batch_size,
                limit=limit,
                domain=domain
            )
            logger.info(f"판례 청크 pgvector 임베딩 생성 완료: {results}")
        
        # FAISS 임베딩 생성
        if method in ['faiss', 'both']:
            logger.info("FAISS 임베딩 생성 시작 (증분 처리: 이미 임베딩된 데이터 제외)")
            faiss_embedder = FaissEmbedder(
                db_url,
                output_dir,
                model_name=model
            )
            
            results = faiss_embedder.generate_embeddings(
                data_type='precedents',
                batch_size=batch_size,
                limit=limit,
                domain=domain
            )
            logger.info(f"판례 청크 FAISS 임베딩 생성 완료: {results}")
            
            # 저장
            success = faiss_embedder.save_embeddings('precedents')
            if success:
                logger.info("FAISS 임베딩 저장 완료")
            else:
                logger.error("FAISS 임베딩 저장 실패")
        
        elapsed = time.time() - start_time
        logger.info(f"임베딩 완료 (소요 시간: {elapsed:.1f}초)")
        return True
    
    except Exception as e:
        logger.error(f"임베딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='증분 데이터 파이프라인 실행')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['collect', 'chunk', 'embed'],
        default=['chunk', 'embed'],
        help='실행할 단계 (기본값: chunk, embed)'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law'],
        default=None,
        help='도메인 필터 (기본값: 전체)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기 (기본값: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='최대 처리 개수 (기본값: 전체)'
    )
    parser.add_argument(
        '--embedding-method',
        choices=['pgvector', 'faiss', 'both'],
        default='both',
        help='임베딩 생성 방법 (기본값: both)'
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
    
    logger.info("=" * 80)
    logger.info("증분 데이터 파이프라인 시작")
    logger.info("=" * 80)
    logger.info(f"실행 단계: {', '.join(args.steps)}")
    if args.domain:
        logger.info(f"도메인 필터: {args.domain}")
    logger.info("=" * 80)
    
    pipeline_start_time = time.time()
    
    # 1. 수집 단계
    if 'collect' in args.steps:
        if not args.domain:
            logger.error("수집 단계는 --domain 인자가 필요합니다.")
            return
        
        success = run_collection(args.domain, args.db)
        if not success:
            logger.error("수집 단계 실패")
            return
    
    # 2. 청킹 단계 (증분 처리)
    if 'chunk' in args.steps:
        success = run_chunking(
            db_url=args.db,
            domain=args.domain,
            batch_size=args.batch_size,
            limit=args.limit
        )
        if not success:
            logger.error("청킹 단계 실패")
            return
    
    # 3. 임베딩 단계 (증분 처리)
    if 'embed' in args.steps:
        success = run_embedding(
            db_url=args.db,
            method=args.embedding_method,
            domain=args.domain,
            batch_size=args.batch_size,
            limit=args.limit,
            output_dir=args.output_dir,
            model=args.model
        )
        if not success:
            logger.error("임베딩 단계 실패")
            return
    
    pipeline_elapsed = time.time() - pipeline_start_time
    logger.info("=" * 80)
    logger.info(f"증분 데이터 파이프라인 완료 (총 소요 시간: {pipeline_elapsed:.1f}초)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

