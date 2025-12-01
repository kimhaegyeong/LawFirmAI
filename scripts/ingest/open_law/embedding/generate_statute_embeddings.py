#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령 조문 벡터 임베딩 생성 스크립트
pgvector 및 FAISS 임베딩 생성
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError as e:
    logger.warning(f"환경 변수 로더를 불러올 수 없습니다: {e}")
except Exception as e:
    logger.warning(f"환경 변수 로드 중 오류: {e}")

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

def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}시간 {minutes}분"


def format_progress(current: int, total: int = None, prefix: str = "") -> str:
    """진행률을 포맷팅"""
    if total is None or total == 0:
        return f"{prefix}{current:,}개 처리됨"
    percentage = (current / total) * 100
    return f"{prefix}{current:,}/{total:,}개 ({percentage:.1f}%)"


def log_statistics(stats: dict, elapsed_time: float, method: str = ""):
    """통계 정보를 로깅"""
    method_prefix = f"[{method}] " if method else ""
    
    logger.info("=" * 80)
    logger.info(f"{method_prefix}임베딩 생성 통계")
    logger.info("-" * 80)
    logger.info(f"  총 처리: {stats.get('total_processed', 0):,}개")
    logger.info(f"  임베딩 생성: {stats.get('total_embedded', 0):,}개")
    if 'total_skipped' in stats:
        logger.info(f"  건너뜀: {stats.get('total_skipped', 0):,}개")
    logger.info(f"  실패: {stats.get('total_failed', 0):,}개")
    logger.info(f"  소요 시간: {format_duration(elapsed_time)}")
    
    if elapsed_time > 0 and stats.get('total_processed', 0) > 0:
        speed = stats.get('total_processed', 0) / elapsed_time
        logger.info(f"  처리 속도: {speed:.1f}개/초")
    
    if stats.get('errors'):
        error_count = len(stats['errors'])
        logger.warning(f"  오류 발생: {error_count}건")
        if error_count <= 5:
            for error in stats['errors']:
                logger.warning(f"    - {error}")
        else:
            logger.warning(f"    (처음 5개 오류만 표시)")
            for error in stats['errors'][:5]:
                logger.warning(f"    - {error}")
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='법령 조문 벡터 임베딩 생성')
    parser.add_argument(
        '--db',
        default=None,
        help='PostgreSQL 데이터베이스 URL (기본값: 환경변수에서 자동 로드)'
    )
    parser.add_argument(
        '--method',
        choices=['pgvector', 'faiss', 'both'],
        default='pgvector',
        help='임베딩 생성 방법 (기본값: pgvector)'
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
        '--domain',
        choices=['civil_law', 'criminal_law', 'administrative_law'],
        default=None,
        help='도메인 필터 (기본값: 전체)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/embeddings/open_law_postgresql'),
        help='FAISS 출력 디렉토리 (기본값: data/embeddings/open_law_postgresql)'
    )
    parser.add_argument(
        '--model',
        default='jhgan/ko-sroberta-multitask',
        help='임베딩 모델 이름 (기본값: jhgan/ko-sroberta-multitask)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='임베딩 버전 (기본값: 활성 버전 또는 자동 생성)'
    )
    parser.add_argument(
        '--chunking-strategy',
        default='article',
        help='청킹 전략 (기본값: article)'
    )
    
    args = parser.parse_args()
    
    # 데이터베이스 URL 확인 및 설정
    if not args.db:
        # 환경 변수에서 자동 로드 시도
        db_url = build_database_url() or os.getenv('DATABASE_URL')
        if db_url:
            args.db = db_url
            logger.debug("환경 변수에서 데이터베이스 URL을 자동으로 로드했습니다.")
        else:
            logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
            logger.info(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
            logger.info(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")
            logger.info(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
            logger.info(f"POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
            logger.info(f"POSTGRES_PASSWORD: {'설정됨' if os.getenv('POSTGRES_PASSWORD') else '설정 안 됨'}")
            logger.info(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
            logger.info(f"build_database_url() 결과: {build_database_url()}")
            return
    
    try:
        script_start_time = time.time()
        logger.info("=" * 80)
        logger.info("법령 조문 임베딩 생성 스크립트 시작")
        logger.info("=" * 80)
        logger.debug(f"설정:")
        logger.debug(f"  방법: {args.method}")
        logger.debug(f"  모델: {args.model}")
        logger.debug(f"  배치 크기: {args.batch_size}")
        logger.debug(f"  최대 처리 개수: {args.limit if args.limit else '전체'}")
        logger.debug(f"  도메인: {args.domain if args.domain else '전체'}")
        logger.debug(f"  청킹 전략: {args.chunking_strategy}")
        if args.version:
            logger.debug(f"  임베딩 버전: {args.version}")
        logger.debug("=" * 80)
        
        # pgvector 임베딩 생성
        if args.method in ['pgvector', 'both']:
            logger.info("▶ [pgvector] 법령 조문 임베딩 생성 시작")
            logger.info("-" * 80)
            pgvector_start_time = time.time()
            
            try:
                pgvector_embedder = PgVectorEmbedder(
                    args.db, 
                    model_name=args.model,
                    version=args.version,
                    chunking_strategy=args.chunking_strategy
                )
                logger.debug(f"PgVectorEmbedder 초기화 완료 (모델: {args.model})")
                
                results = pgvector_embedder.generate_statute_embeddings(
                    batch_size=args.batch_size,
                    limit=args.limit,
                    domain=args.domain,
                    version=args.version,
                    chunking_strategy=args.chunking_strategy
                )
                
                pgvector_elapsed = time.time() - pgvector_start_time
                log_statistics(results, pgvector_elapsed, "pgvector")
                
            except Exception as e:
                logger.error(f"[pgvector] 임베딩 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                if args.method == 'pgvector':  # pgvector만 실행하는 경우에만 종료
                    raise
        
        # FAISS 임베딩 생성
        if args.method in ['faiss', 'both']:
            logger.info("▶ [FAISS] 법령 조문 임베딩 생성 시작")
            logger.info("-" * 80)
            faiss_start_time = time.time()
            
            try:
                faiss_embedder = FaissEmbedder(
                    args.db,
                    args.output_dir,
                    model_name=args.model
                )
                logger.debug(f"FaissEmbedder 초기화 완료 (모델: {args.model}, 출력 디렉토리: {args.output_dir})")
                
                results = faiss_embedder.generate_embeddings(
                    data_type='statutes',
                    batch_size=args.batch_size,
                    limit=args.limit,
                    domain=args.domain
                )
                
                faiss_embed_elapsed = time.time() - faiss_start_time
                log_statistics(results, faiss_embed_elapsed, "FAISS")
                
                # 저장
                logger.info("▶ [FAISS] 임베딩 저장 시작")
                save_start_time = time.time()
                success = faiss_embedder.save_embeddings('statutes')
                save_elapsed = time.time() - save_start_time
                
                if success:
                    logger.info(f"✓ [FAISS] 임베딩 저장 완료 (소요 시간: {format_duration(save_elapsed)})")
                else:
                    logger.error("✗ [FAISS] 임베딩 저장 실패")
                    sys.exit(1)
                    
            except Exception as e:
                logger.error(f"[FAISS] 임베딩 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                if args.method == 'faiss':  # FAISS만 실행하는 경우에만 종료
                    raise
        
        script_elapsed = time.time() - script_start_time
        logger.info("=" * 80)
        logger.info("✓ 법령 조문 임베딩 생성 완료")
        logger.info(f"  총 소요 시간: {format_duration(script_elapsed)}")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

