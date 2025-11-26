#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비교 테스트 실행 스크립트
FAISS vs pgvector 성능 및 정확도 비교
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
    from scripts.ingest.open_law.embedding.pgvector.pgvector_search import PgVectorSearcher
    from scripts.ingest.open_law.embedding.faiss.faiss_search import FaissSearcher
    from scripts.ingest.open_law.embedding.comparison.benchmark import EmbeddingBenchmark
    from scripts.ingest.open_law.embedding.comparison.search_comparison import SearchComparison
    from scripts.ingest.open_law.embedding.comparison.report_generator import ReportGenerator
    from scripts.ingest.open_law.embedding.comparison.test_queries import TEST_QUERIES
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.embedding.pgvector.pgvector_search import PgVectorSearcher
    from ingest.open_law.embedding.faiss.faiss_search import FaissSearcher
    from ingest.open_law.embedding.comparison.benchmark import EmbeddingBenchmark
    from ingest.open_law.embedding.comparison.search_comparison import SearchComparison
    from ingest.open_law.embedding.comparison.report_generator import ReportGenerator
    from ingest.open_law.embedding.comparison.test_queries import TEST_QUERIES

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
    parser = argparse.ArgumentParser(description='FAISS vs pgvector 비교 테스트')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--faiss-index',
        type=Path,
        required=True,
        help='FAISS 인덱스 파일 경로 또는 디렉토리'
    )
    parser.add_argument(
        '--data-type',
        choices=['precedents', 'statutes'],
        default='precedents',
        help='비교할 데이터 타입'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/comparison'),
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='검색 결과 수'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='각 쿼리 반복 횟수'
    )
    parser.add_argument(
        '--queries',
        nargs='+',
        default=None,
        help='테스트 쿼리 리스트 (지정하지 않으면 기본 쿼리 사용)'
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
    
    if not args.faiss_index.exists():
        logger.error(f"FAISS 인덱스 파일이 존재하지 않습니다: {args.faiss_index}")
        return
    
    queries = args.queries if args.queries else TEST_QUERIES
    
    try:
        # 검색 엔진 초기화
        logger.info("검색 엔진 초기화 중...")
        pgvector_searcher = PgVectorSearcher(args.db, model_name=args.model)
        faiss_searcher = FaissSearcher(args.faiss_index, args.db, model_name=args.model)
        
        # 벤치마크 실행
        logger.info("벤치마크 실행 중...")
        benchmark = EmbeddingBenchmark()
        benchmark_results = benchmark.run_full_benchmark(
            pgvector_searcher,
            faiss_searcher,
            queries=queries
        )
        
        # 검색 결과 비교
        logger.info("검색 결과 비교 중...")
        comparison = SearchComparison()
        comparison_results = comparison.compare_all_queries(
            queries,
            pgvector_searcher,
            faiss_searcher,
            top_k=args.top_k
        )
        
        # 리포트 생성
        logger.info("리포트 생성 중...")
        report_generator = ReportGenerator(args.output_dir)
        
        all_results = {
            "benchmark": benchmark_results,
            "comparison": comparison_results
        }
        
        report_generator.generate_performance_report(benchmark_results)
        report_generator.generate_comparison_report(comparison_results)
        report_generator.generate_summary_report(all_results)
        
        logger.info("비교 테스트 완료")
        logger.info(f"리포트 저장 위치: {args.output_dir}")
        
        # 간단한 요약 출력
        print("\n=== 비교 테스트 요약 ===")
        print(f"pgvector 평균 검색 시간: {benchmark_results['pgvector']['avg_time']:.4f}초")
        print(f"FAISS 평균 검색 시간: {benchmark_results['faiss']['avg_time']:.4f}초")
        print(f"평균 결과 일치도: {comparison_results['avg_overlap_ratio']:.2%}")
        print(f"평균 순위 상관관계: {comparison_results['avg_rank_correlation']:.4f}")
    
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

