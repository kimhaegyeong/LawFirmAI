"""
성능 모니터링 테스트 스크립트

실제 검색 성능을 측정하고 버전별로 비교합니다.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from scripts.utils.version_performance_monitor import VersionPerformanceMonitor
from scripts.utils.faiss_version_manager import FAISSVersionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_search_performance(
    db_path: str,
    queries: list,
    faiss_version: str = None,
    num_iterations: int = 5
):
    """
    검색 성능 테스트
    
    Args:
        db_path: 데이터베이스 경로
        queries: 테스트 쿼리 리스트
        faiss_version: FAISS 버전 (None이면 활성 버전)
        num_iterations: 반복 횟수
    """
    logger.info("=" * 80)
    logger.info("Performance Monitoring Test")
    logger.info("=" * 80)
    
    # 검색 엔진 초기화
    engine = SemanticSearchEngineV2(
        db_path=db_path,
        use_external_index=False
    )
    
    # 성능 모니터 초기화
    monitor = VersionPerformanceMonitor()
    
    # FAISS 버전 관리자에서 활성 버전 조회
    from scripts.utils.faiss_version_manager import FAISSVersionManager
    faiss_manager = FAISSVersionManager("data/vector_store")
    active_version = faiss_manager.get_active_version()
    
    version_name = faiss_version or active_version or engine.current_faiss_version or "default"
    logger.info(f"Testing version: {version_name}")
    logger.info(f"Number of queries: {len(queries)}")
    logger.info(f"Iterations per query: {num_iterations}")
    logger.info("")
    
    results_summary = []
    
    for query_idx, query in enumerate(queries, 1):
        logger.info(f"Query {query_idx}/{len(queries)}: {query}")
        
        latencies = []
        relevance_scores = []
        
        for iteration in range(num_iterations):
            query_id = f"query_{query_idx}_iter_{iteration}"
            
            # 검색 실행
            start_time = time.time()
            search_results = engine.search(
                query=query,
                k=10,
                faiss_version=faiss_version
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # 평균 관련성 점수 계산
            avg_relevance = sum(r.get('score', 0.0) for r in search_results) / len(search_results) if search_results else 0.0
            
            latencies.append(latency_ms)
            relevance_scores.append(avg_relevance)
            
            # 성능 로깅
            monitor.log_search(
                version=version_name,
                query_id=query_id,
                latency_ms=latency_ms,
                relevance_score=avg_relevance
            )
            
            logger.debug(f"  Iteration {iteration + 1}: {latency_ms:.2f}ms, relevance: {avg_relevance:.4f}")
        
        # 통계 계산
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        results_summary.append({
            'query': query,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'avg_relevance': avg_relevance,
            'num_results': len(search_results) if search_results else 0
        })
        
        logger.info(f"  Average latency: {avg_latency:.2f}ms (min: {min_latency:.2f}ms, max: {max_latency:.2f}ms)")
        logger.info(f"  Average relevance: {avg_relevance:.4f}")
        logger.info(f"  Results count: {len(search_results) if search_results else 0}")
        logger.info("")
    
    # 전체 통계
    logger.info("=" * 80)
    logger.info("Overall Statistics")
    logger.info("=" * 80)
    
    all_latencies = [r['avg_latency_ms'] for r in results_summary]
    all_relevances = [r['avg_relevance'] for r in results_summary]
    
    logger.info(f"Average latency: {sum(all_latencies) / len(all_latencies):.2f}ms")
    logger.info(f"Min latency: {min(all_latencies):.2f}ms")
    logger.info(f"Max latency: {max(all_latencies):.2f}ms")
    logger.info(f"Average relevance: {sum(all_relevances) / len(all_relevances):.4f}")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Total iterations: {len(queries) * num_iterations}")
    
    # 버전별 메트릭 조회
    metrics = monitor.get_version_metrics(version_name)
    if metrics:
        logger.info("")
        logger.info("Version Metrics:")
        logger.info(f"  Total searches: {metrics.get('total_searches', 0)}")
        logger.info(f"  Average latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"  Average relevance: {metrics.get('avg_relevance_score', 0):.4f}")
    
    return results_summary


def compare_versions(
    db_path: str,
    version1: str,
    version2: str,
    queries: list,
    num_iterations: int = 3
):
    """
    두 버전의 성능 비교
    
    Args:
        db_path: 데이터베이스 경로
        version1: 첫 번째 버전
        version2: 두 번째 버전
        queries: 테스트 쿼리 리스트
        num_iterations: 반복 횟수
    """
    logger.info("=" * 80)
    logger.info("Version Comparison Test")
    logger.info("=" * 80)
    logger.info(f"Version 1: {version1}")
    logger.info(f"Version 2: {version2}")
    logger.info("")
    
    # 각 버전 테스트
    logger.info("Testing Version 1...")
    test_search_performance(db_path, queries, version1, num_iterations)
    
    logger.info("Testing Version 2...")
    test_search_performance(db_path, queries, version2, num_iterations)
    
    # 성능 비교
    monitor = VersionPerformanceMonitor()
    comparison = monitor.compare_performance(version1, version2)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Performance Comparison")
    logger.info("=" * 80)
    logger.info(f"Latency improvement: {comparison.get('latency_improvement_percent', 0):.2f}%")
    logger.info(f"Relevance improvement: {comparison.get('relevance_improvement_percent', 0):.2f}%")
    logger.info(f"Better version: {comparison.get('better_version', 'N/A')}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Performance monitoring test")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--version", help="FAISS version to test (default: active version)")
    parser.add_argument("--version1", help="First version for comparison")
    parser.add_argument("--version2", help="Second version for comparison")
    parser.add_argument("--queries", nargs="+", default=[
        "전세금 반환 보증",
        "계약 해지",
        "손해배상",
        "임대차 계약",
        "부동산 매매"
    ], help="Test queries")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per query")
    
    args = parser.parse_args()
    
    if args.version1 and args.version2:
        # 버전 비교 모드
        compare_versions(
            args.db,
            args.version1,
            args.version2,
            args.queries,
            args.iterations
        )
    else:
        # 단일 버전 테스트 모드
        test_search_performance(
            args.db,
            args.queries,
            args.version,
            args.iterations
        )


if __name__ == "__main__":
    main()

