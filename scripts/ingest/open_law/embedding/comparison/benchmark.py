# -*- coding: utf-8 -*-
"""
성능 벤치마크
FAISS vs pgvector 성능 측정
"""

import logging
import time
from typing import List, Dict, Any, Optional
import statistics
import psutil
import os

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

try:
    from scripts.ingest.open_law.embedding.comparison.test_queries import TEST_QUERIES
except ImportError:
    import sys
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from ingest.open_law.embedding.comparison.test_queries import TEST_QUERIES

logger = get_logger(__name__)


class EmbeddingBenchmark:
    """임베딩 성능 벤치마크"""
    
    def __init__(self):
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
    
    def benchmark_embedding_generation(
        self,
        embedder,
        texts: List[str],
        batch_size: int = 100,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        임베딩 생성 성능 측정
        
        Args:
            embedder: 임베딩 생성기
            texts: 테스트 텍스트 리스트
            batch_size: 배치 크기
            iterations: 반복 횟수
        
        Returns:
            성능 통계
        """
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # 메모리 사용량 측정 시작
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 임베딩 생성
            start_time = time.time()
            embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=False)
            end_time = time.time()
            
            # 메모리 사용량 측정 종료
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            memory_usage.append(mem_after - mem_before)
        
        return {
            "total_texts": len(texts),
            "batch_size": batch_size,
            "iterations": iterations,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_memory_mb": statistics.mean(memory_usage),
            "texts_per_second": len(texts) / statistics.mean(times)
        }
    
    def benchmark_index_building(
        self,
        indexer,
        embeddings,
        index_type: str = "ivfflat"
    ) -> Dict[str, Any]:
        """
        인덱스 빌드 성능 측정
        
        Args:
            indexer: 인덱스 생성기
            embeddings: 임베딩 배열
            index_type: 인덱스 타입
        
        Returns:
            성능 통계
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        index = indexer.build_index(embeddings, index_type=index_type)
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "total_vectors": embeddings.shape[0],
            "dimension": embeddings.shape[1],
            "index_type": index_type,
            "build_time": end_time - start_time,
            "memory_usage_mb": mem_after - mem_before,
            "vectors_per_second": embeddings.shape[0] / (end_time - start_time)
        }
    
    def benchmark_search(
        self,
        searcher,
        queries: List[str],
        top_k: int = 10,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        검색 성능 측정
        
        Args:
            searcher: 검색 엔진
            queries: 테스트 쿼리 리스트
            top_k: 반환할 결과 수
            iterations: 각 쿼리 반복 횟수
        
        Returns:
            성능 통계
        """
        all_times = []
        per_query_times = {query: [] for query in queries}
        
        for query in queries:
            query_times = []
            
            for _ in range(iterations):
                start_time = time.time()
                results = searcher.search(query, top_k=top_k)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                query_times.append(elapsed_time)
                all_times.append(elapsed_time)
            
            per_query_times[query] = query_times
        
        # 통계 계산
        sorted_times = sorted(all_times)
        n = len(sorted_times)
        
        return {
            "total_queries": len(queries),
            "iterations_per_query": iterations,
            "top_k": top_k,
            "avg_time": statistics.mean(all_times),
            "min_time": min(all_times),
            "max_time": max(all_times),
            "p50_time": sorted_times[n // 2] if n > 0 else 0,
            "p95_time": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99_time": sorted_times[int(n * 0.99)] if n > 0 else 0,
            "std_time": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "qps": len(queries) * iterations / sum(all_times) if sum(all_times) > 0 else 0,
            "per_query_times": per_query_times
        }
    
    def run_full_benchmark(
        self,
        pgvector_searcher,
        faiss_searcher,
        queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        전체 벤치마크 실행
        
        Args:
            pgvector_searcher: pgvector 검색 엔진
            faiss_searcher: FAISS 검색 엔진
            queries: 테스트 쿼리 리스트 (None이면 기본 쿼리 사용)
        
        Returns:
            벤치마크 결과
        """
        if queries is None:
            queries = TEST_QUERIES
        
        self.logger.info("전체 벤치마크 시작")
        
        # pgvector 검색 벤치마크
        self.logger.info("pgvector 검색 벤치마크 실행 중...")
        pgvector_results = self.benchmark_search(
            pgvector_searcher,
            queries,
            top_k=10,
            iterations=10
        )
        
        # FAISS 검색 벤치마크
        self.logger.info("FAISS 검색 벤치마크 실행 중...")
        faiss_results = self.benchmark_search(
            faiss_searcher,
            queries,
            top_k=10,
            iterations=10
        )
        
        return {
            "pgvector": pgvector_results,
            "faiss": faiss_results,
            "queries": queries
        }

