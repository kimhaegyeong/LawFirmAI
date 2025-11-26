# -*- coding: utf-8 -*-
"""
검색 결과 비교
FAISS vs pgvector 검색 결과 일치도 및 순위 비교
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)


class SearchComparison:
    """검색 결과 비교"""
    
    def __init__(self):
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
    
    def compare_results(
        self,
        query: str,
        pgvector_results: List[Dict],
        faiss_results: List[Dict],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        검색 결과 비교
        
        Args:
            query: 검색 쿼리
            pgvector_results: pgvector 검색 결과
            faiss_results: FAISS 검색 결과
            top_k: 비교할 상위 k개
        
        Returns:
            비교 결과
        """
        # ID 추출
        pgvector_ids = [r["id"] for r in pgvector_results[:top_k]]
        faiss_ids = [r["id"] for r in faiss_results[:top_k]]
        
        # Overlap 계산
        overlap = self.calculate_overlap(
            pgvector_results[:top_k],
            faiss_results[:top_k],
            top_k
        )
        
        # 순위 상관관계 계산
        rank_correlation = self.calculate_rank_correlation(
            pgvector_results[:top_k],
            faiss_results[:top_k]
        )
        
        # 스코어 분포
        pgvector_scores = [r.get("similarity", 0) for r in pgvector_results[:top_k]]
        faiss_scores = [r.get("similarity", 0) for r in faiss_results[:top_k]]
        
        return {
            "query": query,
            "top_k": top_k,
            "overlap": overlap,
            "overlap_ratio": overlap / top_k if top_k > 0 else 0,
            "rank_correlation": rank_correlation,
            "pgvector_ids": pgvector_ids,
            "faiss_ids": faiss_ids,
            "pgvector_scores": pgvector_scores,
            "faiss_scores": faiss_scores,
            "pgvector_avg_score": sum(pgvector_scores) / len(pgvector_scores) if pgvector_scores else 0,
            "faiss_avg_score": sum(faiss_scores) / len(faiss_scores) if faiss_scores else 0
        }
    
    def calculate_overlap(
        self,
        results1: List[Dict],
        results2: List[Dict],
        top_k: int = 10
    ) -> float:
        """
        검색 결과 일치도 계산 (Top-K overlap)
        
        Args:
            results1: 첫 번째 검색 결과
            results2: 두 번째 검색 결과
            top_k: 비교할 상위 k개
        
        Returns:
            일치하는 결과 수
        """
        ids1 = set(r["id"] for r in results1[:top_k])
        ids2 = set(r["id"] for r in results2[:top_k])
        
        return len(ids1 & ids2)
    
    def calculate_rank_correlation(
        self,
        results1: List[Dict],
        results2: List[Dict]
    ) -> float:
        """
        순위 상관관계 계산 (Kendall's tau)
        
        Args:
            results1: 첫 번째 검색 결과
            results2: 두 번째 검색 결과
        
        Returns:
            Kendall's tau 값 (-1 ~ 1)
        """
        # ID를 순위로 변환
        rank1 = {r["id"]: i for i, r in enumerate(results1)}
        rank2 = {r["id"]: i for i, r in enumerate(results2)}
        
        # 공통 ID 찾기
        common_ids = set(rank1.keys()) & set(rank2.keys())
        
        if len(common_ids) < 2:
            return 0.0
        
        # Kendall's tau 계산
        concordant = 0
        discordant = 0
        
        common_ids_list = list(common_ids)
        for i in range(len(common_ids_list)):
            for j in range(i + 1, len(common_ids_list)):
                id1, id2 = common_ids_list[i], common_ids_list[j]
                
                rank1_diff = rank1[id1] - rank1[id2]
                rank2_diff = rank2[id1] - rank2[id2]
                
                if rank1_diff * rank2_diff > 0:
                    concordant += 1
                elif rank1_diff * rank2_diff < 0:
                    discordant += 1
        
        total_pairs = len(common_ids) * (len(common_ids) - 1) / 2
        
        if total_pairs == 0:
            return 0.0
        
        tau = (concordant - discordant) / total_pairs
        return tau
    
    def compare_all_queries(
        self,
        queries: List[str],
        pgvector_searcher,
        faiss_searcher,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        모든 쿼리에 대해 비교 실행
        
        Args:
            queries: 테스트 쿼리 리스트
            pgvector_searcher: pgvector 검색 엔진
            faiss_searcher: FAISS 검색 엔진
            top_k: 비교할 상위 k개
        
        Returns:
            전체 비교 결과
        """
        comparisons = []
        
        for query in queries:
            self.logger.info(f"쿼리 비교 중: {query}")
            
            # 검색 실행
            pgvector_results = pgvector_searcher.search(query, top_k=top_k)
            faiss_results = faiss_searcher.search(query, top_k=top_k)
            
            # 비교
            comparison = self.compare_results(
                query,
                pgvector_results,
                faiss_results,
                top_k
            )
            comparisons.append(comparison)
        
        # 전체 통계
        avg_overlap = sum(c["overlap"] for c in comparisons) / len(comparisons) if comparisons else 0
        avg_overlap_ratio = sum(c["overlap_ratio"] for c in comparisons) / len(comparisons) if comparisons else 0
        avg_rank_correlation = sum(c["rank_correlation"] for c in comparisons) / len(comparisons) if comparisons else 0
        
        return {
            "total_queries": len(queries),
            "top_k": top_k,
            "avg_overlap": avg_overlap,
            "avg_overlap_ratio": avg_overlap_ratio,
            "avg_rank_correlation": avg_rank_correlation,
            "comparisons": comparisons
        }

