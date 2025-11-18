# -*- coding: utf-8 -*-
"""
Adaptive Hybrid Weights
하이브리드 검색 동적 가중치 계산기
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AdaptiveHybridWeights:
    """하이브리드 검색 동적 가중치 계산기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("AdaptiveHybridWeights initialized")
    
    def calculate_weights(
        self,
        query: str,
        query_type: str = "general_question",
        keyword_coverage: float = 0.5,
        query_complexity: Optional[str] = None
    ) -> Dict[str, float]:
        """
        동적 가중치 계산
        
        Args:
            query: 검색 쿼리
            query_type: 질문 유형
            keyword_coverage: Keyword Coverage 점수 (0.0-1.0)
            query_complexity: 쿼리 복잡도 ("simple", "medium", "complex")
        
        Returns:
            Dict[str, float]: semantic과 keyword 가중치
        """
        try:
            # 기본 가중치
            base_semantic = 0.6
            base_keyword = 0.4
            
            # 1. Keyword Coverage 기반 조정
            if keyword_coverage < 0.3:
                # 키워드 매칭이 매우 낮으면 의미적 검색 강화
                semantic_weight = 0.75
                keyword_weight = 0.25
            elif keyword_coverage < 0.5:
                # 키워드 매칭이 낮으면 의미적 검색 강화
                semantic_weight = 0.7
                keyword_weight = 0.3
            elif keyword_coverage > 0.8:
                # 키워드 매칭이 높으면 키워드 검색 강화
                semantic_weight = 0.4
                keyword_weight = 0.6
            elif keyword_coverage > 0.9:
                # 키워드 매칭이 매우 높으면 키워드 검색 더 강화
                semantic_weight = 0.3
                keyword_weight = 0.7
            else:
                # 중간 범위는 기본 가중치 사용
                semantic_weight = base_semantic
                keyword_weight = base_keyword
            
            # 2. 질문 유형별 조정
            type_adjustments = {
                "law_inquiry": {"semantic": +0.05, "keyword": -0.05},  # 법령 조문은 의미적 검색 강화
                "precedent_search": {"semantic": +0.1, "keyword": -0.1},  # 판례는 의미적 검색 더 강화
                "procedure_inquiry": {"semantic": -0.05, "keyword": +0.05},  # 절차는 키워드 검색 강화
                "legal_advice": {"semantic": +0.05, "keyword": -0.05},  # 법률 자문은 의미적 검색 강화
            }
            
            if query_type in type_adjustments:
                adj = type_adjustments[query_type]
                semantic_weight = max(0.1, min(0.9, semantic_weight + adj["semantic"]))
                keyword_weight = max(0.1, min(0.9, keyword_weight + adj["keyword"]))
            
            # 3. 쿼리 복잡도 기반 조정
            if query_complexity is None:
                query_complexity = self._estimate_complexity(query)
            
            if query_complexity == "simple":
                # 단순 쿼리는 키워드 검색 강화
                semantic_weight = max(0.2, semantic_weight - 0.1)
                keyword_weight = min(0.8, keyword_weight + 0.1)
            elif query_complexity == "complex":
                # 복잡한 쿼리는 의미적 검색 강화
                semantic_weight = min(0.8, semantic_weight + 0.1)
                keyword_weight = max(0.2, keyword_weight - 0.1)
            
            # 정규화 (합이 1.0이 되도록)
            total = semantic_weight + keyword_weight
            semantic_weight = semantic_weight / total
            keyword_weight = keyword_weight / total
            
            self.logger.debug(
                f"Weights calculated: semantic={semantic_weight:.2f}, "
                f"keyword={keyword_weight:.2f} "
                f"(coverage={keyword_coverage:.2f}, type={query_type}, complexity={query_complexity})"
            )
            
            return {
                "semantic": semantic_weight,
                "keyword": keyword_weight
            }
        
        except Exception as e:
            self.logger.error(f"Weight calculation failed: {e}")
            return {"semantic": 0.6, "keyword": 0.4}
    
    def _estimate_complexity(self, query: str) -> str:
        """쿼리 복잡도 추정"""
        query_length = len(query)
        word_count = len(query.split())
        
        # 법률 용어 개수 (간단한 추정)
        legal_terms = ["법", "조문", "판례", "계약", "손해배상", "불법행위"]
        legal_term_count = sum(1 for term in legal_terms if term in query)
        
        if query_length < 20 and word_count < 5 and legal_term_count < 2:
            return "simple"
        elif query_length > 100 or word_count > 15 or legal_term_count > 3:
            return "complex"
        else:
            return "medium"

