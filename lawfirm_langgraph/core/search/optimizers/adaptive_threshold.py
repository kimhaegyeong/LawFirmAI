# -*- coding: utf-8 -*-
"""
Adaptive Threshold
적응형 유사도 임계값 계산기
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__)


class AdaptiveThreshold:
    """적응형 유사도 임계값 계산기"""
    
    def __init__(self, base_threshold: float = 0.5):
        """
        초기화
        
        Args:
            base_threshold: 기본 임계값
        """
        self.logger = get_logger(__name__)
        self.base_threshold = base_threshold
        self.logger.info(f"AdaptiveThreshold initialized (base={base_threshold})")
    
    def calculate_threshold(
        self,
        query: str,
        query_type: str = "general_question",
        result_count: int = 0,
        min_results: int = 5,
        max_results: int = 20
    ) -> float:
        """
        적응형 임계값 계산
        
        Args:
            query: 검색 쿼리
            query_type: 질문 유형
            result_count: 현재 결과 수
            min_results: 최소 결과 수
            max_results: 최대 결과 수
        
        Returns:
            float: 계산된 임계값
        """
        try:
            threshold = self.base_threshold
            
            # 1. 질문 유형별 기본 임계값 조정
            type_thresholds = {
                "law_inquiry": 0.55,  # 법령 조문은 조금 높은 임계값
                "precedent_search": 0.5,  # 판례는 기본 임계값
                "procedure_inquiry": 0.45,  # 절차는 낮은 임계값 (다양한 결과 필요)
                "legal_advice": 0.5,  # 법률 자문은 기본 임계값
            }
            
            if query_type in type_thresholds:
                threshold = type_thresholds[query_type]
            
            # 2. 결과 수에 따른 동적 조정
            if result_count < min_results:
                # 결과가 부족하면 임계값 낮춤
                threshold = max(0.2, threshold - 0.1)
                self.logger.debug(f"Lowering threshold due to low result count: {threshold:.2f}")
            elif result_count > max_results:
                # 결과가 너무 많으면 임계값 높임
                threshold = min(0.8, threshold + 0.1)
                self.logger.debug(f"Raising threshold due to high result count: {threshold:.2f}")
            
            # 3. 쿼리 길이 기반 조정
            query_length = len(query)
            if query_length < 10:
                # 짧은 쿼리는 낮은 임계값 (다양한 결과)
                threshold = max(0.3, threshold - 0.05)
            elif query_length > 100:
                # 긴 쿼리는 높은 임계값 (정확한 결과)
                threshold = min(0.7, threshold + 0.05)
            
            self.logger.debug(
                f"Adaptive threshold calculated: {threshold:.3f} "
                f"(type={query_type}, results={result_count})"
            )
            
            return threshold
        
        except Exception as e:
            self.logger.error(f"Threshold calculation failed: {e}")
            return self.base_threshold
    
    def adjust_threshold_for_retry(
        self,
        current_threshold: float,
        retry_count: int,
        max_retries: int = 3
    ) -> float:
        """
        재시도 시 임계값 조정
        
        Args:
            current_threshold: 현재 임계값
            retry_count: 재시도 횟수
            max_retries: 최대 재시도 횟수
        
        Returns:
            float: 조정된 임계값
        """
        if retry_count >= max_retries:
            return 0.2  # 최소 임계값
        
        # 재시도할 때마다 임계값을 0.1씩 낮춤
        new_threshold = max(0.2, current_threshold - 0.1 * retry_count)
        
        self.logger.debug(
            f"Threshold adjusted for retry: {current_threshold:.3f} -> {new_threshold:.3f} "
            f"(retry={retry_count})"
        )
        
        return new_threshold

