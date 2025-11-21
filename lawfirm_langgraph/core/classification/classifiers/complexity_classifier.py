# -*- coding: utf-8 -*-
"""
Complexity Classifier
질문 복잡도 분류기
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from enum import Enum
from typing import Tuple

from core.workflow.state.workflow_types import QueryComplexity

logger = get_logger(__name__)


class ComplexityClassifier:
    """질문 복잡도 분류기"""

    @staticmethod
    def classify_complexity(query: str) -> Tuple[QueryComplexity, bool]:
        """
        질문 복잡도 분류

        Args:
            query: 질문 텍스트

        Returns:
            (복잡도, 검색 필요 여부)
        """
        if not query or len(query.strip()) < 5:
            return QueryComplexity.MODERATE, True

        query_lower = query.lower()
        query_length = len(query)

        # 간단한 질문 패턴
        simple_patterns = [
            "안녕", "고마워", "감사", "뭐야", "뭔가", "무엇",
            "정의", "의미", "뜻", "이게 뭐"
        ]

        # 복잡한 질문 패턴
        complex_patterns = [
            "판례", "법원", "소송", "계약서", "고소장",
            "절차", "방법", "어떻게", "과정", "단계"
        ]

        # 간단한 질문 체크
        if any(pattern in query_lower for pattern in simple_patterns):
            if query_length < 30:
                return QueryComplexity.SIMPLE, False

        # 복잡한 질문 체크
        if any(pattern in query_lower for pattern in complex_patterns):
            return QueryComplexity.COMPLEX, True

        # 질문 길이 기반 분류
        if query_length < 20:
            return QueryComplexity.SIMPLE, False
        elif query_length > 100:
            return QueryComplexity.COMPLEX, True
        else:
            return QueryComplexity.MODERATE, True

    @staticmethod
    def classify_with_llm(query: str, llm_client=None) -> Tuple[QueryComplexity, bool]:
        """
        LLM을 사용한 복잡도 분류 (향후 구현)

        Args:
            query: 질문 텍스트
            llm_client: LLM 클라이언트 (선택적)

        Returns:
            (복잡도, 검색 필요 여부)
        """
        # 현재는 기본 분류 사용
        return ComplexityClassifier.classify_complexity(query)

