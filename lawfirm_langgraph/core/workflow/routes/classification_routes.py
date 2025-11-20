# -*- coding: utf-8 -*-
"""
Classification Routes
분류 관련 라우팅 함수들
"""

import logging
from typing import Optional

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_utils import WorkflowUtils


logger = logging.getLogger(__name__)


class QueryComplexity:
    """질문 복잡도 Enum 대체 클래스"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class ClassificationRoutes:
    """분류 관련 라우팅 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        ClassificationRoutes 초기화
        
        Args:
            logger_instance: 로거 인스턴스
        """
        self.logger = logger_instance or logger
    
    def route_by_complexity(self, state: LegalWorkflowState) -> str:
        """
        복잡도에 따라 라우팅
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            라우팅 키 ("ethical_reject", "simple", "moderate", "complex")
        """
        # 윤리적 문제 감지 확인 (최우선)
        is_problematic = WorkflowUtils.get_state_value(state, "is_ethically_problematic", False)
        if is_problematic:
            self.logger.warning("윤리적 문제 감지: ethical_reject로 라우팅")
            return "ethical_reject"
        
        # 복잡도 확인
        complexity = None
        
        # 여러 방법으로 complexity 확인
        if isinstance(state, dict) and "query_complexity" in state:
            complexity = state["query_complexity"]
        elif isinstance(state, dict) and "common" in state:
            if isinstance(state["common"], dict):
                complexity = state["common"].get("query_complexity")
        elif isinstance(state, dict) and "metadata" in state:
            if isinstance(state["metadata"], dict):
                complexity = state["metadata"].get("query_complexity")
        elif isinstance(state, dict) and "classification" in state:
            if isinstance(state["classification"], dict):
                complexity = state["classification"].get("query_complexity")
        
        if not complexity:
            complexity = WorkflowUtils.get_state_value(state, "query_complexity", None)
        
        # 기본값
        if not complexity:
            complexity = QueryComplexity.MODERATE
        
        # Enum인 경우 값으로 변환
        if hasattr(complexity, 'value'):
            complexity = complexity.value
        
        # 문자열 비교
        if complexity == QueryComplexity.SIMPLE or complexity == "simple":
            self.logger.info(f"✅ [ROUTE] 간단한 질문 → direct_answer")
            return "simple"
        elif complexity == QueryComplexity.MODERATE or complexity == "moderate":
            self.logger.info(f"🔄 [ROUTE] 중간 질문 → classification_parallel")
            return "moderate"
        else:
            self.logger.info(f"🔀 [ROUTE] 복잡한 질문 → classification_parallel")
            return "complex"
    
    def route_by_complexity_with_agentic(self, state: LegalWorkflowState) -> str:
        """
        Agentic 모드용 복잡도 라우팅
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            라우팅 키
        """
        # 윤리적 문제 감지 확인 (최우선)
        is_problematic = WorkflowUtils.get_state_value(state, "is_ethically_problematic", False)
        if is_problematic:
            return "ethical_reject"
        
        # 기본 복잡도 라우팅 사용
        route = self.route_by_complexity(state)
        
        # complex인 경우 agentic_decision으로 라우팅
        if route == "complex":
            return "complex"  # agentic_decision으로 라우팅됨
        
        return route

