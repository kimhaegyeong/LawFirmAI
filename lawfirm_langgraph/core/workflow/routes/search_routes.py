# -*- coding: utf-8 -*-
"""
Search Routes
검색 관련 라우팅 함수들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

try:
    from lawfirm_langgraph.core.agents.state_definitions import LegalWorkflowState
except ImportError:
    from core.agents.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils


logger = get_logger(__name__)


class QueryComplexity:
    """질문 복잡도 Enum 대체 클래스"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class SearchRoutes:
    """검색 관련 라우팅 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        SearchRoutes 초기화
        
        Args:
            logger_instance: 로거 인스턴스
        """
        self.logger = logger_instance or logger
    
    def should_analyze_document(self, state: LegalWorkflowState) -> str:
        """
        문서 분석 필요 여부 결정
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "analyze" 또는 "skip"
        """
        if state.get("uploaded_document"):
            return "analyze"
        return "skip"
    
    def should_skip_search_adaptive(self, state: LegalWorkflowState) -> str:
        """
        Adaptive RAG: 질문 복잡도에 따라 검색 스킵 결정
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "skip" 또는 "continue"
        """
        # 캐시 히트 체크
        cache_hit = WorkflowUtils.get_state_value(state, "search_cache_hit", False)
        if cache_hit:
            return "skip"
        
        # 복잡도 기반 스킵 결정
        needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)
        complexity = WorkflowUtils.get_state_value(state, "query_complexity", QueryComplexity.MODERATE)
        
        # Enum인 경우 값으로 변환
        if hasattr(complexity, 'value'):
            complexity = complexity.value
        
        if not needs_search or complexity == QueryComplexity.SIMPLE or complexity == "simple":
            self.logger.info(f"⏭️ 검색 스킵: 간단한 질문 (복잡도: {complexity})")
            return "skip"
        
        return "continue"
    
    def should_expand_keywords_ai(
        self,
        state: LegalWorkflowState,
        ai_keyword_generator=None
    ) -> str:
        """
        AI 키워드 확장 여부 결정
        
        Args:
            state: 워크플로우 상태
            ai_keyword_generator: AI 키워드 생성기 (선택적)
        
        Returns:
            "expand" 또는 "skip"
        """
        if not ai_keyword_generator:
            return "skip"
        
        keywords = WorkflowUtils.get_state_value(state, "extracted_keywords", [])
        if len(keywords) < 3:
            return "skip"
        
        # 복잡한 질문인 경우 확장
        query_type = WorkflowUtils.get_state_value(state, "query_type", "")
        complex_types = ["precedent_search", "law_inquiry", "legal_advice"]
        
        if query_type in complex_types:
            return "expand"
        
        return "skip"
    
    def should_use_multi_query_agent(self, state: LegalWorkflowState) -> str:
        """
        멀티 질의 검색 에이전트 사용 여부 결정
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "multi_query_agent" 또는 "standard_search"
        """
        query = WorkflowUtils.get_state_value(state, "query", "")
        complexity = WorkflowUtils.get_state_value(state, "query_complexity", QueryComplexity.MODERATE)
        needs_search = WorkflowUtils.get_state_value(state, "needs_search", True)
        
        # Enum인 경우 값으로 변환
        if hasattr(complexity, 'value'):
            complexity = complexity.value
        
        # 검색이 필요하지 않으면 표준 검색으로
        if not needs_search:
            return "standard_search"
        
        # 복잡한 질문인 경우 멀티 질의 에이전트 사용
        if complexity == QueryComplexity.COMPLEX or complexity == "complex":
            self.logger.info(f"🔍 [ROUTING] Using multi-query agent for complex query: '{query[:50]}...'")
            return "multi_query_agent"
        
        # 질문 길이가 길거나 여러 키워드가 포함된 경우
        if query and (len(query) > 25 or len(query.split()) > 4):
            self.logger.info(f"🔍 [ROUTING] Using multi-query agent for long/multi-keyword query: '{query[:50]}...' (length={len(query)}, words={len(query.split())})")
            return "multi_query_agent"
        
        # 질문에 여러 법률 개념이 포함된 경우 (예: "사유", "절차", "효과", "요건" 등)
        legal_concepts = ["사유", "절차", "효과", "요건", "조건", "방법", "절차", "기간", "효력", "무효", "취소"]
        if query and sum(1 for concept in legal_concepts if concept in query) >= 2:
            self.logger.info(f"🔍 [ROUTING] Using multi-query agent for multi-concept query: '{query[:50]}...'")
            return "multi_query_agent"
        
        # 기본값: 표준 검색
        self.logger.debug(f"🔍 [ROUTING] Using standard search for query: '{query[:50]}...' (complexity={complexity}, length={len(query) if query else 0})")
        return "standard_search"

