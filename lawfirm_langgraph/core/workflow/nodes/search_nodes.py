# -*- coding: utf-8 -*-
"""
Search Nodes
검색 관련 워크플로우 노드들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class SearchNodes:
    """검색 관련 노드 클래스"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        SearchNodes 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스 (Mixin 메서드 사용)
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def expand_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 확장"""
        if self.workflow:
            return self.workflow.expand_keywords(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def prepare_search_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 쿼리 준비"""
        if self.workflow:
            return self.workflow.prepare_search_query(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def execute_searches_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """병렬 검색 실행"""
        if self.workflow:
            return self.workflow.execute_searches_parallel(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def process_search_results_combined(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 처리 통합"""
        if self.workflow:
            return self.workflow.process_search_results_combined(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
