# -*- coding: utf-8 -*-
"""
Agentic Routes
Agentic 모드 관련 라우팅 함수들
"""

import logging
from typing import Optional

from core.agents.state_definitions import LegalWorkflowState
from core.agents.workflow_utils import WorkflowUtils


logger = logging.getLogger(__name__)


class AgenticRoutes:
    """Agentic 모드 관련 라우팅 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        AgenticRoutes 초기화
        
        Args:
            logger_instance: 로거 인스턴스
        """
        self.logger = logger_instance or logger
    
    def route_after_agentic(self, state: LegalWorkflowState) -> str:
        """
        Agentic 노드 실행 후 라우팅 (검색 결과 유무에 따라)
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "has_results" 또는 "no_results"
        """
        search_results = WorkflowUtils.get_state_value(state, "search", {}).get("results", [])
        if not search_results:
            # 다른 위치에서도 검색 결과 확인
            retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])
            semantic_results = WorkflowUtils.get_state_value(state, "semantic_results", [])
            keyword_results = WorkflowUtils.get_state_value(state, "keyword_results", [])
            
            if retrieved_docs or semantic_results or keyword_results:
                return "has_results"
            return "no_results"
        
        return "has_results"

