# -*- coding: utf-8 -*-
"""
Agentic Edges
Agentic 모드 관련 엣지 정의
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

from langgraph.graph import StateGraph

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class AgenticEdges:
    """Agentic 모드 관련 엣지 빌더"""
    
    def __init__(
        self,
        route_after_agentic_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AgenticEdges 초기화
        
        Args:
            route_after_agentic_func: Agentic 노드 후 라우팅 함수
            logger_instance: 로거 인스턴스
        """
        self.route_after_agentic_func = route_after_agentic_func
        self.logger = logger_instance or logger
    
    def add_agentic_edges(
        self,
        workflow: StateGraph,
        answer_generation_node: str = "generate_and_validate_answer"
    ) -> None:
        """
        Agentic 모드 관련 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
            answer_generation_node: 답변 생성 노드 이름
        """
        if self.route_after_agentic_func:
            workflow.add_conditional_edges(
                "agentic_decision",
                self.route_after_agentic_func,
                {
                    "has_results": "prepare_documents_and_terms",
                    "no_results": answer_generation_node,
                }
            )
        else:
            self.logger.warning("route_after_agentic_func가 설정되지 않았습니다")

