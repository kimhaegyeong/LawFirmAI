# -*- coding: utf-8 -*-
"""
Agentic Nodes
Agentic AI 관련 워크플로우 노드들
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


class AgenticNodes:
    """Agentic AI 관련 노드 클래스"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AgenticNodes 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스 (Mixin 메서드 사용)
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def agentic_decision_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """Agentic 결정 노드"""
        if self.workflow:
            return self.workflow.agentic_decision_node(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")

