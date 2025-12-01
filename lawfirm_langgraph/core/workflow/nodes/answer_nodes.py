# -*- coding: utf-8 -*-
"""
Answer Nodes
답변 생성 관련 워크플로우 노드들
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


class AnswerNodes:
    """답변 생성 관련 노드 클래스"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AnswerNodes 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스 (Mixin 메서드 사용)
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def generate_and_validate_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 생성 및 검증"""
        if self.workflow:
            return self.workflow.generate_and_validate_answer(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def generate_answer_stream(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """스트리밍 답변 생성"""
        if self.workflow:
            return self.workflow.generate_answer_stream(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def generate_answer_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 답변 생성"""
        if self.workflow:
            return self.workflow.generate_answer_final(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def continue_answer_generation(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 생성 계속"""
        if self.workflow:
            return self.workflow.continue_answer_generation(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
