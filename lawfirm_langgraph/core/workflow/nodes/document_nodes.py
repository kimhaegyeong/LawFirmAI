# -*- coding: utf-8 -*-
"""
Document Nodes
문서 관련 워크플로우 노드들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class DocumentNodes:
    """문서 관련 노드 클래스"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        DocumentNodes 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스 (Mixin 메서드 사용)
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def analyze_document(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 분석"""
        if self.workflow:
            return self.workflow.analyze_document(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def prepare_documents_and_terms(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 및 용어 준비"""
        if self.workflow:
            return self.workflow.prepare_documents_and_terms(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")

