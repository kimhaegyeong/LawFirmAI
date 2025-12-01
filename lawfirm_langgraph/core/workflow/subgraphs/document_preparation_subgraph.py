# -*- coding: utf-8 -*-
"""
Document Preparation Subgraph
문서 준비 관련 노드들을 서브그래프로 구성
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

from langgraph.graph import END, StateGraph

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class DocumentPreparationSubgraph:
    """문서 준비 서브그래프"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        DocumentPreparationSubgraph 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def build_subgraph(self) -> StateGraph:
        """
        문서 준비 서브그래프 구축
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)
        
        # 노드 추가
        if self.workflow:
            subgraph.add_node("prepare_documents_and_terms", self.workflow.prepare_documents_and_terms)
        else:
            raise RuntimeError("workflow_instance가 설정되지 않았습니다")
        
        # 엣지 설정
        subgraph.set_entry_point("prepare_documents_and_terms")
        subgraph.add_edge("prepare_documents_and_terms", END)
        
        return subgraph.compile()

