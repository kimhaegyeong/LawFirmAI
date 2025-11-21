# -*- coding: utf-8 -*-
"""
Classification Subgraph
분류 관련 노드들을 서브그래프로 구성
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

from langgraph.graph import END, StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class ClassificationSubgraph:
    """분류 서브그래프"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        ClassificationSubgraph 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def build_subgraph(self) -> StateGraph:
        """
        분류 서브그래프 구축
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)
        
        # 노드 추가
        if self.workflow:
            subgraph.add_node("assess_urgency", self.workflow.assess_urgency)
            subgraph.add_node("resolve_multi_turn", self.workflow.resolve_multi_turn)
            subgraph.add_node("route_expert", self.workflow.route_expert)
        else:
            raise RuntimeError("workflow_instance가 설정되지 않았습니다")
        
        # 엣지 설정
        subgraph.set_entry_point("assess_urgency")
        subgraph.add_edge("assess_urgency", "resolve_multi_turn")
        subgraph.add_edge("resolve_multi_turn", "route_expert")
        subgraph.add_edge("route_expert", END)
        
        return subgraph.compile()

