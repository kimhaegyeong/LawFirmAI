# -*- coding: utf-8 -*-
"""
Answer Generation Subgraph
답변 생성 관련 노드들을 서브그래프로 구성
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


class AnswerGenerationSubgraph:
    """답변 생성 서브그래프"""
    
    def __init__(
        self,
        workflow_instance=None,
        should_retry_validation_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AnswerGenerationSubgraph 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스
            should_retry_validation_func: 검증 후 재시도 여부 결정 함수
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.should_retry_validation_func = should_retry_validation_func
        self.logger = logger_instance or logger
    
    def build_subgraph(self, answer_node_name: str = "generate_and_validate_answer") -> StateGraph:
        """
        답변 생성 서브그래프 구축
        
        Args:
            answer_node_name: 답변 생성 노드 이름
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)
        
        # 노드 추가
        if self.workflow:
            if answer_node_name == "generate_answer_stream":
                subgraph.add_node("generate_answer_stream", self.workflow.generate_answer_stream)
                if hasattr(self.workflow, 'generate_answer_final'):
                    subgraph.add_node("generate_answer_final", self.workflow.generate_answer_final)
            elif answer_node_name == "generate_answer_final":
                subgraph.add_node("generate_answer_final", self.workflow.generate_answer_final)
            else:
                subgraph.add_node("generate_and_validate_answer", self.workflow.generate_and_validate_answer)
        else:
            raise RuntimeError("workflow_instance가 설정되지 않았습니다")
        
        # 엣지 설정
        entry_point_name = None
        if answer_node_name == "generate_answer_stream" and hasattr(self.workflow, 'generate_answer_final'):
            entry_point_name = "generate_answer_stream"
            subgraph.set_entry_point(entry_point_name)
            # 스트리밍 노드 후 최종 노드로 (조건부)
            subgraph.add_edge("generate_answer_stream", "generate_answer_final")
            
            # 검증 및 재시도 로직
            if self.should_retry_validation_func:
                subgraph.add_conditional_edges(
                    "generate_answer_final",
                    self.should_retry_validation_func,
                    {
                        "accept": END,
                        "retry_generate": "generate_answer_stream",
                        "retry_search": END  # 서브그래프에서는 END로
                    }
                )
            else:
                subgraph.add_edge("generate_answer_final", END)
        else:
            if answer_node_name == "generate_answer_final":
                entry_point_name = "generate_answer_final"
            else:
                entry_point_name = "generate_and_validate_answer"
            
            subgraph.set_entry_point(entry_point_name)
            
            # 검증 및 재시도 로직
            if self.should_retry_validation_func:
                node_name = answer_node_name if answer_node_name != "generate_and_validate_answer" else "generate_and_validate_answer"
                subgraph.add_conditional_edges(
                    node_name,
                    self.should_retry_validation_func,
                    {
                        "accept": END,
                        "retry_generate": node_name,
                        "retry_search": END  # 서브그래프에서는 END로
                    }
                )
            else:
                subgraph.add_edge(entry_point_name, END)
        
        return subgraph.compile()

