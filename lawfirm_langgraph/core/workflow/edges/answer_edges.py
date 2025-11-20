# -*- coding: utf-8 -*-
"""
Answer Edges
답변 생성 관련 엣지 정의
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState


logger = logging.getLogger(__name__)


class AnswerEdges:
    """답변 생성 관련 엣지 빌더"""
    
    def __init__(
        self,
        should_retry_validation_func=None,
        should_skip_final_node_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        AnswerEdges 초기화
        
        Args:
            should_retry_validation_func: 검증 후 재시도 여부 결정 함수
            should_skip_final_node_func: 최종 노드 스킵 여부 결정 함수
            logger_instance: 로거 인스턴스
        """
        self.should_retry_validation_func = should_retry_validation_func
        self.should_skip_final_node_func = should_skip_final_node_func
        self.logger = logger_instance or logger
    
    def add_answer_generation_edges(
        self,
        workflow: StateGraph,
        answer_node: str = "generate_and_validate_answer"
    ) -> None:
        """
        답변 생성 관련 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
            answer_node: 답변 생성 노드 이름
        """
        # 문서 준비 후 답변 생성
        workflow.add_edge("prepare_documents_and_terms", answer_node)
        
        # 답변 생성 노드 라우팅
        if answer_node == "generate_answer_stream":
            # 스트리밍 모드
            if self.should_skip_final_node_func:
                workflow.add_conditional_edges(
                    "generate_answer_stream",
                    self.should_skip_final_node_func,
                    {
                        "skip": END,
                        "finalize": "generate_answer_final"
                    }
                )
            
            if self.should_retry_validation_func:
                workflow.add_conditional_edges(
                    "generate_answer_final",
                    self.should_retry_validation_func,
                    {
                        "accept": END,
                        "retry_generate": "generate_answer_stream",
                        "retry_search": "expand_keywords"
                    }
                )
            else:
                workflow.add_edge("generate_answer_final", END)
        else:
            # 일반 모드 또는 최종 노드만 사용
            if self.should_retry_validation_func:
                workflow.add_conditional_edges(
                    answer_node,
                    self.should_retry_validation_func,
                    {
                        "accept": END,
                        "retry_generate": answer_node,
                        "retry_search": "expand_keywords"
                    }
                )
            else:
                workflow.add_edge(answer_node, END)
    
    def add_direct_answer_edge(self, workflow: StateGraph) -> None:
        """
        직접 답변 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
        """
        workflow.add_edge("direct_answer_node", END)

