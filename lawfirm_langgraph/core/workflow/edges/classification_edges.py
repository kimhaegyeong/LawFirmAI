# -*- coding: utf-8 -*-
"""
Classification Edges
분류 관련 엣지 정의
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState


logger = logging.getLogger(__name__)


class ClassificationEdges:
    """분류 관련 엣지 빌더"""
    
    def __init__(
        self,
        route_by_complexity_func=None,
        route_by_complexity_with_agentic_func=None,
        should_analyze_document_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        ClassificationEdges 초기화
        
        Args:
            route_by_complexity_func: 복잡도 기반 라우팅 함수
            route_by_complexity_with_agentic_func: Agentic 모드용 복잡도 라우팅 함수
            should_analyze_document_func: 문서 분석 필요 여부 결정 함수
            logger_instance: 로거 인스턴스
        """
        self.route_by_complexity_func = route_by_complexity_func
        self.route_by_complexity_with_agentic_func = route_by_complexity_with_agentic_func
        self.should_analyze_document_func = should_analyze_document_func
        self.logger = logger_instance or logger
    
    def add_classification_edges(
        self,
        workflow: StateGraph,
        use_agentic_mode: bool = False
    ) -> None:
        """
        분류 관련 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
            use_agentic_mode: Agentic 모드 사용 여부
        """
        if use_agentic_mode and self.route_by_complexity_with_agentic_func:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self.route_by_complexity_with_agentic_func,
                {
                    "ethical_reject": "ethical_rejection",
                    "simple": "direct_answer_node",
                    "moderate": "classification_parallel",
                    "complex": "agentic_decision",
                }
            )
        elif self.route_by_complexity_func:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self.route_by_complexity_func,
                {
                    "ethical_reject": "ethical_rejection",
                    "simple": "direct_answer_node",
                    "moderate": "classification_parallel",
                    "complex": "classification_parallel",
                }
            )
        else:
            self.logger.warning("라우팅 함수가 설정되지 않았습니다")
    
    def add_document_analysis_edges(self, workflow: StateGraph) -> None:
        """
        문서 분석 관련 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
        """
        if self.should_analyze_document_func:
            workflow.add_conditional_edges(
                "route_expert",
                self.should_analyze_document_func,
                {
                    "analyze": "analyze_document",
                    "skip": "expand_keywords"
                }
            )
        else:
            # 기본 엣지 (문서 분석 없이 바로 검색)
            workflow.add_edge("route_expert", "expand_keywords")

