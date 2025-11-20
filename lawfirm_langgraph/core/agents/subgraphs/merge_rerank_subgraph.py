# -*- coding: utf-8 -*-
"""
병합 및 재순위 서브그래프
검색 결과 병합과 키워드 가중치 적용, 재순위를 처리하는 서브그래프
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from core.agents.state_definitions import LegalWorkflowState
from core.agents.handlers.search_result_processor import SearchResultProcessor


class MergeAndRerankSubgraph:
    """
    병합 및 재순위 서브그래프
    
    검색 결과 병합, 키워드 가중치 적용, 재순위를 처리하는 서브그래프
    """

    def __init__(
        self,
        search_result_processor: SearchResultProcessor,
        logger: Optional[logging.Logger] = None
    ):
        """
        MergeAndRerankSubgraph 초기화
        
        Args:
            search_result_processor: 검색 결과 처리 핸들러
            logger: 로거 (없으면 자동 생성)
        """
        self.processor = search_result_processor
        self.logger = logger or logging.getLogger(__name__)

    def build_subgraph(self) -> StateGraph:
        """
        병합 및 재순위 서브그래프 구축
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)

        # 노드 추가
        subgraph.add_node("merge_results", self.merge_results_node)
        subgraph.add_node("apply_keyword_weights", self.apply_keyword_weights_node)
        subgraph.add_node("rerank_documents", self.rerank_documents_node)

        # 엣지 설정
        subgraph.set_entry_point("merge_results")
        subgraph.add_edge("merge_results", "apply_keyword_weights")
        subgraph.add_edge("apply_keyword_weights", "rerank_documents")
        subgraph.add_edge("rerank_documents", END)

        return subgraph.compile()

    def merge_results_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """결과 병합 노드"""
        # 병합 로직은 processor의 _merge_and_rerank_results에서 처리됨
        return state

    def apply_keyword_weights_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 가중치 적용 노드"""
        # 키워드 가중치 적용 로직은 processor 내부에서 처리됨
        return state

    def rerank_documents_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 재순위 노드"""
        # 재순위 로직은 processor 내부에서 처리됨
        return state

