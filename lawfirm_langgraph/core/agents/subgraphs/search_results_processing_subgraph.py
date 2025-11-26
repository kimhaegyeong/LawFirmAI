# -*- coding: utf-8 -*-
"""
검색 결과 처리 서브그래프
LangGraph의 Subgraph 패턴을 활용한 검색 결과 처리 워크플로우
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

try:
    from lawfirm_langgraph.core.agents.state_definitions import LegalWorkflowState
except ImportError:
    from core.agents.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.agents.handlers.search_result_processor import SearchResultProcessor
except ImportError:
    from core.agents.handlers.search_result_processor import SearchResultProcessor


class SearchResultsProcessingSubgraph:
    """
    검색 결과 처리 서브그래프
    
    검색 결과의 품질 평가, 병합, 재순위, 필터링을 처리하는 서브그래프
    """

    def __init__(
        self,
        search_result_processor: SearchResultProcessor,
        logger: Optional[logging.Logger] = None
    ):
        """
        SearchResultsProcessingSubgraph 초기화
        
        Args:
            search_result_processor: 검색 결과 처리 핸들러
            logger: 로거 (없으면 자동 생성)
        """
        self.processor = search_result_processor
        self.logger = logger or logging.getLogger(__name__)

    def build_subgraph(self) -> StateGraph:
        """
        검색 결과 처리 서브그래프 구축
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)

        # 노드 추가
        subgraph.add_node("evaluate_quality", self.evaluate_quality_node)
        subgraph.add_node("conditional_retry", self.conditional_retry_node)
        subgraph.add_node("merge_and_rerank", self.merge_and_rerank_node)
        subgraph.add_node("filter_validate", self.filter_validate_node)
        subgraph.add_node("update_metadata", self.update_metadata_node)

        # 엣지 설정
        subgraph.set_entry_point("evaluate_quality")
        
        # 조건부 엣지: 품질 평가 후 재검색 여부 결정
        subgraph.add_conditional_edges(
            "evaluate_quality",
            self._should_retry,
            {
                "retry": "conditional_retry",
                "continue": "merge_and_rerank"
            }
        )
        
        subgraph.add_edge("conditional_retry", "merge_and_rerank")
        subgraph.add_edge("merge_and_rerank", "filter_validate")
        subgraph.add_edge("filter_validate", "update_metadata")
        subgraph.add_edge("update_metadata", END)

        return subgraph.compile()

    def evaluate_quality_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """품질 평가 노드"""
        # 품질 평가만 수행 (processor의 내부 메서드 활용)
        from core.workflow.utils.workflow_utils import WorkflowUtils
        import asyncio
        from core.agents.tasks.search_result_tasks import SearchResultTasks
        
        state_values = WorkflowUtils.get_state_values_batch(
            state,
            keys=["semantic_results", "keyword_results", "query", "query_type", "search_params"],
            defaults={
                "semantic_results": [],
                "keyword_results": [],
                "query": "",
                "query_type": "",
                "search_params": {}
            }
        )
        
        quality_evaluation = asyncio.run(SearchResultTasks.evaluate_quality_parallel(
            semantic_results=state_values["semantic_results"],
            keyword_results=state_values["keyword_results"],
            query=state_values["query"],
            query_type=state_values["query_type"],
            search_params=state_values["search_params"],
            evaluate_semantic_func=self.processor.evaluate_semantic_quality,
            evaluate_keyword_func=self.processor.evaluate_keyword_quality
        ))
        
        WorkflowUtils.set_state_value(state, "search_quality_evaluation", quality_evaluation)
        return state

    def conditional_retry_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """조건부 재검색 노드"""
        # 재검색 로직은 processor의 _perform_conditional_retry에서 처리
        # 현재는 processor.process_search_results_combined에서 통합 처리
        return state

    def merge_and_rerank_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """병합 및 재순위 노드"""
        # 병합 및 재순위는 processor.process_search_results_combined에서 처리
        # Subgraph를 사용할 때는 전체 프로세스를 한 번에 실행
        return self.processor.process_search_results_combined(state)

    def filter_validate_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """필터링 및 검증 노드"""
        # 필터링 및 검증은 processor.process_search_results_combined에서 처리
        return state

    def update_metadata_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """메타데이터 업데이트 노드"""
        # 메타데이터 업데이트는 processor.process_search_results_combined에서 처리
        return state

    def _should_retry(self, state: LegalWorkflowState) -> str:
        """
        재검색 여부 결정
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            "retry" 또는 "continue"
        """
        from core.workflow.utils.workflow_utils import WorkflowUtils
        
        quality_evaluation = WorkflowUtils.get_state_value(state, "search_quality_evaluation", {})
        
        if not quality_evaluation:
            return "continue"
        
        needs_retry = quality_evaluation.get("needs_retry", False)
        overall_quality = quality_evaluation.get("overall_quality", 1.0)
        
        semantic_count = WorkflowUtils.get_state_value(state, "semantic_count", 0)
        keyword_count = WorkflowUtils.get_state_value(state, "keyword_count", 0)
        total_count = semantic_count + keyword_count
        
        if needs_retry and overall_quality < 0.6 and total_count < 10:
            return "retry"
        
        return "continue"

