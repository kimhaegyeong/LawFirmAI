# -*- coding: utf-8 -*-
"""
Search Subgraph
검색 관련 노드들을 서브그래프로 구성
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState


logger = logging.getLogger(__name__)


class SearchSubgraph:
    """검색 서브그래프"""
    
    def __init__(
        self,
        workflow_instance=None,
        search_results_subgraph=None,
        should_skip_search_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        SearchSubgraph 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스
            search_results_subgraph: 검색 결과 처리 서브그래프 (선택적)
            should_skip_search_func: 검색 스킵 여부 결정 함수
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.search_results_subgraph = search_results_subgraph
        self.should_skip_search_func = should_skip_search_func
        self.logger = logger_instance or logger
    
    def build_subgraph(self) -> StateGraph:
        """
        검색 서브그래프 구축
        
        Returns:
            컴파일된 서브그래프
        """
        subgraph = StateGraph(LegalWorkflowState)
        
        # 노드 추가
        if self.workflow:
            subgraph.add_node("expand_keywords", self.workflow.expand_keywords)
            subgraph.add_node("prepare_search_query", self.workflow.prepare_search_query)
            subgraph.add_node("execute_searches_parallel", self.workflow.execute_searches_parallel)
            
            # 검색 결과 처리 노드 (서브그래프 사용 가능)
            if self.search_results_subgraph:
                subgraph.add_node("process_search_results", self.search_results_subgraph)
            else:
                subgraph.add_node("process_search_results_combined", self.workflow.process_search_results_combined)
        else:
            raise RuntimeError("workflow_instance가 설정되지 않았습니다")
        
        # 엣지 설정
        subgraph.set_entry_point("expand_keywords")
        subgraph.add_edge("expand_keywords", "prepare_search_query")
        
        # 조건부 엣지: 검색 스킵 여부 결정
        if self.should_skip_search_func:
            subgraph.add_conditional_edges(
                "prepare_search_query",
                self.should_skip_search_func,
                {
                    "skip": END,
                    "continue": "execute_searches_parallel"
                }
            )
        else:
            subgraph.add_edge("prepare_search_query", "execute_searches_parallel")
        
        # 검색 결과 처리 노드로 연결
        if self.search_results_subgraph:
            subgraph.add_edge("execute_searches_parallel", "process_search_results")
            subgraph.add_edge("process_search_results", END)
        else:
            subgraph.add_edge("execute_searches_parallel", "process_search_results_combined")
            subgraph.add_edge("process_search_results_combined", END)
        
        return subgraph.compile()

