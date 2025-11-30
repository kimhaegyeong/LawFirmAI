# -*- coding: utf-8 -*-
"""
Search Edges
검색 관련 엣지 정의
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

from langgraph.graph import StateGraph

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class SearchEdges:
    """검색 관련 엣지 빌더"""
    
    def __init__(
        self,
        should_skip_search_adaptive_func=None,
        should_use_multi_query_agent_func=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        SearchEdges 초기화
        
        Args:
            should_skip_search_adaptive_func: 검색 스킵 여부 결정 함수
            should_use_multi_query_agent_func: 멀티 질의 에이전트 사용 여부 결정 함수
            logger_instance: 로거 인스턴스
        """
        self.should_skip_search_adaptive_func = should_skip_search_adaptive_func
        self.should_use_multi_query_agent_func = should_use_multi_query_agent_func
        self.logger = logger_instance or logger
    
    def add_search_edges(
        self,
        workflow: StateGraph,
        answer_generation_node: str = "generate_and_validate_answer"
    ) -> None:
        """
        검색 관련 엣지 추가
        
        Args:
            workflow: StateGraph 인스턴스
            answer_generation_node: 답변 생성 노드 이름
        """
        # 문서 분석 후 검색으로
        workflow.add_edge("analyze_document", "expand_keywords")
        
        # 🔥 개선: expand_keywords → classify_complexity_after_keywords → multi_query_search_agent
        # (classification_edges.py에서 이미 처리하므로 여기서는 제거)
        # 키워드 확장 후 복잡도 재평가가 먼저 실행되고, 그 후 멀티 질의 에이전트로 이동
        # 따라서 expand_keywords에서 직접 연결하는 엣지는 제거됨
        
        # 멀티 질의 에이전트 실행 후 결과 처리 노드로 연결 (병합 및 중복 제거를 위해)
        # 노드가 존재하는지 확인 (노드 이름 리스트로 확인)
        workflow_nodes = list(workflow.nodes.keys()) if hasattr(workflow, 'nodes') else []
        if "multi_query_search_agent" in workflow_nodes:
            # 🔥 multi-query 결과도 process_search_results_combined에서 처리하도록 연결
            if "process_search_results_combined" in workflow_nodes:
                workflow.add_edge("multi_query_search_agent", "process_search_results_combined")
            else:
                # process_search_results_combined가 없으면 prepare_documents_and_terms로 직접 연결
                workflow.add_edge("multi_query_search_agent", "prepare_documents_and_terms")
        
        # 검색 쿼리 준비 후 조건부 검색 실행
        if self.should_skip_search_adaptive_func:
            workflow.add_conditional_edges(
                "prepare_search_query",
                self.should_skip_search_adaptive_func,
                {
                    "skip": answer_generation_node,
                    "continue": "execute_searches_parallel"
                }
            )
        else:
            workflow.add_edge("prepare_search_query", "execute_searches_parallel")
        
        # 검색 실행 후 결과 처리
        workflow.add_edge("execute_searches_parallel", "process_search_results_combined")
        
        # 검색 결과 처리 후 문서 준비
        workflow.add_edge("process_search_results_combined", "prepare_documents_and_terms")

