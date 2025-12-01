# -*- coding: utf-8 -*-
"""
Workflow Graph Builder
워크플로우 그래프 구축 로직을 처리하는 빌더
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.edges.classification_edges import ClassificationEdges
except ImportError:
    from core.workflow.edges.classification_edges import ClassificationEdges
try:
    from lawfirm_langgraph.core.workflow.edges.search_edges import SearchEdges
except ImportError:
    from core.workflow.edges.search_edges import SearchEdges
try:
    from lawfirm_langgraph.core.workflow.edges.answer_edges import AnswerEdges
except ImportError:
    from core.workflow.edges.answer_edges import AnswerEdges
try:
    from lawfirm_langgraph.core.workflow.edges.agentic_edges import AgenticEdges
except ImportError:
    from core.workflow.edges.agentic_edges import AgenticEdges


class WorkflowGraphBuilder:
    """워크플로우 그래프 빌더"""

    def __init__(
        self,
        config,
        logger,
        route_by_complexity_func=None,
        route_by_complexity_with_agentic_func=None,
        route_after_agentic_func=None,
        should_analyze_document_func=None,
        should_skip_search_adaptive_func=None,
        should_retry_validation_func=None,
        should_skip_final_node_func=None,
        classification_edges: Optional[ClassificationEdges] = None,
        search_edges: Optional[SearchEdges] = None,
        answer_edges: Optional[AnswerEdges] = None,
        agentic_edges: Optional[AgenticEdges] = None
    ):
        self.config = config
        self.logger = logger
        self._route_by_complexity_func = route_by_complexity_func
        self._route_by_complexity_with_agentic_func = route_by_complexity_with_agentic_func
        self._route_after_agentic_func = route_after_agentic_func
        self._should_analyze_document_func = should_analyze_document_func
        self._should_skip_search_adaptive_func = should_skip_search_adaptive_func
        self._should_retry_validation_func = should_retry_validation_func
        self._should_skip_final_node_func = should_skip_final_node_func
        
        # 엣지 빌더 초기화 (없으면 자동 생성)
        self.classification_edges = classification_edges or ClassificationEdges(
            route_by_complexity_func=route_by_complexity_func,
            route_by_complexity_with_agentic_func=route_by_complexity_with_agentic_func,
            should_analyze_document_func=should_analyze_document_func,
            logger_instance=logger
        )
        self.search_edges = search_edges or SearchEdges(
            should_skip_search_adaptive_func=should_skip_search_adaptive_func,
            logger_instance=logger
        )
        self.answer_edges = answer_edges or AnswerEdges(
            should_retry_validation_func=should_retry_validation_func,
            should_skip_final_node_func=should_skip_final_node_func,
            logger_instance=logger
        )
        self.agentic_edges = agentic_edges or AgenticEdges(
            route_after_agentic_func=route_after_agentic_func,
            logger_instance=logger
        )

    def build_graph(
        self,
        node_handlers: Dict[str, Callable]
    ) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)

        self.add_nodes(workflow, node_handlers)
        self.setup_entry_point(workflow)
        self.setup_routing(workflow)
        self.add_edges(workflow)

        return workflow

    def add_nodes(
        self,
        workflow: StateGraph,
        node_handlers: Dict[str, Callable]
    ) -> None:
        """노드 추가"""
        workflow.add_node("classify_query_and_complexity", node_handlers.get("classify_query_and_complexity"))
        
        # 윤리적 거부 노드 추가
        from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
        workflow.add_node("ethical_rejection", EthicalRejectionNode.ethical_rejection_node)
        self.logger.info("ethical_rejection node added to workflow")
        
        workflow.add_node("direct_answer", node_handlers.get("direct_answer_node"))
        workflow.add_node("classification_parallel", node_handlers.get("classification_parallel"))
        workflow.add_node("assess_urgency", node_handlers.get("assess_urgency"))
        workflow.add_node("resolve_multi_turn", node_handlers.get("resolve_multi_turn"))
        workflow.add_node("route_expert", node_handlers.get("route_expert"))
        workflow.add_node("analyze_document", node_handlers.get("analyze_document"))
        workflow.add_node("expand_keywords", node_handlers.get("expand_keywords"))
        workflow.add_node("prepare_search_query", node_handlers.get("prepare_search_query"))
        workflow.add_node("execute_searches_parallel", node_handlers.get("execute_searches_parallel"))
        workflow.add_node("process_search_results_combined", node_handlers.get("process_search_results_combined"))
        workflow.add_node("prepare_documents_and_terms", node_handlers.get("prepare_documents_and_terms"))
        workflow.add_node("generate_and_validate_answer", node_handlers.get("generate_and_validate_answer"))
        workflow.add_node("continue_answer_generation", node_handlers.get("continue_answer_generation"))
        
        # 스트리밍 노드 추가
        if node_handlers.get("generate_answer_stream"):
            workflow.add_node("generate_answer_stream", node_handlers.get("generate_answer_stream"))
            self.logger.info("generate_answer_stream node added to workflow")
        
        if node_handlers.get("generate_answer_final"):
            workflow.add_node("generate_answer_final", node_handlers.get("generate_answer_final"))
            self.logger.info("generate_answer_final node added to workflow")

        if self.config.use_agentic_mode:
            workflow.add_node("agentic_decision", node_handlers.get("agentic_decision_node"))
            self.logger.info("Agentic decision node added to workflow")

    def setup_entry_point(self, workflow: StateGraph) -> None:
        """Entry point 설정"""
        workflow.set_entry_point("classify_query_and_complexity")

    def setup_routing(self, workflow: StateGraph) -> None:
        """라우팅 설정 (엣지 빌더 사용)"""
        answer_node = self._get_answer_generation_node()
        
        # 분류 관련 라우팅 엣지 추가
        self.classification_edges.add_classification_edges(
            workflow,
            use_agentic_mode=self.config.use_agentic_mode
        )
        
        # 문서 분석 관련 엣지 추가
        self.classification_edges.add_document_analysis_edges(workflow)
        
        # 검색 관련 엣지 추가 (라우팅 포함)
        self.search_edges.add_search_edges(
            workflow,
            answer_generation_node=answer_node
        )
        
        # 답변 생성 관련 엣지 추가
        self.answer_edges.add_answer_generation_edges(
            workflow,
            answer_node=answer_node
        )
        
        # Agentic 모드 관련 엣지 추가
        if self.config.use_agentic_mode:
            self.agentic_edges.add_agentic_edges(
                workflow,
                answer_generation_node=answer_node
            )

    def _get_answer_generation_node(self) -> str:
        """
        환경 변수에 따라 답변 생성 노드 선택
        
        Returns:
            노드 이름 (generate_answer_stream, generate_answer_final, 또는 generate_and_validate_answer)
        """
        use_streaming_mode = os.getenv("USE_STREAMING_MODE", "true").lower() == "true"
        
        if use_streaming_mode:
            # API 모드: 스트리밍 노드 사용
            return "generate_answer_stream"
        else:
            # 테스트 모드: 최종 노드 사용 (검증 및 포맷팅 포함)
            return "generate_answer_final"
    
    def add_edges(self, workflow: StateGraph) -> None:
        """엣지 추가 (엣지 빌더 사용)"""
        # 윤리적 거부 노드는 END로 연결
        workflow.add_edge("ethical_rejection", END)
        
        # 분류 병렬 처리 후 route_expert로
        workflow.add_edge("classification_parallel", "route_expert")
        
        # 직접 답변 엣지 추가
        self.answer_edges.add_direct_answer_edge(workflow)
        
        # 나머지 엣지들은 setup_routing에서 처리됨
        # (검색 엣지, 답변 생성 엣지 등)

