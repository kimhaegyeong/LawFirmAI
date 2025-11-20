# -*- coding: utf-8 -*-
"""
Workflow Graph Builder
워크플로우 그래프 구축 로직을 처리하는 빌더
"""

import logging
import os
from typing import Any, Callable, Dict

from langgraph.graph import END, StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState


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
        should_skip_final_node_func=None
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
        """라우팅 설정"""
        if self.config.use_agentic_mode:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self._route_by_complexity_with_agentic_func,
                {
                    "simple": "direct_answer",
                    "moderate": "classification_parallel",
                    "complex": "agentic_decision",
                }
            )
        else:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self._route_by_complexity_func,
                {
                    "simple": "direct_answer",
                    "moderate": "classification_parallel",
                    "complex": "classification_parallel",
                }
            )

        # agentic_decision 라우팅은 setup_routing의 끝에서 처리

        workflow.add_conditional_edges(
            "route_expert",
            self._should_analyze_document_func,
            {
                "analyze": "analyze_document",
                "skip": "expand_keywords"
            }
        )

        workflow.add_conditional_edges(
            "prepare_search_query",
            self._should_skip_search_adaptive_func,
            {
                "skip": self._get_answer_generation_node(),
                "continue": "execute_searches_parallel"
            }
        )

        # 답변 생성 노드 라우팅 (환경 변수 기반)
        answer_node = self._get_answer_generation_node()
        if answer_node == "generate_answer_stream":
            # 성능 최적화: 조건부로 최종 노드 스킵 가능
            # 스트리밍 노드 -> 조건부 최종 노드 -> 검증
            workflow.add_conditional_edges(
                "generate_answer_stream",
                self._should_skip_final_node_func,
                {
                    "skip": END,  # 이미 검증/포맷팅 완료된 경우 스킵
                    "finalize": "generate_answer_final"  # 검증/포맷팅 필요한 경우
                }
            )
            workflow.add_conditional_edges(
                "generate_answer_final",
                self._should_retry_validation_func,
                {
                    "accept": END,
                    "retry_generate": "generate_answer_stream",
                    "retry_search": "expand_keywords"
                }
            )
        elif answer_node == "generate_answer_final":
            # 최종 노드만 사용 (테스트 모드)
            workflow.add_conditional_edges(
                "generate_answer_final",
                self._should_retry_validation_func,
                {
                    "accept": END,
                    "retry_generate": "generate_answer_final",
                    "retry_search": "expand_keywords"
                }
            )
        else:
            # 기본 노드 사용 (generate_and_validate_answer)
            workflow.add_conditional_edges(
                "generate_and_validate_answer",
                self._should_retry_validation_func,
                {
                    "accept": END,
                    "retry_generate": "generate_and_validate_answer",
                    "retry_search": "expand_keywords"
                }
            )
        
        # agentic_decision에서도 답변 생성 노드 사용
        if self.config.use_agentic_mode:
            workflow.add_conditional_edges(
                "agentic_decision",
                self._route_after_agentic_func,
                {
                    "has_results": "prepare_documents_and_terms",
                    "no_results": self._get_answer_generation_node(),
                }
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
        """엣지 추가"""
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("classification_parallel", "route_expert")
        workflow.add_edge("analyze_document", "expand_keywords")
        workflow.add_edge("expand_keywords", "prepare_search_query")
        workflow.add_edge("execute_searches_parallel", "process_search_results_combined")
        workflow.add_edge("process_search_results_combined", "prepare_documents_and_terms")
        
        # 답변 생성 노드로의 엣지 (환경 변수 기반)
        answer_node = self._get_answer_generation_node()
        workflow.add_edge("prepare_documents_and_terms", answer_node)

