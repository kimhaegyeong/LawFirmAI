# -*- coding: utf-8 -*-
"""
Modular Graph Builder
레지스트리 기반 모듈화된 그래프 빌더
"""

import logging
import os
from typing import Optional

from langgraph.graph import END, StateGraph

from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.registry.node_registry import NodeRegistry
from core.workflow.registry.subgraph_registry import SubgraphRegistry
from core.workflow.edges.classification_edges import ClassificationEdges
from core.workflow.edges.search_edges import SearchEdges
from core.workflow.edges.answer_edges import AnswerEdges
from core.workflow.edges.agentic_edges import AgenticEdges


logger = logging.getLogger(__name__)


class ModularGraphBuilder:
    """모듈화된 그래프 빌더"""
    
    def __init__(
        self,
        config,
        logger_instance: Optional[logging.Logger] = None,
        node_registry: Optional[NodeRegistry] = None,
        subgraph_registry: Optional[SubgraphRegistry] = None,
        classification_edges: Optional[ClassificationEdges] = None,
        search_edges: Optional[SearchEdges] = None,
        answer_edges: Optional[AnswerEdges] = None,
        agentic_edges: Optional[AgenticEdges] = None
    ):
        """
        ModularGraphBuilder 초기화
        
        Args:
            config: 설정 객체
            logger_instance: 로거 인스턴스
            node_registry: 노드 레지스트리
            subgraph_registry: 서브그래프 레지스트리
            classification_edges: 분류 엣지 빌더
            search_edges: 검색 엣지 빌더
            answer_edges: 답변 엣지 빌더
            agentic_edges: Agentic 엣지 빌더
        """
        self.config = config
        self.logger = logger_instance or logger
        self.node_registry = node_registry or NodeRegistry(logger_instance=self.logger)
        self.subgraph_registry = subgraph_registry or SubgraphRegistry(logger_instance=self.logger)
        self.classification_edges = classification_edges
        self.search_edges = search_edges
        self.answer_edges = answer_edges
        self.agentic_edges = agentic_edges
    
    def build_graph(self) -> StateGraph:
        """
        모듈화된 그래프 구축
        
        Returns:
            StateGraph 인스턴스
        """
        workflow = StateGraph(LegalWorkflowState)
        
        # 1. 노드 등록
        self._register_nodes(workflow)
        
        # 2. 서브그래프 등록
        self._register_subgraphs(workflow)
        
        # 3. 엔트리 포인트 설정
        workflow.set_entry_point("classify_query_and_complexity")
        
        # 4. 엣지 추가
        self._add_edges(workflow)
        
        return workflow
    
    def _register_nodes(self, workflow: StateGraph) -> None:
        """노드 등록"""
        nodes = self.node_registry.get_all_nodes()
        for node_name, node_func in nodes.items():
            if node_func:
                workflow.add_node(node_name, node_func)
                self.logger.info(f"노드 등록됨: {node_name}")
            else:
                self.logger.warning(f"노드 함수가 None입니다: {node_name}")
    
    def _register_subgraphs(self, workflow: StateGraph) -> None:
        """서브그래프 등록"""
        subgraphs = self.subgraph_registry.get_all_subgraphs()
        for subgraph_name, subgraph in subgraphs.items():
            workflow.add_node(subgraph_name, subgraph)
            self.logger.info(f"서브그래프 등록됨: {subgraph_name}")
    
    def _add_edges(self, workflow: StateGraph) -> None:
        """엣지 추가"""
        # 분류 엣지
        if self.classification_edges:
            self.classification_edges.add_classification_edges(
                workflow,
                use_agentic_mode=self.config.use_agentic_mode if hasattr(self.config, 'use_agentic_mode') else False
            )
            self.classification_edges.add_document_analysis_edges(workflow)
        
        # 검색 엣지
        if self.search_edges:
            answer_node = self._get_answer_generation_node()
            self.search_edges.add_search_edges(workflow, answer_node)
        
        # 답변 생성 엣지
        if self.answer_edges:
            answer_node = self._get_answer_generation_node()
            self.answer_edges.add_answer_generation_edges(workflow, answer_node)
            self.answer_edges.add_direct_answer_edge(workflow)
        
        # Agentic 엣지
        if self.agentic_edges and hasattr(self.config, 'use_agentic_mode') and self.config.use_agentic_mode:
            answer_node = self._get_answer_generation_node()
            self.agentic_edges.add_agentic_edges(workflow, answer_node)
        
        # 일반 엣지
        workflow.add_edge("classification_parallel", "route_expert")
    
    def _get_answer_generation_node(self) -> str:
        """
        환경 변수에 따라 답변 생성 노드 선택
        
        Returns:
            노드 이름
        """
        use_streaming_mode = os.getenv("USE_STREAMING_MODE", "true").lower() == "true"
        
        if use_streaming_mode:
            return "generate_answer_stream"
        else:
            return "generate_answer_final"

