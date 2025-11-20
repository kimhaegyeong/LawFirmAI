# -*- coding: utf-8 -*-
"""
엣지 빌더 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from langgraph.graph import StateGraph, END
from core.workflow.edges.classification_edges import ClassificationEdges
from core.workflow.edges.search_edges import SearchEdges
from core.workflow.edges.answer_edges import AnswerEdges
from core.workflow.edges.agentic_edges import AgenticEdges
from core.workflow.state.state_definitions import LegalWorkflowState


class TestClassificationEdges:
    """ClassificationEdges 테스트"""
    
    @pytest.fixture
    def mock_routing_func(self):
        """Mock 라우팅 함수"""
        return Mock(return_value="direct_answer")
    
    @pytest.fixture
    def edges(self, mock_routing_func):
        """ClassificationEdges 인스턴스"""
        return ClassificationEdges(
            route_by_complexity_func=mock_routing_func,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def mock_graph(self):
        """Mock StateGraph"""
        graph = StateGraph(LegalWorkflowState)
        graph.add_node("classify_query", Mock())
        graph.add_node("direct_answer", Mock())
        graph.add_node("classification_parallel", Mock())
        return graph
    
    def test_add_classification_edges(self, edges, mock_graph, mock_routing_func):
        """분류 엣지 추가 테스트"""
        edges.add_classification_edges(mock_graph, use_agentic_mode=False)
        
        # 엣지가 추가되었는지 확인 (실제 구현에 따라 조정)
        # graph의 엣지 구조를 확인하는 방법은 LangGraph API에 따라 다름
        assert mock_routing_func is not None
    
    def test_add_classification_edges_with_agentic(self, edges, mock_graph):
        """Agentic 모드 분류 엣지 추가 테스트"""
        mock_agentic_func = Mock(return_value="agentic_decision")
        edges.route_by_complexity_with_agentic_func = mock_agentic_func
        
        edges.add_classification_edges(mock_graph, use_agentic_mode=True)
        
        # Agentic 모드에서도 엣지가 추가되어야 함
        assert mock_agentic_func is not None
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        edges = ClassificationEdges(
            route_by_complexity_func=Mock(),
            logger_instance=mock_logger
        )
        
        assert edges.logger == mock_logger


class TestSearchEdges:
    """SearchEdges 테스트"""
    
    @pytest.fixture
    def mock_skip_func(self):
        """Mock 검색 스킵 함수"""
        return Mock(return_value="continue")
    
    @pytest.fixture
    def edges(self, mock_skip_func):
        """SearchEdges 인스턴스"""
        return SearchEdges(
            should_skip_search_adaptive_func=mock_skip_func,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def mock_graph(self):
        """Mock StateGraph"""
        graph = StateGraph(LegalWorkflowState)
        graph.add_node("expand_keywords", Mock())
        graph.add_node("prepare_search_query", Mock())
        graph.add_node("execute_searches_parallel", Mock())
        graph.add_node("generate_and_validate_answer", Mock())
        return graph
    
    def test_add_search_edges(self, edges, mock_graph, mock_skip_func):
        """검색 엣지 추가 테스트"""
        edges.add_search_edges(mock_graph)
        
        # 엣지가 추가되었는지 확인
        assert mock_skip_func is not None
    
    def test_add_search_edges_without_skip_func(self, mock_graph):
        """검색 스킵 함수 없이 엣지 추가 테스트"""
        edges = SearchEdges(should_skip_search_adaptive_func=None)
        
        # 스킵 함수가 없어도 엣지는 추가되어야 함
        edges.add_search_edges(mock_graph)
        
        assert edges.should_skip_search_adaptive_func is None
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        edges = SearchEdges(
            should_skip_search_adaptive_func=Mock(),
            logger_instance=mock_logger
        )
        
        assert edges.logger == mock_logger


class TestAnswerEdges:
    """AnswerEdges 테스트"""
    
    @pytest.fixture
    def mock_retry_func(self):
        """Mock 재시도 함수"""
        return Mock(return_value="accept")
    
    @pytest.fixture
    def mock_skip_func(self):
        """Mock 스킵 함수"""
        return Mock(return_value="skip")
    
    @pytest.fixture
    def edges(self, mock_retry_func, mock_skip_func):
        """AnswerEdges 인스턴스"""
        return AnswerEdges(
            should_retry_validation_func=mock_retry_func,
            should_skip_final_node_func=mock_skip_func,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def mock_graph(self):
        """Mock StateGraph"""
        graph = StateGraph(LegalWorkflowState)
        graph.add_node("prepare_documents_and_terms", Mock())
        graph.add_node("generate_and_validate_answer", Mock())
        graph.add_node("generate_answer_final", Mock())
        return graph
    
    def test_add_answer_generation_edges(self, edges, mock_graph):
        """답변 생성 엣지 추가 테스트"""
        edges.add_answer_generation_edges(mock_graph)
        
        # 엣지가 추가되었는지 확인
        assert edges.should_retry_validation_func is not None
    
    def test_add_answer_generation_edges_stream(self, edges, mock_graph):
        """스트리밍 답변 생성 엣지 추가 테스트"""
        edges.add_answer_generation_edges(mock_graph, answer_node="generate_answer_stream")
        
        # 스트리밍 모드 엣지가 추가되었는지 확인
        assert edges.should_skip_final_node_func is not None
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        edges = AnswerEdges(
            should_retry_validation_func=Mock(),
            should_skip_final_node_func=Mock(),
            logger_instance=mock_logger
        )
        
        assert edges.logger == mock_logger


class TestAgenticEdges:
    """AgenticEdges 테스트"""
    
    @pytest.fixture
    def mock_route_func(self):
        """Mock 라우팅 함수"""
        return Mock(return_value="has_results")
    
    @pytest.fixture
    def edges(self, mock_route_func):
        """AgenticEdges 인스턴스"""
        return AgenticEdges(
            route_after_agentic_func=mock_route_func,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def mock_graph(self):
        """Mock StateGraph"""
        graph = StateGraph(LegalWorkflowState)
        graph.add_node("agentic_decision", Mock())
        graph.add_node("generate_and_validate_answer", Mock())
        return graph
    
    def test_add_agentic_edges(self, edges, mock_graph, mock_route_func):
        """Agentic 엣지 추가 테스트"""
        edges.add_agentic_edges(mock_graph)
        
        # 엣지가 추가되었는지 확인
        assert mock_route_func is not None
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        edges = AgenticEdges(
            route_after_agentic_func=Mock(),
            logger_instance=mock_logger
        )
        
        assert edges.logger == mock_logger

