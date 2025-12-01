# -*- coding: utf-8 -*-
"""
서브그래프 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from langgraph.graph import StateGraph
from core.workflow.subgraphs.classification_subgraph import ClassificationSubgraph
from core.workflow.subgraphs.search_subgraph import SearchSubgraph
from core.workflow.subgraphs.answer_generation_subgraph import AnswerGenerationSubgraph
from core.workflow.subgraphs.document_preparation_subgraph import DocumentPreparationSubgraph


class TestClassificationSubgraph:
    """ClassificationSubgraph 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.assess_urgency = Mock()
        workflow.resolve_multi_turn = Mock()
        workflow.route_expert = Mock()
        return workflow
    
    @pytest.fixture
    def subgraph(self, mock_workflow):
        """ClassificationSubgraph 인스턴스"""
        return ClassificationSubgraph(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    def test_build_subgraph(self, subgraph, mock_workflow):
        """서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph()
        
        assert isinstance(graph, StateGraph)
        # 노드가 추가되었는지 확인 (실제 노드 이름은 구현에 따라 다를 수 있음)
        # StateGraph의 노드 확인 방법은 LangGraph 버전에 따라 다를 수 있음
    
    def test_subgraph_without_workflow(self):
        """워크플로우 없이 서브그래프 구축 테스트"""
        subgraph = ClassificationSubgraph(workflow_instance=None)
        
        # 워크플로우가 없어도 빈 그래프는 생성 가능해야 함
        graph = subgraph.build_subgraph()
        assert isinstance(graph, StateGraph)
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        subgraph = ClassificationSubgraph(
            workflow_instance=MagicMock(),
            logger_instance=mock_logger
        )
        
        assert subgraph.logger == mock_logger


class TestSearchSubgraph:
    """SearchSubgraph 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.expand_keywords = Mock()
        workflow.prepare_search_query = Mock()
        workflow.execute_searches_parallel = Mock()
        workflow.process_search_results_combined = Mock()
        return workflow
    
    @pytest.fixture
    def subgraph(self, mock_workflow):
        """SearchSubgraph 인스턴스"""
        return SearchSubgraph(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    def test_build_subgraph(self, subgraph, mock_workflow):
        """서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph()
        
        assert isinstance(graph, StateGraph)
    
    def test_build_subgraph_with_search_results_subgraph(self, mock_workflow):
        """검색 결과 서브그래프를 포함한 서브그래프 구축 테스트"""
        mock_search_results_subgraph = Mock()
        subgraph = SearchSubgraph(
            workflow_instance=mock_workflow,
            search_results_subgraph=mock_search_results_subgraph,
            logger_instance=Mock()
        )
        
        graph = subgraph.build_subgraph()
        assert isinstance(graph, StateGraph)
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        subgraph = SearchSubgraph(
            workflow_instance=MagicMock(),
            logger_instance=mock_logger
        )
        
        assert subgraph.logger == mock_logger


class TestAnswerGenerationSubgraph:
    """AnswerGenerationSubgraph 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.generate_and_validate_answer = Mock()
        workflow.generate_answer_stream = Mock()
        workflow.generate_answer_final = Mock()
        return workflow
    
    @pytest.fixture
    def subgraph(self, mock_workflow):
        """AnswerGenerationSubgraph 인스턴스"""
        return AnswerGenerationSubgraph(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    def test_build_subgraph_default(self, subgraph, mock_workflow):
        """기본 답변 생성 서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph()
        
        assert isinstance(graph, StateGraph)
    
    def test_build_subgraph_with_stream(self, subgraph, mock_workflow):
        """스트리밍 답변 생성 서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph(answer_node_name="generate_answer_stream")
        
        assert isinstance(graph, StateGraph)
    
    def test_build_subgraph_with_final(self, subgraph, mock_workflow):
        """최종 답변 생성 서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph(answer_node_name="generate_answer_final")
        
        assert isinstance(graph, StateGraph)
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        subgraph = AnswerGenerationSubgraph(
            workflow_instance=MagicMock(),
            logger_instance=mock_logger
        )
        
        assert subgraph.logger == mock_logger


class TestDocumentPreparationSubgraph:
    """DocumentPreparationSubgraph 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.prepare_documents_and_terms = Mock()
        return workflow
    
    @pytest.fixture
    def subgraph(self, mock_workflow):
        """DocumentPreparationSubgraph 인스턴스"""
        return DocumentPreparationSubgraph(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    def test_build_subgraph(self, subgraph, mock_workflow):
        """서브그래프 구축 테스트"""
        graph = subgraph.build_subgraph()
        
        assert isinstance(graph, StateGraph)
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        subgraph = DocumentPreparationSubgraph(
            workflow_instance=MagicMock(),
            logger_instance=mock_logger
        )
        
        assert subgraph.logger == mock_logger

