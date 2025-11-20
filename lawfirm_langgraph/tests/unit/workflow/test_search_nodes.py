# -*- coding: utf-8 -*-
"""
SearchNodes 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.nodes.search_nodes import SearchNodes
from core.workflow.state.state_definitions import LegalWorkflowState


class TestSearchNodes:
    """SearchNodes 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.expand_keywords = Mock(return_value={"extracted_keywords": ["test"]})
        workflow.prepare_search_query = Mock(return_value={"search_params": {}})
        workflow.execute_searches_parallel = Mock(return_value={"semantic_results": []})
        workflow.process_search_results_combined = Mock(return_value={"retrieved_docs": []})
        return workflow
    
    @pytest.fixture
    def search_nodes(self, mock_workflow):
        """SearchNodes 인스턴스"""
        return SearchNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "계약 해지 사유는?", "session_id": "test"},
            "query": "계약 해지 사유는?",
            "extracted_keywords": [],
        }
    
    def test_expand_keywords(self, search_nodes, sample_state, mock_workflow):
        """키워드 확장 테스트"""
        result = search_nodes.expand_keywords(sample_state)
        
        mock_workflow.expand_keywords.assert_called_once_with(sample_state)
        assert result == {"extracted_keywords": ["test"]}
    
    def test_prepare_search_query(self, search_nodes, sample_state, mock_workflow):
        """검색 쿼리 준비 테스트"""
        result = search_nodes.prepare_search_query(sample_state)
        
        mock_workflow.prepare_search_query.assert_called_once_with(sample_state)
        assert result == {"search_params": {}}
    
    def test_execute_searches_parallel(self, search_nodes, sample_state, mock_workflow):
        """병렬 검색 실행 테스트"""
        result = search_nodes.execute_searches_parallel(sample_state)
        
        mock_workflow.execute_searches_parallel.assert_called_once_with(sample_state)
        assert result == {"semantic_results": []}
    
    def test_process_search_results_combined(self, search_nodes, sample_state, mock_workflow):
        """검색 결과 처리 통합 테스트"""
        result = search_nodes.process_search_results_combined(sample_state)
        
        mock_workflow.process_search_results_combined.assert_called_once_with(sample_state)
        assert result == {"retrieved_docs": []}
    
    def test_workflow_instance_required(self):
        """workflow_instance 필수 검증 테스트"""
        nodes = SearchNodes(workflow_instance=None)
        
        with pytest.raises(RuntimeError, match="workflow_instance가 설정되지 않았습니다"):
            nodes.expand_keywords({})
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        nodes = SearchNodes(workflow_instance=MagicMock(), logger_instance=mock_logger)
        
        assert nodes.logger == mock_logger

