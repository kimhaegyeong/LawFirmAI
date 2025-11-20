# -*- coding: utf-8 -*-
"""
ClassificationNodes 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.nodes.classification_nodes import ClassificationNodes
from core.workflow.state.state_definitions import LegalWorkflowState


class TestClassificationNodes:
    """ClassificationNodes 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.classify_query_and_complexity = Mock(return_value={"query": "test"})
        workflow.classification_parallel = Mock(return_value={"query": "test"})
        workflow.assess_urgency = Mock(return_value={"query": "test"})
        workflow.resolve_multi_turn = Mock(return_value={"query": "test"})
        workflow.route_expert = Mock(return_value={"query": "test"})
        workflow.direct_answer_node = Mock(return_value={"answer": "test"})
        return workflow
    
    @pytest.fixture
    def classification_nodes(self, mock_workflow):
        """ClassificationNodes 인스턴스"""
        return ClassificationNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "계약 해지 사유는?", "session_id": "test"},
            "query": "계약 해지 사유는?",
            "query_complexity": "moderate",
        }
    
    def test_classify_query_and_complexity(self, classification_nodes, sample_state, mock_workflow):
        """질문 분류 및 복잡도 판단 테스트"""
        result = classification_nodes.classify_query_and_complexity(sample_state)
        
        mock_workflow.classify_query_and_complexity.assert_called_once_with(sample_state)
        assert result == {"query": "test"}
    
    def test_classification_parallel(self, classification_nodes, sample_state, mock_workflow):
        """병렬 분류 테스트"""
        result = classification_nodes.classification_parallel(sample_state)
        
        mock_workflow.classification_parallel.assert_called_once_with(sample_state)
        assert result == {"query": "test"}
    
    def test_assess_urgency(self, classification_nodes, sample_state, mock_workflow):
        """긴급도 평가 테스트"""
        result = classification_nodes.assess_urgency(sample_state)
        
        mock_workflow.assess_urgency.assert_called_once_with(sample_state)
        assert result == {"query": "test"}
    
    def test_resolve_multi_turn(self, classification_nodes, sample_state, mock_workflow):
        """멀티턴 처리 테스트"""
        result = classification_nodes.resolve_multi_turn(sample_state)
        
        mock_workflow.resolve_multi_turn.assert_called_once_with(sample_state)
        assert result == {"query": "test"}
    
    def test_route_expert(self, classification_nodes, sample_state, mock_workflow):
        """전문가 라우팅 테스트"""
        result = classification_nodes.route_expert(sample_state)
        
        mock_workflow.route_expert.assert_called_once_with(sample_state)
        assert result == {"query": "test"}
    
    def test_direct_answer(self, classification_nodes, sample_state, mock_workflow):
        """직접 답변 테스트"""
        result = classification_nodes.direct_answer(sample_state)
        
        mock_workflow.direct_answer_node.assert_called_once_with(sample_state)
        assert result == {"answer": "test"}
    
    def test_workflow_instance_required(self):
        """workflow_instance 필수 검증 테스트"""
        nodes = ClassificationNodes(workflow_instance=None)
        
        with pytest.raises(RuntimeError, match="workflow_instance가 설정되지 않았습니다"):
            nodes.classify_query_and_complexity({})
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        nodes = ClassificationNodes(workflow_instance=MagicMock(), logger_instance=mock_logger)
        
        assert nodes.logger == mock_logger
    
    def test_initialization_without_logger(self):
        """로거 없이 초기화 테스트"""
        nodes = ClassificationNodes(workflow_instance=MagicMock())
        
        assert nodes.logger is not None

