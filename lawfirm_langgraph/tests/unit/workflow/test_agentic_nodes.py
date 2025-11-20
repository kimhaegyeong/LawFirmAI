# -*- coding: utf-8 -*-
"""
AgenticNodes 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.nodes.agentic_nodes import AgenticNodes
from core.workflow.state.state_definitions import LegalWorkflowState


class TestAgenticNodes:
    """AgenticNodes 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.agentic_decision_node = Mock(return_value={"agentic_decision": "test"})
        return workflow
    
    @pytest.fixture
    def agentic_nodes(self, mock_workflow):
        """AgenticNodes 인스턴스"""
        return AgenticNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "복잡한 법률 질문", "session_id": "test"},
            "query": "복잡한 법률 질문",
            "query_complexity": "complex",
        }
    
    def test_agentic_decision_node(self, agentic_nodes, sample_state, mock_workflow):
        """Agentic 결정 노드 테스트"""
        result = agentic_nodes.agentic_decision_node(sample_state)
        
        mock_workflow.agentic_decision_node.assert_called_once_with(sample_state)
        assert result == {"agentic_decision": "test"}
    
    def test_workflow_instance_required(self):
        """workflow_instance 필수 검증 테스트"""
        nodes = AgenticNodes(workflow_instance=None)
        
        with pytest.raises(RuntimeError, match="workflow_instance가 설정되지 않았습니다"):
            nodes.agentic_decision_node({})
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        nodes = AgenticNodes(workflow_instance=MagicMock(), logger_instance=mock_logger)
        
        assert nodes.logger == mock_logger

