# -*- coding: utf-8 -*-
"""
AnswerNodes 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.nodes.answer_nodes import AnswerNodes
from core.workflow.state.state_definitions import LegalWorkflowState


class TestAnswerNodes:
    """AnswerNodes 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.generate_and_validate_answer = Mock(return_value={"answer": "test answer"})
        workflow.generate_answer_stream = Mock(return_value={"answer": "streaming answer"})
        workflow.generate_answer_final = Mock(return_value={"answer": "final answer"})
        workflow.continue_answer_generation = Mock(return_value={"answer": "continued answer"})
        return workflow
    
    @pytest.fixture
    def answer_nodes(self, mock_workflow):
        """AnswerNodes 인스턴스"""
        return AnswerNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "계약 해지 사유는?", "session_id": "test"},
            "query": "계약 해지 사유는?",
            "retrieved_docs": [],
            "answer": "",
        }
    
    def test_generate_and_validate_answer(self, answer_nodes, sample_state, mock_workflow):
        """답변 생성 및 검증 테스트"""
        result = answer_nodes.generate_and_validate_answer(sample_state)
        
        mock_workflow.generate_and_validate_answer.assert_called_once_with(sample_state)
        assert result == {"answer": "test answer"}
    
    def test_generate_answer_stream(self, answer_nodes, sample_state, mock_workflow):
        """스트리밍 답변 생성 테스트"""
        result = answer_nodes.generate_answer_stream(sample_state)
        
        mock_workflow.generate_answer_stream.assert_called_once_with(sample_state)
        assert result == {"answer": "streaming answer"}
    
    def test_generate_answer_final(self, answer_nodes, sample_state, mock_workflow):
        """최종 답변 생성 테스트"""
        result = answer_nodes.generate_answer_final(sample_state)
        
        mock_workflow.generate_answer_final.assert_called_once_with(sample_state)
        assert result == {"answer": "final answer"}
    
    def test_continue_answer_generation(self, answer_nodes, sample_state, mock_workflow):
        """답변 생성 계속 테스트"""
        result = answer_nodes.continue_answer_generation(sample_state)
        
        mock_workflow.continue_answer_generation.assert_called_once_with(sample_state)
        assert result == {"answer": "continued answer"}
    
    def test_workflow_instance_required(self):
        """workflow_instance 필수 검증 테스트"""
        nodes = AnswerNodes(workflow_instance=None)
        
        with pytest.raises(RuntimeError, match="workflow_instance가 설정되지 않았습니다"):
            nodes.generate_and_validate_answer({})
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        nodes = AnswerNodes(workflow_instance=MagicMock(), logger_instance=mock_logger)
        
        assert nodes.logger == mock_logger

