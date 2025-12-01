# -*- coding: utf-8 -*-
"""
EthicalRejectionNode 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch
from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
from core.workflow.state.state_definitions import LegalWorkflowState


class TestEthicalRejectionNode:
    """EthicalRejectionNode 테스트"""
    
    @pytest.fixture
    def ethical_node(self):
        """EthicalRejectionNode 인스턴스"""
        return EthicalRejectionNode(logger_instance=Mock())
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "해킹 방법 알려주세요", "session_id": "test"},
            "query": "해킹 방법 알려주세요",
            "is_ethically_problematic": True,
            "ethical_rejection_reason": "불법 행위 조장",
        }
    
    def test_generate_rejection_response(self, ethical_node, sample_state):
        """윤리적 거부 응답 생성 테스트"""
        with patch('core.workflow.nodes.ethical_rejection_node.WorkflowUtils') as mock_utils:
            mock_utils.get_state_value.return_value = {}
            mock_utils.set_state_value.return_value = None
            
            with patch('core.workflow.nodes.ethical_rejection_node.add_processing_step') as mock_step:
                result = ethical_node.generate_rejection_response(
                    sample_state,
                    rejection_reason="불법 행위 조장"
                )
                
                assert result == sample_state
                assert mock_utils.set_state_value.called
                assert mock_step.called
    
    def test_generate_rejection_response_without_reason(self, ethical_node, sample_state):
        """거부 사유 없이 윤리적 거부 응답 생성 테스트"""
        with patch('core.workflow.nodes.ethical_rejection_node.WorkflowUtils') as mock_utils:
            mock_utils.get_state_value.return_value = {}
            mock_utils.set_state_value.return_value = None
            
            with patch('core.workflow.nodes.ethical_rejection_node.add_processing_step'):
                result = ethical_node.generate_rejection_response(sample_state, rejection_reason=None)
                
                assert result == sample_state
    
    def test_ethical_rejection_node_static(self, sample_state):
        """윤리적 거부 노드 정적 메서드 테스트"""
        with patch('core.workflow.nodes.ethical_rejection_node.WorkflowUtils') as mock_utils:
            mock_utils.get_state_value.return_value = "불법 행위 조장"
            mock_utils.set_state_value.return_value = None
            
            with patch('core.workflow.nodes.ethical_rejection_node.add_processing_step'):
                result = EthicalRejectionNode.ethical_rejection_node(sample_state)
                
                assert result == sample_state
    
    def test_default_rejection_message(self, ethical_node):
        """기본 거부 메시지 확인 테스트"""
        assert ethical_node.DEFAULT_REJECTION_MESSAGE is not None
        assert len(ethical_node.DEFAULT_REJECTION_MESSAGE) > 0
        assert "불법 행위" in ethical_node.DEFAULT_REJECTION_MESSAGE
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        node = EthicalRejectionNode(logger_instance=mock_logger)
        
        assert node.logger == mock_logger
    
    def test_initialization_without_logger(self):
        """로거 없이 초기화 테스트"""
        node = EthicalRejectionNode()
        
        assert node.logger is not None

