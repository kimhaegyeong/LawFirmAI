# -*- coding: utf-8 -*-
"""
DocumentNodes 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.nodes.document_nodes import DocumentNodes
from core.workflow.state.state_definitions import LegalWorkflowState


class TestDocumentNodes:
    """DocumentNodes 테스트"""
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.analyze_document = Mock(return_value={"document_analysis": {}})
        workflow.prepare_documents_and_terms = Mock(return_value={"prepared_docs": []})
        return workflow
    
    @pytest.fixture
    def document_nodes(self, mock_workflow):
        """DocumentNodes 인스턴스"""
        return DocumentNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """샘플 State"""
        return {
            "input": {"query": "계약서 분석", "session_id": "test"},
            "query": "계약서 분석",
            "retrieved_docs": [],
        }
    
    def test_analyze_document(self, document_nodes, sample_state, mock_workflow):
        """문서 분석 테스트"""
        result = document_nodes.analyze_document(sample_state)
        
        mock_workflow.analyze_document.assert_called_once_with(sample_state)
        assert result == {"document_analysis": {}}
    
    def test_prepare_documents_and_terms(self, document_nodes, sample_state, mock_workflow):
        """문서 및 용어 준비 테스트"""
        result = document_nodes.prepare_documents_and_terms(sample_state)
        
        mock_workflow.prepare_documents_and_terms.assert_called_once_with(sample_state)
        assert result == {"prepared_docs": []}
    
    def test_workflow_instance_required(self):
        """workflow_instance 필수 검증 테스트"""
        nodes = DocumentNodes(workflow_instance=None)
        
        with pytest.raises(RuntimeError, match="workflow_instance가 설정되지 않았습니다"):
            nodes.analyze_document({})
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        nodes = DocumentNodes(workflow_instance=MagicMock(), logger_instance=mock_logger)
        
        assert nodes.logger == mock_logger

