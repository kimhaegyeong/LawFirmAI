# -*- coding: utf-8 -*-
"""
Legal Workflow Enhanced 테스트
langgraph_core/workflow/legal_workflow_enhanced.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow


class TestEnhancedLegalQuestionWorkflow:
    """EnhancedLegalQuestionWorkflow 테스트"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock LangGraphConfig"""
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
        
        config = LangGraphConfig(
            enable_checkpoint=False,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            langgraph_enabled=True,
            use_agentic_mode=False
        )
        return config
    
    @pytest.fixture
    def workflow(self, mock_config):
        """EnhancedLegalQuestionWorkflow 인스턴스 생성"""
        with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.AnswerGenerator'):
            with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.LegalDataConnectorV2'):
                with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.SearchHandler'):
                    with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.StateGraph'):
                        workflow = EnhancedLegalQuestionWorkflow(config=mock_config)
                        return workflow
    
    def test_init(self, mock_config):
        """초기화 테스트"""
        with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.AnswerGenerator'):
            with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.LegalDataConnectorV2'):
                with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.SearchHandler'):
                    with patch('lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced.StateGraph'):
                        workflow = EnhancedLegalQuestionWorkflow(config=mock_config)
                        
                        assert workflow.config == mock_config
                        assert workflow.logger is not None
    
    def test_get_statistics(self, workflow):
        """통계 조회 테스트"""
        result = workflow.get_statistics()
        
        assert isinstance(result, dict)
    
    def test_process_legal_terms(self, workflow):
        """법률 용어 처리 테스트"""
        state = {
            "query": "계약 해지",
            "session_id": "test_session",
            "extracted_keywords": []
        }
        
        result = workflow.process_legal_terms(state)
        
        assert isinstance(result, dict)
        assert "extracted_keywords" in result or "query" in result

