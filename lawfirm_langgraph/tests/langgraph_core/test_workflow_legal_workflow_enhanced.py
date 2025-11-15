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
    
    @pytest.mark.asyncio
    async def test_process_async(self, workflow):
        """비동기 처리 테스트"""
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "answer": "테스트 답변",
            "sources": [],
            "confidence": 0.9
        })
        workflow.graph = mock_graph
        
        result = await workflow.process_async(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(result, dict)
        assert "answer" in result or "errors" in result
    
    @pytest.mark.asyncio
    async def test_process_async_with_error(self, workflow):
        """에러 발생 시 비동기 처리 테스트"""
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        workflow.graph = mock_graph
        
        result = await workflow.process_async(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(result, dict)
        assert "error" in result or "errors" in result
    
    def test_get_state(self, workflow):
        """상태 조회 테스트"""
        mock_graph = Mock()
        mock_graph.get_state = Mock(return_value={"query": "test"})
        workflow.graph = mock_graph
        
        result = workflow.get_state("test_session")
        
        assert isinstance(result, dict) or result is None

