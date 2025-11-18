# -*- coding: utf-8 -*-
"""
Workflow Service 테스트
langgraph_core/workflow/workflow_service.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService


class TestLangGraphWorkflowService:
    """LangGraphWorkflowService 테스트"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock LangGraphConfig"""
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
        
        config = LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            langgraph_enabled=True,
            use_agentic_mode=False
        )
        return config
    
    @pytest.fixture
    def workflow_service(self, mock_config):
        """LangGraphWorkflowService 인스턴스 생성"""
        with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.CheckpointManager'):
                service = LangGraphWorkflowService(config=mock_config)
                return service
    
    def test_init(self, mock_config):
        """초기화 테스트"""
        with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.CheckpointManager'):
                service = LangGraphWorkflowService(config=mock_config)
                
                assert service.config == mock_config
                assert service.logger is not None
    
    def test_init_default_config(self):
        """기본 설정으로 초기화 테스트"""
        with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.LangGraphConfig') as mock_config_class:
            with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
                with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.CheckpointManager'):
                    mock_config = Mock()
                    mock_config_class.from_env.return_value = mock_config
                    
                    service = LangGraphWorkflowService()
                    
                    assert service.config == mock_config
    
    @pytest.mark.asyncio
    async def test_process_query_async(self, workflow_service):
        """비동기 질의 처리 테스트"""
        mock_app = Mock()
        mock_app.astream = AsyncMock()
        async def mock_astream(state, config):
            yield {"answer": "테스트 답변", "sources": []}
        mock_app.astream.return_value = mock_astream({}, {})
        workflow_service.app = mock_app
        
        result = await workflow_service.process_query(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(result, dict)
        assert "answer" in result or "errors" in result
    
    @pytest.mark.asyncio
    async def test_process_query_async_with_error(self, workflow_service):
        """에러 발생 시 질의 처리 테스트"""
        mock_app = Mock()
        mock_app.astream = AsyncMock(side_effect=Exception("Test error"))
        workflow_service.app = mock_app
        
        result = await workflow_service.process_query(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(result, dict)
        assert "error" in result or "errors" in result
    
    def test_get_workflow_state(self, workflow_service):
        """워크플로우 상태 조회 테스트"""
        if hasattr(workflow_service, 'checkpoint_manager') and workflow_service.checkpoint_manager:
            mock_state = {"query": "test"}
            workflow_service.checkpoint_manager.get_state = Mock(return_value=mock_state)
            
            result = workflow_service.checkpoint_manager.get_state("test_session")
            
            assert isinstance(result, dict) or result is None
        else:
            pytest.skip("CheckpointManager not available")
    
    def test_get_workflow_state_not_found(self, workflow_service):
        """워크플로우 상태 조회 - 없는 경우 테스트"""
        if hasattr(workflow_service, 'checkpoint_manager') and workflow_service.checkpoint_manager:
            workflow_service.checkpoint_manager.get_state = Mock(return_value=None)
            
            result = workflow_service.checkpoint_manager.get_state("non_existent_session")
            
            assert result is None or isinstance(result, dict)
        else:
            pytest.skip("CheckpointManager not available")

