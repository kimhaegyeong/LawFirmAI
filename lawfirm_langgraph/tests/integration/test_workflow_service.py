# -*- coding: utf-8 -*-
"""
LangGraphWorkflowService 테스트
워크플로우 서비스 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType


class TestLangGraphWorkflowService:
    """LangGraphWorkflowService 테스트"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            langgraph_enabled=True,
            use_agentic_mode=False,
        )
    
    @pytest.fixture
    def service(self, config):
        """워크플로우 서비스 인스턴스"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            return LangGraphWorkflowService(config)
    
    def test_service_initialization(self, config):
        """서비스 초기화 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            
            assert service.config == config
            assert service.config.langgraph_enabled is True
    
    def test_service_initialization_with_default_config(self):
        """기본 설정으로 서비스 초기화 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            with patch('lawfirm_langgraph.config.langgraph_config.LangGraphConfig') as MockConfig:
                mock_config_instance = LangGraphConfig.from_env()
                MockConfig.from_env.return_value = mock_config_instance
                
                from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
                
                service = LangGraphWorkflowService()
                
                assert service.config is not None
    
    def test_validate_config(self, service):
        """설정 유효성 검사 테스트"""
        service.config.checkpoint_db_path = "./test.db"
        
        errors = service.validate_config()
        
        assert isinstance(errors, list)
    
    @pytest.mark.asyncio
    async def test_process_query_async_success(self, service):
        """비동기 쿼리 처리 성공 테스트"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(return_value={
            "query": "테스트 질문",
            "answer": "테스트 답변",
            "context": [],
            "retrieved_docs": [],
            "processing_steps": ["step1"],
            "errors": [],
        })
        
        service.workflow = mock_workflow
        
        result = await service.process_query_async("테스트 질문", "test_session")
        
        assert "answer" in result
        assert result["answer"] == "테스트 답변"
        assert "processing_steps" in result
    
    @pytest.mark.asyncio
    async def test_process_query_async_error_handling(self, service):
        """비동기 쿼리 처리 에러 핸들링 테스트"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(side_effect=Exception("테스트 에러"))
        
        service.workflow = mock_workflow
        
        result = await service.process_query_async("테스트 질문", "test_session")
        
        assert "errors" in result
        assert len(result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_test_workflow(self, service):
        """워크플로우 테스트 메서드 테스트"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(return_value={
            "query": "계약서 작성 시 주의사항은?",
            "answer": "테스트 답변",
            "context": [],
            "retrieved_docs": [],
            "processing_steps": ["step1", "step2"],
            "errors": [],
        })
        
        service.workflow = mock_workflow
        
        test_result = await service.test_workflow("테스트 질문")
        
        assert "test_passed" in test_result
        assert "test_query" in test_result
        assert "result" in test_result
        assert test_result["test_query"] == "테스트 질문"
    
    @pytest.mark.asyncio
    async def test_test_workflow_failure(self, service):
        """워크플로우 테스트 실패 케이스"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(side_effect=Exception("에러 발생"))
        
        service.workflow = mock_workflow
        
        test_result = await service.test_workflow("테스트 질문")
        
        assert test_result["test_passed"] is False
        assert "error" in test_result
    
    def test_get_service_status(self, service):
        """서비스 상태 조회 테스트"""
        status = service.get_service_status()
        
        assert isinstance(status, dict)
        assert "langgraph_enabled" in status
        assert "checkpoint_enabled" in status
        assert "workflow_initialized" in status
    
    @pytest.mark.asyncio
    async def test_process_query_with_checkpoint(self, service):
        """체크포인트를 사용한 쿼리 처리 테스트"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(return_value={
            "query": "테스트 질문",
            "answer": "테스트 답변",
            "context": [],
            "retrieved_docs": [],
            "processing_steps": [],
            "errors": [],
        })
        
        service.workflow = mock_workflow
        service.config.enable_checkpoint = True
        
        result = await service.process_query_async(
            "테스트 질문",
            "test_session",
            enable_checkpoint=True
        )
        
        assert "answer" in result
    
    @pytest.mark.asyncio
    async def test_process_query_without_checkpoint(self, service):
        """체크포인트 없이 쿼리 처리 테스트"""
        mock_workflow = MagicMock()
        mock_workflow.invoke_async = AsyncMock(return_value={
            "query": "테스트 질문",
            "answer": "테스트 답변",
            "context": [],
            "retrieved_docs": [],
            "processing_steps": [],
            "errors": [],
        })
        
        service.workflow = mock_workflow
        
        result = await service.process_query_async(
            "테스트 질문",
            "test_session",
            enable_checkpoint=False
        )
        
        assert "answer" in result

