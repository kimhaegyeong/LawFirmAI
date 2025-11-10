# -*- coding: utf-8 -*-
"""
LangGraph 통합 테스트
전체 워크플로우 통합 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType


class TestFullWorkflow:
    """전체 워크플로우 통합 테스트"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        return LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            langgraph_enabled=True,
            use_agentic_mode=False,
            max_iterations=10,
        )
    
    @pytest.mark.asyncio
    async def test_simple_query_workflow(self, config):
        """간단한 쿼리 워크플로우 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "안녕하세요",
                "answer": "안녕하세요. 법률 관련 질문이 있으시면 도와드리겠습니다.",
                "context": [],
                "retrieved_docs": [],
                "processing_steps": ["classification", "direct_answer"],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            result = await service.process_query_async("안녕하세요", "test_session")
            
            assert "answer" in result
            assert len(result["processing_steps"]) > 0
            assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_complex_query_workflow(self, config):
        """복잡한 쿼리 워크플로우 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "계약서 작성 시 주의해야 할 법적 사항은?",
                "answer": "계약서 작성 시 주의사항...",
                "context": ["계약서 관련 법률 정보"],
                "retrieved_docs": [
                    {
                        "content": "계약서 법률 정보",
                        "metadata": {"source": "test", "similarity": 0.85},
                    }
                ],
                "processing_steps": [
                    "classification",
                    "search",
                    "context_building",
                    "answer_generation",
                ],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            result = await service.process_query_async(
                "계약서 작성 시 주의해야 할 법적 사항은?",
                "test_session"
            )
            
            assert "answer" in result
            assert len(result["retrieved_docs"]) > 0
            assert len(result["context"]) > 0
            assert "search" in result["processing_steps"]
            assert "answer_generation" in result["processing_steps"]
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, config):
        """멀티턴 대화 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            
            # 첫 번째 턴
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "계약서 검토 요청",
                "answer": "계약서 검토를 도와드리겠습니다.",
                "conversation_history": [
                    {"role": "user", "content": "계약서 검토 요청"},
                    {"role": "assistant", "content": "계약서 검토를 도와드리겠습니다."},
                ],
                "processing_steps": ["classification"],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            # 첫 번째 질문
            result1 = await service.process_query_async("계약서 검토 요청", "test_session")
            
            assert "answer" in result1
            assert "conversation_history" in result1
            
            # 두 번째 턴 (이전 대화 맥락 포함)
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "위험한 조항이 있나요?",
                "answer": "위험한 조항 분석 결과...",
                "conversation_history": [
                    {"role": "user", "content": "계약서 검토 요청"},
                    {"role": "assistant", "content": "계약서 검토를 도와드리겠습니다."},
                    {"role": "user", "content": "위험한 조항이 있나요?"},
                    {"role": "assistant", "content": "위험한 조항 분석 결과..."},
                ],
                "processing_steps": ["classification", "context_building"],
                "errors": [],
            })
            
            result2 = await service.process_query_async("위험한 조항이 있나요?", "test_session")
            
            assert "answer" in result2
            assert len(result2.get("conversation_history", [])) > 2
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, config):
        """에러 복구 워크플로우 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            
            # 첫 번째 시도 실패
            mock_workflow_instance.invoke_async = AsyncMock(side_effect=Exception("일시적 에러"))
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            # 에러 발생 시뮬레이션
            result = await service.process_query_async("테스트 질문", "test_session")
            
            assert "errors" in result
            assert len(result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_workflow(self, config):
        """체크포인트 워크플로우 테스트"""
        config.enable_checkpoint = True
        config.checkpoint_storage = CheckpointStorageType.MEMORY
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "테스트 질문",
                "answer": "테스트 답변",
                "processing_steps": [],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            result = await service.process_query_async(
                "테스트 질문",
                "test_session",
                enable_checkpoint=True
            )
            
            assert "answer" in result


class TestAgenticMode:
    """Agentic 모드 테스트"""
    
    @pytest.fixture
    def agentic_config(self):
        """Agentic 모드 설정"""
        return LangGraphConfig(
            enable_checkpoint=True,
            checkpoint_storage=CheckpointStorageType.MEMORY,
            langgraph_enabled=True,
            use_agentic_mode=True,  # Agentic 모드 활성화
        )
    
    @pytest.mark.asyncio
    async def test_agentic_mode_workflow(self, agentic_config):
        """Agentic 모드 워크플로우 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "복잡한 법률 질문",
                "answer": "Agentic 모드로 생성된 답변",
                "tools_used": ["legal_search", "document_analysis"],
                "processing_steps": ["tool_selection", "tool_execution", "answer_generation"],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(agentic_config)
            service.workflow = mock_workflow_instance
            
            result = await service.process_query_async("복잡한 법률 질문", "test_session")
            
            assert "answer" in result
            # Agentic 모드에서는 도구 사용 정보가 있을 수 있음
            if "tools_used" in result:
                assert len(result["tools_used"]) > 0


class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """응답 시간 테스트"""
        import time
        
        config = LangGraphConfig(
            enable_checkpoint=False,  # 체크포인트 비활성화로 빠른 테스트
            langgraph_enabled=True,
        )
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "테스트 질문",
                "answer": "테스트 답변",
                "processing_steps": [],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            start_time = time.time()
            result = await service.process_query_async("테스트 질문", "test_session")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert "answer" in result
            assert response_time < 5.0  # 5초 이내 응답 (Mock 환경이므로 빠를 것)
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """동시 쿼리 처리 테스트"""
        config = LangGraphConfig(
            enable_checkpoint=False,
            langgraph_enabled=True,
        )
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow') as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.invoke_async = AsyncMock(return_value={
                "query": "테스트 질문",
                "answer": "테스트 답변",
                "processing_steps": [],
                "errors": [],
            })
            MockWorkflow.return_value = mock_workflow_instance
            
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service.workflow = mock_workflow_instance
            
            # 동시에 여러 쿼리 실행
            queries = [f"질문 {i}" for i in range(5)]
            tasks = [
                service.process_query_async(query, f"session_{i}")
                for i, query in enumerate(queries)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert "answer" in result
                assert len(result["errors"]) == 0

