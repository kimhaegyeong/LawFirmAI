# -*- coding: utf-8 -*-
"""
ChatService 테스트
채팅 서비스 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any

from lawfirm_langgraph.core.services.chat_service import ChatService
from lawfirm_langgraph.core.utils.config import Config


class TestChatService:
    """ChatService 테스트"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        config = Config()
        config.database_path = ":memory:"
        return config
    
    @pytest.fixture
    def chat_service(self, config):
        """ChatService 인스턴스"""
        with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
            with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
                with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                    with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                        with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                            with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                                with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                            with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                                service = ChatService(config)
                                                                return service
    
    def test_chat_service_initialization(self, config):
        """ChatService 초기화 테스트"""
        with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
            with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
                with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                    with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                        with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                            with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                                with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                            with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                                service = ChatService(config)
                                                                
                                                                assert service.config == config
                                                                assert service.use_langgraph is False
                                                                assert service.langgraph_service is None
    
    def test_chat_service_initialization_with_langgraph(self, config):
        """LangGraph를 사용한 ChatService 초기화 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.LangGraphWorkflowService') as mock_langgraph:
            with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
                with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
                    with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                        with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                            with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                                with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                            with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                                with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                                    service = ChatService(config)
                                                                    service.use_langgraph = True
                                                                    service.langgraph_service = mock_langgraph.return_value
                                                                    
                                                                    assert service.use_langgraph is True
                                                                    assert service.langgraph_service is not None
    
    @pytest.mark.asyncio
    async def test_process_message_simple(self, chat_service):
        """간단한 메시지 처리 테스트"""
        chat_service.cache_manager.get.return_value = None  # 캐시 미스
        with patch.object(chat_service, 'validate_input', return_value=True):
            with patch.object(chat_service, '_process_phase1_context', new_callable=AsyncMock) as mock_phase1:
                with patch.object(chat_service, '_process_phase2_personalization', new_callable=AsyncMock) as mock_phase2:
                    with patch.object(chat_service, '_process_phase3_memory_quality', new_callable=AsyncMock) as mock_phase3:
                        with patch.object(chat_service, '_generate_response', new_callable=AsyncMock) as mock_generate:
                            mock_phase1.return_value = {
                                "enabled": True,
                                "context": None,
                                "errors": []
                            }
                            mock_phase2.return_value = {
                                "enabled": True,
                                "errors": []
                            }
                            mock_phase3.return_value = {
                                "enabled": True,
                                "errors": []
                            }
                            mock_generate.return_value = {
                                "response": "테스트 답변",
                                "confidence": 0.9,
                                "sources": [],
                                "question_type": "general_question",
                                "legal_references": [],
                                "processing_steps": [],
                                "metadata": {},
                                "errors": []
                            }
                            
                            result = await chat_service.process_message("테스트 질문")
                            
                            assert "response" in result
                            assert result["response"] == "테스트 답변"
                            assert "processing_time" in result
                            assert "session_id" in result
                            assert "user_id" in result
    
    @pytest.mark.asyncio
    async def test_process_message_with_invalid_input(self, chat_service):
        """잘못된 입력 처리 테스트"""
        chat_service.cache_manager.get.return_value = None  # 캐시 미스
        with patch.object(chat_service, 'validate_input', return_value=False):
            result = await chat_service.process_message("")
            
            assert "response" in result
            assert result["response"] == "올바른 질문을 입력해주세요."
            assert result["confidence"] == 0.0
            # 실제 코드는 phase_info.phase1.error 키를 사용
            phase1_info = result.get("phase_info", {}).get("phase1", {})
            assert phase1_info.get("error") == "Invalid input" or "error" in phase1_info
    
    @pytest.mark.asyncio
    async def test_process_message_with_langgraph(self, config):
        """LangGraph를 사용한 메시지 처리 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.LangGraphWorkflowService') as mock_langgraph:
            with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
                with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
                    with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                        with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                            with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                                with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                            with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                                with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                                    mock_service = MagicMock()
                                                                    # 실제 코드는 process_query (async) 메서드를 사용
                                                                    mock_service.process_query = AsyncMock(return_value={
                                                                        "answer": "LangGraph 답변",
                                                                        "confidence": 0.9,
                                                                        "sources": [],
                                                                        "processing_time": 0.1,
                                                                        "session_id": "test_session",
                                                                        "query_type": "general",
                                                                        "legal_references": [],
                                                                        "processing_steps": [],
                                                                        "metadata": {},
                                                                        "errors": []
                                                                    })
                                                                    mock_langgraph.return_value = mock_service
                                                                    
                                                                    service = ChatService(config)
                                                                    service.use_langgraph = True
                                                                    service.langgraph_service = mock_service
                                                                    service.cache_manager.get.return_value = None  # 캐시 미스
                                                                    
                                                                    result = await service.process_message("테스트 질문")
                                                                    
                                                                    # process_query의 "answer"가 "response"로 변환됨
                                                                    assert "response" in result
                                                                    assert result["response"] == "LangGraph 답변"
    
    @pytest.mark.asyncio
    async def test_process_message_error_handling(self, chat_service):
        """에러 핸들링 테스트"""
        chat_service.cache_manager.get.return_value = None  # 캐시 미스
        with patch.object(chat_service, 'validate_input', side_effect=Exception("테스트 에러")):
            result = await chat_service.process_message("테스트 질문")
            
            assert "response" in result
            assert "error" in result or "errors" in result
    
    def test_get_service_status(self, chat_service):
        """서비스 상태 조회 테스트"""
        status = chat_service.get_service_status()
        
        assert isinstance(status, dict)
        assert "service_name" in status
        assert status["service_name"] == "ChatService"
        assert "langgraph_enabled" in status
        assert "timestamp" in status
    
    def test_get_service_status_with_langgraph(self, config):
        """LangGraph를 사용한 서비스 상태 조회 테스트"""
        with patch('lawfirm_langgraph.core.workflow.workflow_service.LangGraphWorkflowService') as mock_langgraph:
            with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
                with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
                    with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                        with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                            with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                                with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                            with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                                with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                                    mock_service = MagicMock()
                                                                    mock_service.get_service_status = Mock(return_value={
                                                                        "workflow_initialized": True
                                                                    })
                                                                    mock_langgraph.return_value = mock_service
                                                                    
                                                                    service = ChatService(config)
                                                                    service.use_langgraph = True
                                                                    service.langgraph_service = mock_service
                                                                    
                                                                    status = service.get_service_status()
                                                                    
                                                                    assert status["langgraph_enabled"] is True
                                                                    assert "langgraph_status" in status or "langgraph_error" in status
    
    @pytest.mark.asyncio
    async def test_test_service(self, chat_service):
        """서비스 테스트 메서드 테스트"""
        with patch.object(chat_service, 'process_message', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "response": "테스트 답변",
                "processing_time": 0.5,
                "sources": []
            }
            
            result = await chat_service.test_service("테스트 질문")
            
            assert "test_passed" in result
            assert result["test_passed"] is True
            assert "test_message" in result
            assert result["test_message"] == "테스트 질문"
            assert "result" in result
            assert "langgraph_enabled" in result
    
    @pytest.mark.asyncio
    async def test_test_service_failure(self, chat_service):
        """서비스 테스트 실패 케이스"""
        with patch.object(chat_service, 'process_message', new_callable=AsyncMock, side_effect=Exception("테스트 에러")):
            result = await chat_service.test_service("테스트 질문")
            
            assert result["test_passed"] is False
            assert "error" in result
    
    def test_validate_input(self, chat_service):
        """입력 검증 테스트"""
        assert chat_service.validate_input("테스트 질문") is True
        assert chat_service.validate_input("") is False
        assert chat_service.validate_input("   ") is False
    
    @pytest.mark.asyncio
    async def test_process_message_with_cache(self, chat_service):
        """캐시를 사용한 메시지 처리 테스트"""
        with patch.object(chat_service, 'cache_manager') as mock_cache:
            if mock_cache:
                mock_cache.get = Mock(return_value={
                    "response": "캐시된 답변",
                    "confidence": 0.9,
                    "sources": [],
                    "processing_time": 0.1
                })
                
                result = await chat_service.process_message("테스트 질문")
                
                assert "from_cache" in result
                assert result["from_cache"] is True

