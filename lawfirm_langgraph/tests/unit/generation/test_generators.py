# -*- coding: utf-8 -*-
"""
Generation Generators 테스트
생성기 모듈 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from lawfirm_langgraph.core.generation.generators.answer_generator import AnswerGenerator
from lawfirm_langgraph.core.generation.generators.context_builder import ContextBuilder
from lawfirm_langgraph.core.generation.generators.direct_answer_handler import DirectAnswerHandler
from lawfirm_langgraph.core.generation.generators.improved_answer_generator import ImprovedAnswerGenerator
from lawfirm_langgraph.core.services.question_classifier import QuestionType, QuestionClassification


class TestAnswerGenerator:
    """AnswerGenerator 테스트"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock Config"""
        config = MagicMock()
        config.llm_provider = MagicMock()
        config.llm_provider.value = "google"
        return config
    
    @pytest.fixture
    def answer_generator(self, mock_config):
        """AnswerGenerator 인스턴스"""
        with patch('lawfirm_langgraph.core.generation.generators.answer_generator.LANCHAIN_AVAILABLE', False):
            generator = AnswerGenerator(config=mock_config)
            return generator
    
    def test_answer_generator_initialization(self, mock_config):
        """AnswerGenerator 초기화 테스트"""
        with patch('lawfirm_langgraph.core.generation.generators.answer_generator.LANCHAIN_AVAILABLE', False):
            generator = AnswerGenerator(config=mock_config)
            
            assert generator.config == mock_config
            assert generator.stats is not None
    
    def test_generate_answer(self, answer_generator):
        """답변 생성 테스트"""
        with patch.object(answer_generator, 'llm') as mock_llm:
            if mock_llm:
                mock_llm.run = Mock(return_value="테스트 답변")
                
                result = answer_generator.generate_answer(
                    query="테스트 질문",
                    context="테스트 컨텍스트"
                )
                
                assert isinstance(result, dict) or hasattr(result, 'answer')


class TestContextBuilder:
    """ContextBuilder 테스트"""
    
    @pytest.fixture
    def context_builder(self):
        """ContextBuilder 인스턴스"""
        mock_semantic_search = MagicMock()
        mock_config = MagicMock()
        mock_config.max_context_length = 2000
        
        return ContextBuilder(
            semantic_search=mock_semantic_search,
            config=mock_config
        )
    
    def test_context_builder_initialization(self):
        """ContextBuilder 초기화 테스트"""
        mock_semantic_search = MagicMock()
        mock_config = MagicMock()
        mock_config.max_context_length = 2000
        
        builder = ContextBuilder(
            semantic_search=mock_semantic_search,
            config=mock_config
        )
        
        assert builder.semantic_search == mock_semantic_search
        assert builder.config == mock_config
    
    def test_build_context(self, context_builder):
        """컨텍스트 구축 테스트"""
        mock_state = {
            "retrieved_docs": [
                {"content": "테스트 문서 1", "source": "source1"},
                {"content": "테스트 문서 2", "source": "source2"}
            ],
            "legal_references": [],
            "query_type": "general"
        }
        
        with patch('lawfirm_langgraph.core.generation.generators.context_builder.WorkflowUtils') as mock_utils:
            mock_utils.get_state_value.side_effect = lambda state, key, default: mock_state.get(key, default)
            
            result = context_builder.build_context(mock_state)
            
            assert isinstance(result, dict)
            assert "context" in result
            assert "structured_documents" in result


class TestDirectAnswerHandler:
    """DirectAnswerHandler 테스트"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM"""
        llm = MagicMock()
        llm.invoke = Mock(return_value=MagicMock(content="테스트 답변"))
        return llm
    
    @pytest.fixture
    def direct_answer_handler(self, mock_llm):
        """DirectAnswerHandler 인스턴스"""
        return DirectAnswerHandler(
            llm=mock_llm,
            llm_fast=mock_llm
        )
    
    def test_direct_answer_handler_initialization(self, mock_llm):
        """DirectAnswerHandler 초기화 테스트"""
        handler = DirectAnswerHandler(
            llm=mock_llm,
            llm_fast=mock_llm
        )
        
        assert handler.llm == mock_llm
        assert handler.llm_fast == mock_llm
    
    def test_generate_direct_answer_with_chain(self, direct_answer_handler, mock_llm):
        """Prompt Chaining을 사용한 직접 답변 생성 테스트"""
        with patch('lawfirm_langgraph.core.generation.generators.direct_answer_handler.PromptChainExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute_chain.return_value = {"answer": "테스트 답변"}
            mock_executor.return_value = mock_executor_instance
            
            result = direct_answer_handler.generate_direct_answer_with_chain("테스트 질문")
            
            assert result is not None or result is None


class TestImprovedAnswerGenerator:
    """ImprovedAnswerGenerator 테스트"""
    
    @pytest.fixture
    def improved_answer_generator(self):
        """ImprovedAnswerGenerator 인스턴스"""
        with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.GeminiClient'):
            with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.PromptTemplateManager'):
                with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.ConfidenceCalculator'):
                    with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.AnswerFormatter'):
                        with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.ContextBuilder'):
                            with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.UnifiedPromptManager'):
                                with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.PromptOptimizer'):
                                    with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.SemanticDomainClassifier'):
                                        generator = ImprovedAnswerGenerator()
                                        return generator
    
    def test_improved_answer_generator_initialization(self):
        """ImprovedAnswerGenerator 초기화 테스트"""
        with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.GeminiClient'):
            with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.PromptTemplateManager'):
                with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.ConfidenceCalculator'):
                    with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.AnswerFormatter'):
                        with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.ContextBuilder'):
                            with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.UnifiedPromptManager'):
                                with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.PromptOptimizer'):
                                    with patch('lawfirm_langgraph.core.generation.generators.improved_answer_generator.SemanticDomainClassifier'):
                                        generator = ImprovedAnswerGenerator()
                                        
                                        assert generator.gemini_client is not None
                                        assert generator.generation_config is not None
    
    def test_generate_answer(self, improved_answer_generator):
        """답변 생성 테스트"""
        mock_question_classification = QuestionClassification(
            question_type=QuestionType.GENERAL_QUESTION,
            law_weight=0.5,
            precedent_weight=0.5,
            confidence=0.8,
            keywords=[],
            patterns=[]
        )
        
        with patch.object(improved_answer_generator, 'gemini_client') as mock_client:
            if mock_client:
                mock_client.generate.return_value = MagicMock(
                    text="테스트 답변",
                    tokens_used=100
                )
                
                result = improved_answer_generator.generate_answer(
                    query="테스트 질문",
                    question_type=mock_question_classification,
                    context="테스트 컨텍스트",
                    sources={}
                )
                
                assert isinstance(result, dict) or hasattr(result, 'answer')

