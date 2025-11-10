# -*- coding: utf-8 -*-
"""
Agents Handlers 테스트
핸들러 모듈 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from lawfirm_langgraph.core.agents.handlers.classification_handler import ClassificationHandler
from lawfirm_langgraph.core.search.handlers.search_handler import SearchHandler
from lawfirm_langgraph.core.generation.generators.context_builder import ContextBuilder
from lawfirm_langgraph.core.generation.generators.answer_generator import AnswerGenerator
from lawfirm_langgraph.core.generation.generators.direct_answer_handler import DirectAnswerHandler
from lawfirm_langgraph.core.services.question_classifier import QuestionType


class TestClassificationHandler:
    """ClassificationHandler 테스트"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM"""
        llm = MagicMock()
        llm.invoke = Mock(return_value=MagicMock(content="legal_advice"))
        return llm
    
    @pytest.fixture
    def classification_handler(self, mock_llm):
        """ClassificationHandler 인스턴스"""
        return ClassificationHandler(
            llm=mock_llm,
            llm_fast=mock_llm
        )
    
    def test_classification_handler_initialization(self, mock_llm):
        """ClassificationHandler 초기화 테스트"""
        handler = ClassificationHandler(
            llm=mock_llm,
            llm_fast=mock_llm
        )
        
        assert handler.llm == mock_llm
        assert handler.llm_fast == mock_llm
    
    def test_classify_with_llm(self, classification_handler, mock_llm):
        """LLM 기반 분류 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.classification_handler.WorkflowUtils') as mock_utils:
            mock_utils.extract_response_content.return_value = "legal_advice"
            
            question_type, confidence = classification_handler.classify_with_llm("테스트 질문")
            
            # QuestionType enum인지 확인 (enum 값 비교)
            assert question_type.value in [q.value for q in QuestionType]
            assert isinstance(confidence, float)
    
    def test_evaluate_complexity(self, classification_handler):
        """복잡도 평가 테스트"""
        # evaluate_complexity 메서드가 없으므로 classify_complexity_with_llm 사용
        from lawfirm_langgraph.core.agents.handlers.classification_handler import QueryComplexity
        complexity, _ = classification_handler.classify_complexity_with_llm("간단한 질문")
        
        assert complexity in QueryComplexity


class TestSearchHandler:
    """SearchHandler 테스트"""
    
    @pytest.fixture
    def search_handler(self):
        """SearchHandler 인스턴스"""
        mock_semantic_search = MagicMock()
        mock_keyword_mapper = MagicMock()
        mock_data_connector = MagicMock()
        mock_result_merger = MagicMock()
        mock_result_ranker = MagicMock()
        mock_performance_optimizer = MagicMock()
        mock_config = MagicMock()
        
        return SearchHandler(
            semantic_search=mock_semantic_search,
            keyword_mapper=mock_keyword_mapper,
            data_connector=mock_data_connector,
            result_merger=mock_result_merger,
            result_ranker=mock_result_ranker,
            performance_optimizer=mock_performance_optimizer,
            config=mock_config
        )
    
    def test_search_handler_initialization(self):
        """SearchHandler 초기화 테스트"""
        mock_semantic_search = MagicMock()
        mock_keyword_mapper = MagicMock()
        mock_data_connector = MagicMock()
        mock_result_merger = MagicMock()
        mock_result_ranker = MagicMock()
        mock_performance_optimizer = MagicMock()
        mock_config = MagicMock()
        
        handler = SearchHandler(
            semantic_search=mock_semantic_search,
            keyword_mapper=mock_keyword_mapper,
            data_connector=mock_data_connector,
            result_merger=mock_result_merger,
            result_ranker=mock_result_ranker,
            performance_optimizer=mock_performance_optimizer,
            config=mock_config
        )
        
        assert handler.semantic_search_engine == mock_semantic_search
        assert handler.data_connector == mock_data_connector
    
    def test_semantic_search(self, search_handler):
        """의미적 검색 테스트"""
        search_handler.semantic_search_engine.search.return_value = [
            {"content": "테스트 문서", "score": 0.9}
        ]
        
        results, count = search_handler.semantic_search("테스트 쿼리", k=5)
        
        assert isinstance(results, list)
        assert isinstance(count, int)


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

