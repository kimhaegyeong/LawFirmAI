# -*- coding: utf-8 -*-
"""
LangChain RAG Tests
LangChain RAG 시스템 테스트
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# 테스트 대상 모듈 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

from services.langchain_rag_service import LangChainRAGService, RAGResult
from services.langfuse_client import LangfuseClient, RAGMetrics
from services.document_processor import LegalDocumentProcessor, DocumentChunk
from services.context_manager import ContextManager, ContextWindow
from services.answer_generator import AnswerGenerator, AnswerResult
from utils.langchain_config import LangChainConfig, VectorStoreType, LLMProvider


class TestLangChainConfig:
    """LangChain 설정 테스트"""
    
    def test_config_initialization(self):
        """설정 초기화 테스트"""
        config = LangChainConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_context_length == 4000
        assert config.search_k == 5
        assert config.similarity_threshold == 0.7
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 1000
    
    def test_config_from_env(self):
        """환경 변수에서 설정 로드 테스트"""
        with patch.dict(os.environ, {
            'CHUNK_SIZE': '500',
            'CHUNK_OVERLAP': '100',
            'LLM_TEMPERATURE': '0.5',
            'LANGFUSE_ENABLED': 'true'
        }):
            config = LangChainConfig.from_env()
            
            assert config.chunk_size == 500
            assert config.chunk_overlap == 100
            assert config.llm_temperature == 0.5
            assert config.langfuse_enabled == True
    
    def test_config_validation(self):
        """설정 유효성 검사 테스트"""
        config = LangChainConfig()
        errors = config.validate()
        
        # 기본 설정은 유효해야 함
        assert len(errors) == 0
        
        # 잘못된 설정 테스트
        config.chunk_size = -1
        config.chunk_overlap = 2000  # chunk_size보다 큼
        config.similarity_threshold = 2.0  # 범위 초과
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("chunk_size must be positive" in error for error in errors)
        assert any("chunk_overlap must be non-negative" in error for error in errors)
        assert any("similarity_threshold must be between 0 and 1" in error for error in errors)


class TestLangfuseClient:
    """Langfuse 클라이언트 테스트"""
    
    def test_client_initialization(self):
        """클라이언트 초기화 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        
        assert client.config == config
        assert client.enabled == False
    
    @patch('services.langfuse_client.LANGFUSE_AVAILABLE', True)
    def test_client_with_langfuse(self):
        """Langfuse 사용 가능한 경우 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.Langfuse') as mock_langfuse:
            client = LangfuseClient(config)
            
            assert client.enabled == True
            mock_langfuse.assert_called_once()
    
    def test_track_rag_query(self):
        """RAG 쿼리 추적 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        
        metrics = RAGMetrics(
            query="test query",
            response_time=1.5,
            retrieved_docs_count=3,
            context_length=1000,
            response_length=200,
            similarity_scores=[0.8, 0.7, 0.9],
            confidence_score=0.8,
            timestamp=None
        )
        
        sources = [{"title": "test doc", "similarity": 0.8}]
        
        trace_id = client.track_rag_query("test query", "test response", metrics, sources)
        
        # Langfuse가 비활성화된 경우 None 반환
        assert trace_id is None
    
    def test_track_error(self):
        """오류 추적 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        
        trace_id = client.track_error("TestError", "Test error message", {"context": "test"})
        
        # Langfuse가 비활성화된 경우 None 반환
        assert trace_id is None


class TestDocumentProcessor:
    """문서 처리기 테스트"""
    
    def test_processor_initialization(self):
        """처리기 초기화 테스트"""
        config = LangChainConfig()
        processor = LegalDocumentProcessor(config)
        
        assert processor.config == config
        assert processor.legal_patterns is not None
        assert 'law_article' in processor.legal_patterns
    
    def test_clean_text(self):
        """텍스트 정리 테스트"""
        config = LangChainConfig()
        processor = LegalDocumentProcessor(config)
        
        dirty_text = "  제1조   법률의   목적\n\n  이 법률은...  "
        clean_text = processor._clean_text(dirty_text)
        
        assert clean_text == "제1조 법률의 목적 이 법률은..."
    
    def test_preprocess_legal_content(self):
        """법률 문서 전처리 테스트"""
        config = LangChainConfig()
        processor = LegalDocumentProcessor(config)
        
        text = "제1조 법률의 목적 제2항 세부사항"
        processed = processor._preprocess_legal_content(text)
        
        assert "제1조" in processed
        assert "제2항" in processed
    
    def test_detect_document_type(self):
        """문서 타입 감지 테스트"""
        config = LangChainConfig()
        processor = LegalDocumentProcessor(config)
        
        # 법령 문서
        law_text = "제1조 이 법률은 민법에 관한 사항을 규정한다."
        doc_type = processor._detect_document_type(law_text)
        assert doc_type == "law"
        
        # 판례 문서
        precedent_text = "2023다12345 대법원 판결"
        doc_type = processor._detect_document_type(precedent_text)
        assert doc_type == "precedent"
        
        # 일반 문서
        general_text = "이것은 일반적인 문서입니다."
        doc_type = processor._detect_document_type(general_text)
        assert doc_type == "general"


class TestContextManager:
    """컨텍스트 관리자 테스트"""
    
    def test_manager_initialization(self):
        """관리자 초기화 테스트"""
        config = LangChainConfig()
        manager = ContextManager(config)
        
        assert manager.config == config
        assert manager.max_context_length == config.max_context_length
        assert len(manager.sessions) == 0
    
    def test_create_session(self):
        """세션 생성 테스트"""
        config = LangChainConfig()
        manager = ContextManager(config)
        
        session_id = "test-session-1"
        session = manager.create_context_session(session_id)
        
        assert session.session_id == session_id
        assert len(session.query_history) == 0
        assert len(session.context_windows) == 0
        assert session_id in manager.sessions
    
    def test_add_query_to_session(self):
        """세션에 쿼리 추가 테스트"""
        config = LangChainConfig()
        manager = ContextManager(config)
        
        session_id = "test-session-1"
        query = "테스트 질문입니다."
        
        result = manager.add_query_to_session(session_id, query)
        
        assert result == True
        session = manager.get_session(session_id)
        assert query in session.query_history
    
    def test_build_context_window(self):
        """컨텍스트 윈도우 구축 테스트"""
        config = LangChainConfig()
        manager = ContextManager(config)
        
        retrieved_docs = [
            {
                'content': '첫 번째 문서 내용입니다.',
                'metadata': {'source': 'doc1.txt'},
                'similarity': 0.9,
                'source': 'doc1.txt',
                'chunk_id': 'chunk1'
            },
            {
                'content': '두 번째 문서 내용입니다.',
                'metadata': {'source': 'doc2.txt'},
                'similarity': 0.8,
                'source': 'doc2.txt',
                'chunk_id': 'chunk2'
            }
        ]
        
        query = "테스트 질문"
        context_windows = manager.build_context_window(retrieved_docs, query)
        
        assert len(context_windows) == 2
        assert context_windows[0].relevance_score >= context_windows[1].relevance_score
    
    def test_get_session_context(self):
        """세션 컨텍스트 조회 테스트"""
        config = LangChainConfig()
        manager = ContextManager(config)
        
        session_id = "test-session-1"
        session = manager.create_context_session(session_id)
        
        # 컨텍스트 윈도우 추가
        context_window = ContextWindow(
            content="테스트 컨텍스트",
            metadata={},
            relevance_score=0.9,
            timestamp=None,
            source_document="test.txt",
            chunk_id="chunk1"
        )
        session.context_windows.append(context_window)
        
        context = manager.get_session_context(session_id)
        
        assert "테스트 컨텍스트" in context
        assert "[문서: test.txt]" in context


class TestAnswerGenerator:
    """답변 생성기 테스트"""
    
    def test_generator_initialization(self):
        """생성기 초기화 테스트"""
        config = LangChainConfig()
        generator = AnswerGenerator(config)
        
        assert generator.config == config
        assert generator.stats['total_queries'] == 0
    
    def test_generate_basic_answer(self):
        """기본 답변 생성 테스트"""
        config = LangChainConfig()
        generator = AnswerGenerator(config)
        
        query = "계약서 검토"
        context = "계약서에는 계약 기간이 명시되어야 합니다. 계약 조건은 명확해야 합니다."
        
        answer = generator._generate_basic_answer(query, context)
        
        assert len(answer) > 0
        assert "주어진 문서를 바탕으로" in answer
    
    def test_calculate_confidence(self):
        """신뢰도 계산 테스트"""
        config = LangChainConfig()
        generator = AnswerGenerator(config)
        
        query = "계약서 검토"
        context = "계약서에는 계약 기간이 명시되어야 합니다."
        answer = "계약서의 계약 기간이 명시되어 있습니다."
        
        confidence = generator._calculate_confidence(query, context, answer)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 키워드 매칭으로 인한 높은 신뢰도
    
    def test_extract_sources(self):
        """소스 추출 테스트"""
        config = LangChainConfig()
        generator = AnswerGenerator(config)
        
        context = "[문서: 계약서.txt]\n계약 내용입니다.\n\n[문서: 법령.txt]\n법령 내용입니다."
        
        sources = generator._extract_sources(context)
        
        assert len(sources) == 2
        assert sources[0]['title'] == "계약서.txt"
        assert sources[1]['title'] == "법령.txt"
    
    def test_generate_answer(self):
        """답변 생성 테스트"""
        config = LangChainConfig()
        generator = AnswerGenerator(config)
        
        query = "계약서 검토 요청"
        context = "[문서: 계약서.txt]\n계약서에는 계약 기간이 명시되어야 합니다."
        
        result = generator.generate_answer(query, context)
        
        assert isinstance(result, AnswerResult)
        assert len(result.answer) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.response_time > 0
        assert result.tokens_used > 0


class TestLangChainRAGService:
    """LangChain RAG 서비스 테스트"""
    
    def test_service_initialization(self):
        """서비스 초기화 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            assert service.config == config
            assert service.langfuse_client is not None
            assert service.document_processor is not None
            assert service.context_manager is not None
            assert service.answer_generator is not None
    
    def test_process_query(self):
        """쿼리 처리 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            # 모킹
            service._retrieve_documents = Mock(return_value=[
                {
                    'content': '테스트 문서 내용',
                    'metadata': {'source': 'test.txt'},
                    'similarity': 0.9,
                    'source': 'test.txt',
                    'chunk_id': 'chunk1'
                }
            ])
            
            query = "테스트 질문"
            result = service.process_query(query)
            
            assert isinstance(result, RAGResult)
            assert len(result.answer) > 0
            assert result.response_time > 0
            assert len(result.retrieved_docs) == 1
    
    def test_retrieve_documents(self):
        """문서 검색 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            # 기존 벡터 저장소 모킹
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = [
                {
                    'text': '테스트 문서',
                    'metadata': {'law_name': 'test_law.txt', 'chunk_id': 'chunk1'},
                    'score': 0.9
                }
            ]
            service.legal_vector_store = mock_vector_store
            
            docs = service._retrieve_documents("테스트 쿼리")
            
            assert len(docs) == 1
            assert docs[0]['content'] == '테스트 문서'
            assert docs[0]['similarity'] == 0.9
    
    def test_build_context(self):
        """컨텍스트 구축 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            context_windows = [
                ContextWindow(
                    content="첫 번째 컨텍스트",
                    metadata={},
                    relevance_score=0.9,
                    timestamp=None,
                    source_document="doc1.txt",
                    chunk_id="chunk1"
                ),
                ContextWindow(
                    content="두 번째 컨텍스트",
                    metadata={},
                    relevance_score=0.8,
                    timestamp=None,
                    source_document="doc2.txt",
                    chunk_id="chunk2"
                )
            ]
            
            context = service._build_context(context_windows)
            
            assert "[문서: doc1.txt]" in context
            assert "[문서: doc2.txt]" in context
            assert "첫 번째 컨텍스트" in context
            assert "두 번째 컨텍스트" in context
    
    def test_get_service_statistics(self):
        """서비스 통계 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            stats = service.get_service_statistics()
            
            assert 'rag_stats' in stats
            assert 'vector_store_stats' in stats
            assert 'context_stats' in stats
            assert 'generator_stats' in stats
            assert 'langfuse_enabled' in stats
            assert 'langchain_available' in stats
    
    def test_validate_configuration(self):
        """설정 유효성 검사 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            errors = service.validate_configuration()
            
            # LangChain이 비활성화된 경우 오류가 있어야 함
            assert len(errors) > 0
            assert any("LangChain is not available" in error for error in errors)


class TestIntegration:
    """통합 테스트"""
    
    def test_end_to_end_rag_flow(self):
        """전체 RAG 플로우 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            # 모킹 설정
            service._retrieve_documents = Mock(return_value=[
                {
                    'content': '민법 제1조는 민사에 관하여 법률에 특별한 규정이 없는 한 관습법에 의한다고 규정하고 있습니다.',
                    'metadata': {'source': '민법.txt'},
                    'similarity': 0.9,
                    'source': '민법.txt',
                    'chunk_id': 'chunk1'
                }
            ])
            
            # 쿼리 처리
            query = "민법 제1조는 무엇을 규정하고 있나요?"
            result = service.process_query(query, session_id="test-session")
            
            # 결과 검증
            assert isinstance(result, RAGResult)
            assert len(result.answer) > 0
            assert result.confidence > 0
            assert len(result.sources) > 0
            assert result.response_time > 0
            assert result.session_id == "test-session"
    
    def test_session_management(self):
        """세션 관리 테스트"""
        config = LangChainConfig()
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            service = LangChainRAGService(config)
            
            session_id = "test-session-1"
            
            # 첫 번째 쿼리
            result1 = service.process_query("첫 번째 질문", session_id=session_id)
            
            # 두 번째 쿼리 (같은 세션)
            result2 = service.process_query("두 번째 질문", session_id=session_id)
            
            # 세션 확인
            session = service.context_manager.get_session(session_id)
            assert session is not None
            assert len(session.query_history) == 2
            assert "첫 번째 질문" in session.query_history
            assert "두 번째 질문" in session.query_history
            
            # 세션 삭제
            success = service.clear_session(session_id)
            assert success == True
            
            # 삭제 확인
            session = service.context_manager.get_session(session_id)
            assert session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
