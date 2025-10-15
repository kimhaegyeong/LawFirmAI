# -*- coding: utf-8 -*-
"""
Langfuse Integration Tests
Langfuse 통합 테스트
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# 테스트 대상 모듈 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

from services.langfuse_client import LangfuseClient, MetricsCollector, RAGMetrics
from services.langchain_rag_service import LangChainRAGService
from utils.langchain_config import LangChainConfig
from datetime import datetime


class TestLangfuseIntegration:
    """Langfuse 통합 테스트"""
    
    def test_langfuse_client_with_mock(self):
        """Langfuse 클라이언트 모킹 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                mock_langfuse = MagicMock()
                mock_langfuse_class.return_value = mock_langfuse
                mock_langfuse.get_current_trace_id.return_value = "test-trace-123"
                
                client = LangfuseClient(config)
                
                assert client.is_enabled() == True
                
                # RAG 쿼리 추적 테스트
                metrics = RAGMetrics(
                    query="test query",
                    response_time=1.5,
                    retrieved_docs_count=3,
                    context_length=1000,
                    response_length=200,
                    similarity_scores=[0.8, 0.7, 0.9],
                    confidence_score=0.8,
                    timestamp=datetime.now()
                )
                
                sources = [{"title": "test doc", "similarity": 0.8}]
                
                trace_id = client.track_rag_query("test query", "test response", metrics, sources)
                
                assert trace_id == "test-trace-123"
                assert mock_langfuse.score.call_count >= 4  # 여러 메트릭 기록
    
    def test_langfuse_client_disabled(self):
        """Langfuse 비활성화 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        
        assert client.is_enabled() == False
        
        # 모든 추적 메서드가 None을 반환해야 함
        metrics = RAGMetrics(
            query="test",
            response_time=1.0,
            retrieved_docs_count=1,
            context_length=100,
            response_length=50,
            similarity_scores=[0.8],
            confidence_score=0.8,
            timestamp=datetime.now()
        )
        
        assert client.track_rag_query("test", "response", metrics, []) is None
        assert client.track_llm_call("model", "prompt", "response", 100, 1.0) is None
        assert client.track_search_performance("query", "semantic", 5, 0.5) is None
        assert client.track_error("Error", "message", {}) is None
    
    def test_metrics_collector(self):
        """메트릭 수집기 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        collector = MetricsCollector(client)
        
        # 메트릭 수집 테스트
        retrieved_docs = [
            {"content": "doc1", "similarity": 0.9},
            {"content": "doc2", "similarity": 0.8}
        ]
        
        metrics = collector.collect_rag_metrics(
            query="test query",
            response="test response",
            retrieved_docs=retrieved_docs,
            response_time=1.5
        )
        
        assert isinstance(metrics, RAGMetrics)
        assert metrics.query == "test query"
        assert metrics.response_time == 1.5
        assert metrics.retrieved_docs_count == 2
        assert metrics.confidence_score > 0
    
    def test_rag_service_with_langfuse(self):
        """Langfuse가 활성화된 RAG 서비스 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langchain_rag_service.LANCHAIN_AVAILABLE', False):
            with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
                with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                    mock_langfuse = MagicMock()
                    mock_langfuse_class.return_value = mock_langfuse
                    mock_langfuse.get_current_trace_id.return_value = "test-trace-456"
                    
                    service = LangChainRAGService(config)
                    
                    # Langfuse 클라이언트가 활성화되었는지 확인
                    assert service.langfuse_client.is_enabled() == True
                    
                    # 모킹 설정
                    service._retrieve_documents = Mock(return_value=[
                        {
                            'content': '테스트 문서',
                            'metadata': {'source': 'test.txt'},
                            'similarity': 0.9,
                            'source': 'test.txt',
                            'chunk_id': 'chunk1'
                        }
                    ])
                    
                    # 쿼리 처리
                    result = service.process_query("테스트 질문")
                    
                    # Langfuse 추적이 호출되었는지 확인
                    assert result.trace_id == "test-trace-456"
                    assert mock_langfuse.score.call_count > 0
    
    def test_error_tracking(self):
        """오류 추적 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                mock_langfuse = MagicMock()
                mock_langfuse_class.return_value = mock_langfuse
                mock_langfuse.get_current_trace_id.return_value = "error-trace-789"
                
                client = LangfuseClient(config)
                
                # 오류 추적
                trace_id = client.track_error(
                    error_type="ValueError",
                    error_message="Test error occurred",
                    context={"function": "test_function", "args": "test_args"}
                )
                
                assert trace_id == "error-trace-789"
                mock_langfuse.score.assert_called_once()
                
                # 호출된 score 메서드의 인자 확인
                call_args = mock_langfuse.score.call_args
                assert call_args[1]['name'] == "error_occurred"
                assert call_args[1]['value'] == 1.0
                assert "ValueError" in call_args[1]['comment']
    
    def test_performance_tracking(self):
        """성능 추적 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                mock_langfuse = MagicMock()
                mock_langfuse_class.return_value = mock_langfuse
                mock_langfuse.get_current_trace_id.return_value = "perf-trace-101"
                
                client = LangfuseClient(config)
                
                # LLM 호출 추적
                llm_trace_id = client.track_llm_call(
                    model="gpt-3.5-turbo",
                    prompt="test prompt",
                    response="test response",
                    tokens_used=150,
                    response_time=2.5
                )
                
                # 검색 성능 추적
                search_trace_id = client.track_search_performance(
                    query="test query",
                    search_type="semantic",
                    results_count=5,
                    response_time=0.8
                )
                
                assert llm_trace_id == "perf-trace-101"
                assert search_trace_id == "perf-trace-101"
                
                # score 메서드가 여러 번 호출되었는지 확인
                assert mock_langfuse.score.call_count >= 4  # LLM + 검색 메트릭들
    
    def test_observable_llm_wrapper(self):
        """관찰 가능한 LLM 래퍼 테스트"""
        from services.langfuse_client import ObservableLLM
        
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                mock_langfuse = MagicMock()
                mock_langfuse_class.return_value = mock_langfuse
                mock_langfuse.get_current_trace_id.return_value = "llm-trace-202"
                
                client = LangfuseClient(config)
                
                # Mock LLM 생성
                mock_llm = Mock()
                mock_llm.model_name = "test-model"
                mock_llm.generate.return_value = "Mock response"
                
                # ObservableLLM 래퍼 생성
                observable_llm = ObservableLLM(mock_llm, client)
                
                # generate 메서드 호출
                with patch('services.langfuse_client.openai') as mock_openai:
                    mock_response = Mock()
                    mock_response.choices = [Mock()]
                    mock_response.choices[0].message.content = "Langfuse response"
                    mock_openai.chat.completions.create.return_value = mock_response
                    
                    result = observable_llm.generate("test prompt")
                    
                    assert result == "Langfuse response"
                    mock_openai.chat.completions.create.assert_called_once()
    
    def test_metrics_collector_performance_summary(self):
        """메트릭 수집기 성능 요약 테스트"""
        config = LangChainConfig()
        config.langfuse_enabled = False
        
        client = LangfuseClient(config)
        collector = MetricsCollector(client)
        
        # 여러 메트릭 수집
        for i in range(5):
            retrieved_docs = [
                {"content": f"doc{i}", "similarity": 0.8 + i * 0.02}
            ]
            
            collector.collect_rag_metrics(
                query=f"query {i}",
                response=f"response {i}",
                retrieved_docs=retrieved_docs,
                response_time=1.0 + i * 0.1
            )
        
        # 성능 요약 조회
        summary = collector.get_performance_summary()
        
        assert summary['total_queries'] == 5
        assert summary['avg_response_time'] > 1.0
        assert summary['avg_confidence'] > 0.8
        assert summary['total_documents_retrieved'] == 5
    
    def test_langfuse_configuration_validation(self):
        """Langfuse 설정 유효성 검사 테스트"""
        # 유효한 설정
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "valid-secret"
        config.langfuse_public_key = "valid-public"
        
        errors = config.validate()
        assert len(errors) == 0
        
        # 잘못된 설정
        config.langfuse_enabled = True
        config.langfuse_secret_key = None
        config.langfuse_public_key = None
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("LANGFUSE_SECRET_KEY is required" in error for error in errors)
        assert any("LANGFUSE_PUBLIC_KEY is required" in error for error in errors)


class TestLangfuseMockScenarios:
    """Langfuse 모킹 시나리오 테스트"""
    
    def test_langfuse_unavailable_scenario(self):
        """Langfuse 사용 불가 시나리오"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', False):
            client = LangfuseClient(config)
            
            assert client.is_enabled() == False
            assert client.langfuse is None
            
            # 모든 추적 메서드가 안전하게 처리되는지 확인
            metrics = RAGMetrics(
                query="test",
                response_time=1.0,
                retrieved_docs_count=1,
                context_length=100,
                response_length=50,
                similarity_scores=[0.8],
                confidence_score=0.8,
                timestamp=datetime.now()
            )
            
            assert client.track_rag_query("test", "response", metrics, []) is None
            assert client.track_llm_call("model", "prompt", "response", 100, 1.0) is None
    
    def test_langfuse_initialization_failure(self):
        """Langfuse 초기화 실패 시나리오"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse', side_effect=Exception("Connection failed")):
                client = LangfuseClient(config)
                
                assert client.is_enabled() == False
                assert client.langfuse is None
    
    def test_langfuse_score_failure(self):
        """Langfuse 점수 기록 실패 시나리오"""
        config = LangChainConfig()
        config.langfuse_enabled = True
        config.langfuse_secret_key = "test-secret"
        config.langfuse_public_key = "test-public"
        
        with patch('services.langfuse_client.LANGFUSE_AVAILABLE', True):
            with patch('services.langfuse_client.Langfuse') as mock_langfuse_class:
                mock_langfuse = MagicMock()
                mock_langfuse_class.return_value = mock_langfuse
                mock_langfuse.score.side_effect = Exception("Score recording failed")
                mock_langfuse.get_current_trace_id.return_value = "test-trace"
                
                client = LangfuseClient(config)
                
                # score 메서드가 실패해도 None을 반환해야 함
                metrics = RAGMetrics(
                    query="test",
                    response_time=1.0,
                    retrieved_docs_count=1,
                    context_length=100,
                    response_length=50,
                    similarity_scores=[0.8],
                    confidence_score=0.8,
                    timestamp=datetime.now()
                )
                
                trace_id = client.track_rag_query("test", "response", metrics, [])
                assert trace_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
