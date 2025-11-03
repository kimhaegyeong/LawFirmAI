# -*- coding: utf-8 -*-
"""
Langfuse Client
Langfuse 클라이언트 및 관찰성 구현
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

try:
    from langfuse import Langfuse, observe, trace
    from langfuse.openai import openai
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Mock classes for when Langfuse is not available
    class Langfuse:
        def __init__(self, *args, **kwargs):
            pass
        def score(self, *args, **kwargs):
            pass
        def get_current_trace_id(self):
            return "mock-trace-id"

    def trace(*args, **kwargs):
        """Mock trace decorator"""
        def decorator(func):
            return func
        return decorator

    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    class openai:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return type('MockResponse', (), {
                        'choices': [type('MockChoice', (), {
                            'message': type('MockMessage', (), {'content': 'Mock response'})()
                        })()]
                    })()

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """RAG 성능 메트릭"""
    query: str
    response_time: float
    retrieved_docs_count: int
    context_length: int
    response_length: int
    similarity_scores: List[float]
    confidence_score: float
    timestamp: datetime


class LangfuseClient:
    """Langfuse 클라이언트 및 관찰성 관리"""

    def __init__(self, config):
        """Langfuse 클라이언트 초기화"""
        self.config = config
        self.langfuse = None
        self.enabled = False

        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse is not available. Install with: pip install langfuse")
            return

        if not config.langfuse_enabled:
            logger.info("Langfuse is disabled in configuration")
            return

        try:
            self.langfuse = Langfuse(
                secret_key=config.langfuse_secret_key,
                public_key=config.langfuse_public_key,
                host=config.langfuse_host,
                debug=config.langfuse_debug
            )
            self.enabled = True
            logger.info("Langfuse client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """Langfuse가 활성화되어 있는지 확인"""
        return self.enabled and self.langfuse is not None

    def track_rag_query(self, query: str, response: str, metrics: RAGMetrics,
                       sources: List[Dict[str, Any]]) -> Optional[str]:
        """RAG 쿼리 추적"""
        if not self.is_enabled():
            return None

        try:
            trace_id = self.langfuse.get_current_trace_id()

            # RAG 성능 메트릭 기록
            self.langfuse.score(
                name="rag_response_time",
                value=metrics.response_time,
                trace_id=trace_id,
                comment=f"Response time for query: {query[:50]}..."
            )

            self.langfuse.score(
                name="rag_retrieved_docs",
                value=metrics.retrieved_docs_count,
                trace_id=trace_id,
                comment=f"Number of retrieved documents"
            )

            self.langfuse.score(
                name="rag_context_length",
                value=metrics.context_length,
                trace_id=trace_id,
                comment=f"Context length used"
            )

            self.langfuse.score(
                name="rag_confidence",
                value=metrics.confidence_score,
                trace_id=trace_id,
                comment=f"Confidence score for response"
            )

            # 평균 유사도 점수 기록
            if metrics.similarity_scores:
                avg_similarity = sum(metrics.similarity_scores) / len(metrics.similarity_scores)
                self.langfuse.score(
                    name="rag_avg_similarity",
                    value=avg_similarity,
                    trace_id=trace_id,
                    comment=f"Average similarity score of retrieved documents"
                )

            logger.info(f"RAG query tracked with trace_id: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to track RAG query: {e}")
            return None

    def track_llm_call(self, model: str, prompt: str, response: str,
                      tokens_used: int, response_time: float) -> Optional[str]:
        """LLM 호출 추적"""
        if not self.is_enabled():
            return None

        try:
            trace_id = self.langfuse.get_current_trace_id()

            # LLM 성능 메트릭 기록
            self.langfuse.score(
                name="llm_response_time",
                value=response_time,
                trace_id=trace_id,
                comment=f"LLM response time for model: {model}"
            )

            self.langfuse.score(
                name="llm_tokens_used",
                value=tokens_used,
                trace_id=trace_id,
                comment=f"Tokens used in LLM call"
            )

            logger.info(f"LLM call tracked with trace_id: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to track LLM call: {e}")
            return None

    def track_search_performance(self, query: str, search_type: str,
                               results_count: int, response_time: float) -> Optional[str]:
        """검색 성능 추적"""
        if not self.is_enabled():
            return None

        try:
            trace_id = self.langfuse.get_current_trace_id()

            self.langfuse.score(
                name="search_response_time",
                value=response_time,
                trace_id=trace_id,
                comment=f"Search response time for {search_type} search"
            )

            self.langfuse.score(
                name="search_results_count",
                value=results_count,
                trace_id=trace_id,
                comment=f"Number of search results"
            )

            logger.info(f"Search performance tracked with trace_id: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to track search performance: {e}")
            return None

    def track_error(self, error_type: str, error_message: str,
                   context: Dict[str, Any]) -> Optional[str]:
        """오류 추적"""
        if not self.is_enabled():
            return None

        try:
            trace_id = self.langfuse.get_current_trace_id()

            self.langfuse.score(
                name="error_occurred",
                value=1.0,
                trace_id=trace_id,
                comment=f"Error: {error_type} - {error_message}"
            )

            logger.error(f"Error tracked with trace_id: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to track error: {e}")
            return None

    def get_current_trace_id(self) -> Optional[str]:
        """현재 추적 ID 반환"""
        if not self.is_enabled():
            return None

        try:
            return self.langfuse.get_current_trace_id()
        except Exception as e:
            logger.error(f"Failed to get current trace ID: {e}")
            return None

    def track_answer_quality_metrics(self, query: str, answer: str,
                                     confidence: float, sources_count: int,
                                     legal_refs_count: int, processing_time: float,
                                     has_errors: bool, overall_quality: float) -> Optional[str]:
        """
        답변 품질 메트릭 추적

        Args:
            query: 사용자 질문
            answer: 답변 내용
            confidence: 신뢰도 점수
            sources_count: 소스 개수
            legal_refs_count: 법률 참조 개수
            processing_time: 처리 시간
            has_errors: 에러 발생 여부
            overall_quality: 종합 품질 점수

        Returns:
            Optional[str]: Trace ID
        """
        if not self.is_enabled():
            return None

        try:
            # Trace 생성 (Langfuse 방식)
            trace = self.langfuse.trace(
                name="answer_quality_tracking",
                input={"query": query, "answer_length": len(answer)},
                output={"overall_quality": overall_quality, "confidence": confidence},
                metadata={
                    "sources_count": sources_count,
                    "legal_refs_count": legal_refs_count,
                    "processing_time": processing_time,
                    "has_errors": has_errors
                }
            )

            trace_id = trace.id
            logger.info(f"Trace created: {trace_id}")

            # 메트릭을 score로 기록
            self.langfuse.score(
                name="answer_quality_score",
                value=overall_quality,
                trace_id=trace_id,
                comment=f"Overall quality score for query: {query[:50]}..."
            )

            self.langfuse.score(
                name="answer_confidence",
                value=confidence,
                trace_id=trace_id,
                comment=f"Confidence score"
            )

            self.langfuse.score(
                name="sources_count",
                value=sources_count,
                trace_id=trace_id,
                comment=f"Number of sources"
            )

            self.langfuse.score(
                name="legal_references_count",
                value=legal_refs_count,
                trace_id=trace_id,
                comment=f"Number of legal references"
            )

            self.langfuse.score(
                name="processing_time",
                value=processing_time,
                trace_id=trace_id,
                comment=f"Processing time in seconds"
            )

            self.langfuse.score(
                name="has_errors",
                value=1.0 if has_errors else 0.0,
                trace_id=trace_id,
                comment=f"Error occurrence indicator"
            )

            # 로깅
            logger.info(f"Answer Quality Metrics tracked:")
            logger.info(f"  Query: {query[:100]}")
            logger.info(f"  Answer Length: {len(answer)} chars")
            logger.info(f"  Confidence: {confidence:.2f}")
            logger.info(f"  Sources Count: {sources_count}")
            logger.info(f"  Legal References: {legal_refs_count}")
            logger.info(f"  Processing Time: {processing_time:.2f}s")
            logger.info(f"  Has Errors: {has_errors}")
            logger.info(f"  Overall Quality: {overall_quality:.2f}")
            logger.info(f"  Trace ID: {trace_id}")

            return trace_id

        except Exception as e:
            logger.error(f"Failed to track answer quality metrics: {e}")
            return None


def track_performance(func):
    """성능 추적 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()

            # 성능 메트릭 기록
            if hasattr(args[0], 'langfuse_client') and args[0].langfuse_client:
                args[0].langfuse_client.track_llm_call(
                    model=getattr(args[0], 'model_name', 'unknown'),
                    prompt=str(kwargs.get('query', '')),
                    response=str(result),
                    tokens_used=len(str(result).split()),
                    response_time=end_time - start_time
                )

            return result
        except Exception as e:
            end_time = time.time()

            # 오류 추적
            if hasattr(args[0], 'langfuse_client') and args[0].langfuse_client:
                args[0].langfuse_client.track_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"function": func.__name__, "args": str(args[1:]), "kwargs": str(kwargs)}
                )

            raise

    return wrapper


class ObservableLLM:
    """관찰 가능한 LLM 래퍼"""

    def __init__(self, llm, langfuse_client: LangfuseClient):
        """관찰 가능한 LLM 초기화"""
        self.llm = llm
        self.langfuse_client = langfuse_client

    @observe()
    def generate(self, prompt: str, **kwargs) -> str:
        """LLM 응답 생성 (Langfuse 추적 포함)"""
        start_time = time.time()

        try:
            if LANGFUSE_AVAILABLE and self.langfuse_client.is_enabled():
                # Langfuse를 통한 LLM 호출
                response = openai.chat.completions.create(
                    model=self.llm.model_name if hasattr(self.llm, 'model_name') else 'unknown',
                    messages=[
                        {"role": "system", "content": "You are a legal expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 1000)
                )
                result = response.choices[0].message.content
            else:
                # 일반 LLM 호출
                result = self.llm.generate(prompt, **kwargs)

            end_time = time.time()

            # 성능 추적
            if self.langfuse_client.is_enabled():
                self.langfuse_client.track_llm_call(
                    model=getattr(self.llm, 'model_name', 'unknown'),
                    prompt=prompt,
                    response=result,
                    tokens_used=len(result.split()),
                    response_time=end_time - start_time
                )

            return result

        except Exception as e:
            end_time = time.time()

            # 오류 추적
            if self.langfuse_client.is_enabled():
                self.langfuse_client.track_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"prompt": prompt, "kwargs": kwargs}
                )

            raise


class MetricsCollector:
    """메트릭 수집기"""

    def __init__(self, langfuse_client: LangfuseClient):
        """메트릭 수집기 초기화"""
        self.langfuse_client = langfuse_client
        self.metrics_history: List[RAGMetrics] = []

    def collect_rag_metrics(self, query: str, response: str,
                           retrieved_docs: List[Dict[str, Any]],
                           response_time: float) -> RAGMetrics:
        """RAG 메트릭 수집"""
        similarity_scores = [doc.get('similarity', 0.0) for doc in retrieved_docs]

        metrics = RAGMetrics(
            query=query,
            response_time=response_time,
            retrieved_docs_count=len(retrieved_docs),
            context_length=sum(len(doc.get('content', '')) for doc in retrieved_docs),
            response_length=len(response),
            similarity_scores=similarity_scores,
            confidence_score=self._calculate_confidence(similarity_scores, response),
            timestamp=datetime.now()
        )

        self.metrics_history.append(metrics)

        # Langfuse에 메트릭 전송
        if self.langfuse_client.is_enabled():
            self.langfuse_client.track_rag_query(
                query=query,
                response=response,
                metrics=metrics,
                sources=retrieved_docs
            )

        return metrics

    def _calculate_confidence(self, similarity_scores: List[float], response: str) -> float:
        """신뢰도 점수 계산"""
        if not similarity_scores:
            return 0.0

        # 평균 유사도 점수 기반 신뢰도 계산
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        # 응답 길이 기반 보정 (너무 짧거나 긴 응답은 신뢰도 감소)
        response_length_factor = min(1.0, len(response) / 100)  # 100자 기준

        confidence = avg_similarity * response_length_factor
        return min(1.0, max(0.0, confidence))

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        response_times = [m.response_time for m in self.metrics_history]
        confidence_scores = [m.confidence_score for m in self.metrics_history]

        return {
            "total_queries": len(self.metrics_history),
            "avg_response_time": sum(response_times) / len(response_times),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "total_documents_retrieved": sum(m.retrieved_docs_count for m in self.metrics_history),
            "avg_context_length": sum(m.context_length for m in self.metrics_history) / len(self.metrics_history)
        }
