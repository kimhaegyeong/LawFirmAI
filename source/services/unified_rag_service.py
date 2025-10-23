# -*- coding: utf-8 -*-
"""
Unified RAG Service
모든 RAG 기능을 통합한 단일 RAG 서비스
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..models.model_manager import LegalModelManager
from .unified_search_engine import UnifiedSearchEngine, UnifiedSearchResult
from .improved_answer_generator import ImprovedAnswerGenerator
from .question_classifier import QuestionClassifier
from .cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 응답 결과"""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    query_analysis: Dict[str, Any]
    search_result: UnifiedSearchResult
    generation_method: str
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UnifiedRAGService:
    """통합 RAG 서비스 클래스"""
    
    def __init__(self,
                 model_manager: LegalModelManager,
                 search_engine: UnifiedSearchEngine,
                 answer_generator: Optional[ImprovedAnswerGenerator] = None,
                 question_classifier: Optional[QuestionClassifier] = None,
                 enable_caching: bool = True):
        """
        통합 RAG 서비스 초기화
        
        Args:
            model_manager: 모델 관리자
            search_engine: 통합 검색 엔진
            answer_generator: 답변 생성기
            question_classifier: 질문 분류기
            enable_caching: 캐싱 활성화
        """
        self.model_manager = model_manager
        self.search_engine = search_engine
        self.answer_generator = answer_generator
        self.question_classifier = question_classifier
        
        # 캐시 매니저
        self.cache_manager = get_cache_manager() if enable_caching else None
        
        # 성능 통계
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'successful_generations': 0,
            'failed_generations': 0
        }
        
        logger.info("UnifiedRAGService initialized successfully")
    
    async def generate_response(self, 
                              query: str, 
                              context: Optional[str] = None,
                              max_length: int = 500,
                              top_k: int = 10,
                              use_cache: bool = True) -> RAGResponse:
        """
        통합 RAG 응답 생성
        
        Args:
            query: 사용자 질문
            context: 추가 컨텍스트
            max_length: 최대 응답 길이
            top_k: 검색 결과 수
            use_cache: 캐시 사용 여부
            
        Returns:
            RAGResponse: RAG 응답 결과
        """
        start_time = time.time()
        
        try:
            # 캐시 확인
            if use_cache and self.cache_manager:
                cache_key = f"rag_{hash(query)}_{max_length}_{top_k}"
                cached_response = self.cache_manager.get(cache_key)
                if cached_response:
                    self._stats['cache_hits'] += 1
                    cached_response['processing_time'] = time.time() - start_time
                    return RAGResponse(**cached_response)
            
            # 질문 분석 및 분류
            query_analysis = await self._analyze_query(query, context)
            
            # 검색 수행
            search_result = await self.search_engine.search(
                query=query,
                top_k=top_k,
                search_types=['vector', 'exact', 'semantic', 'precedent'],
                use_cache=use_cache
            )
            
            # 답변 생성
            response_result = await self._generate_answer(
                query, query_analysis, search_result, max_length
            )
            
            # 응답 구성
            rag_response = RAGResponse(
                response=response_result['answer'],
                confidence=response_result['confidence'],
                sources=search_result.results,
                query_analysis=query_analysis,
                search_result=search_result,
                generation_method=response_result['method'],
                processing_time=time.time() - start_time
            )
            
            # 캐시 저장
            if use_cache and self.cache_manager:
                try:
                    cache_data = {
                        'response': rag_response.response,
                        'confidence': rag_response.confidence,
                        'sources': rag_response.sources,
                        'query_analysis': rag_response.query_analysis,
                        'search_result': rag_response.search_result,
                        'generation_method': rag_response.generation_method,
                        'processing_time': rag_response.processing_time,
                        'timestamp': rag_response.timestamp
                    }
                    # ttl 매개변수가 지원되는지 확인
                    if hasattr(self.cache_manager, 'set_with_ttl'):
                        self.cache_manager.set_with_ttl(cache_key, cache_data, ttl=3600)
                    else:
                        self.cache_manager.set(cache_key, cache_data)
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")
            
            # 통계 업데이트
            self._update_stats(rag_response.processing_time, success=True)
            
            return rag_response
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            self._update_stats(time.time() - start_time, success=False)
            
            # 에러 응답 반환
            return RAGResponse(
                response="죄송합니다. 응답 생성 중 오류가 발생했습니다.",
                confidence=0.0,
                sources=[],
                query_analysis={'error': str(e)},
                search_result=UnifiedSearchResult(
                    query=query,
                    results=[],
                    search_time=0.0,
                    search_types_used=[],
                    total_results=0,
                    confidence=0.0
                ),
                generation_method="error_fallback",
                processing_time=time.time() - start_time
            )
    
    async def _analyze_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """질문 분석 및 분류"""
        try:
            analysis_result = {
                'original_query': query,
                'processed_query': query.strip(),
                'query_type': 'general',
                'intent': 'unknown',
                'confidence': 0.5,
                'context': context,
                'timestamp': datetime.now()
            }
            
            # 질문 분류기 사용
            if self.question_classifier:
                try:
                    classification = self.question_classifier.classify_question(query)
                    analysis_result.update({
                        'query_type': getattr(classification.question_type, 'value', 'general'),
                        'confidence': getattr(classification, 'confidence', 0.5),
                        'classification_details': classification
                    })
                except Exception as e:
                    logger.warning(f"Question classification failed: {e}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                'original_query': query,
                'processed_query': query.strip(),
                'query_type': 'general',
                'intent': 'unknown',
                'confidence': 0.3,
                'context': context,
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def _generate_answer(self, 
                             query: str, 
                             query_analysis: Dict[str, Any], 
                             search_result: UnifiedSearchResult,
                             max_length: int) -> Dict[str, Any]:
        """답변 생성"""
        try:
            logger.info(f"Generating answer for query: {query}")
            logger.info(f"Search results count: {len(search_result.results)}")
            # 1순위: 개선된 답변 생성기
            if self.answer_generator:
                try:
                    answer_result = self.answer_generator.generate_answer(
                        query=query,
                        question_type=query_analysis.get('classification_details'),
                        context="",
                        sources=search_result.results,
                        conversation_history=None
                    )
                    return {
                        'answer': answer_result.answer,
                        'confidence': 0.8,
                        'method': 'improved_generator'
                    }
                except Exception as e:
                    logger.debug(f"Improved answer generator failed: {e}")
            else:
                logger.warning("answer_generator is None, skipping improved generator")
            
            # 2순위: 모델 매니저 직접 사용 (자연스러운 응답)
            if self.model_manager:
                try:
                    # 검색 결과를 컨텍스트로 사용
                    context_text = "\n".join([
                        f"- {result.get('content', '')[:200]}..." 
                        for result in search_result.results[:5]
                    ])
                    
                    # 실제 검색 결과를 사용한 답변 생성
                    logger.info(f"Context text length: {len(context_text)} characters")
                    if context_text.strip():
                        # Gemini API를 사용하여 실제 답변 생성
                        prompt = f"""다음은 법률 관련 질문에 대한 검색 결과입니다.

질문: {query}

검색 결과:
{context_text}

위 검색 결과를 바탕으로 질문에 대한 구체적이고 완전한 답변을 제공해주세요. 
- 검색된 내용을 참고하여 정확한 정보를 제공하세요
- 법률 조문이나 판례가 있다면 구체적으로 인용하세요
- 자연스럽고 친근한 톤으로 답변하세요
- 불필요한 서론이나 반복적인 표현은 피하세요
- 추가 질문을 요청하지 말고 가능한 한 완전한 답변을 작성하세요"""

                        try:
                            # Gemini 클라이언트 직접 사용
                            from .gemini_client import GeminiClient
                            gemini_client = GeminiClient()
                            gemini_response = gemini_client.generate(prompt)
                            response = gemini_response.response
                            logger.info(f"Gemini generation successful, response length: {len(response)}")
                        except Exception as e:
                            logger.error(f"Gemini generation failed: {e}")
                            response = f"질문하신 '{query}'에 대해 관련 정보를 찾았습니다. {context_text[:200]}..."
                    else:
                        response = f"'{query}'에 대한 질문을 받았습니다. 더 구체적인 정보가 필요하시면 추가 질문을 해주세요."
                    
                    return {
                        'answer': response,
                        'confidence': 0.6,
                        'method': 'model_manager_template'
                    }
                except Exception as e:
                    logger.debug(f"Model manager generation failed: {e}")
            else:
                logger.warning("model_manager is None, skipping model manager generation")
            
            # 3순위: 템플릿 기반 답변
            return self._generate_template_answer(query, query_analysis, search_result)
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                'confidence': 0.1,
                'method': 'error_fallback'
            }
    
    def _generate_template_answer(self, 
                                query: str, 
                                query_analysis: Dict[str, Any], 
                                search_result: UnifiedSearchResult) -> Dict[str, Any]:
        """템플릿 기반 답변 생성 (자연스러운 답변으로 개선)"""
        query_type = query_analysis.get('query_type', 'general')
        
        if query_type == 'legal_advice':
            return {
                'answer': f"법률적 조언을 요청하셨습니다. '{query}'에 대해 관련 법령과 절차에 대한 일반적인 정보를 제공할 수 있지만, 구체적인 법률 자문은 변호사와 상담하시는 것을 권장합니다.",
                'confidence': 0.6,
                'method': 'template_legal'
            }
        elif query_type == 'precedent':
            if search_result.results:
                return {
                    'answer': f"'{query}'와 관련된 판례를 {len(search_result.results)}건 찾았습니다. 관련 판례에 대한 자세한 정보가 필요하시면 구체적으로 질문해주세요.",
                    'confidence': 0.7,
                    'method': 'template_precedent'
                }
            else:
                return {
                    'answer': f"'{query}'와 관련된 판례를 찾을 수 없습니다. 다른 키워드로 검색해보시거나 더 구체적인 질문을 해주세요.",
                    'confidence': 0.5,
                    'method': 'template_no_precedent'
                }
        else:
            if search_result.results:
                return {
                    'answer': f"'{query}'에 대해 관련 정보를 찾았습니다. 구체적인 답변을 위해 추가 질문을 해주시면 더 자세히 안내해드리겠습니다.",
                    'confidence': 0.6,
                    'method': 'template_general'
                }
            else:
                return {
                    'answer': f"'{query}'에 대한 질문을 받았습니다. 더 구체적인 정보가 필요하시면 추가 질문을 해주세요.",
                    'confidence': 0.4,
                    'method': 'template_general'
                }
    
    def _update_stats(self, processing_time: float, success: bool = True):
        """통계 업데이트"""
        self._stats['total_queries'] += 1
        self._stats['total_processing_time'] += processing_time
        self._stats['avg_processing_time'] = self._stats['total_processing_time'] / self._stats['total_queries']
        
        if success:
            self._stats['successful_generations'] += 1
        else:
            self._stats['failed_generations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return self._stats.copy()
    
    def clear_cache(self):
        """캐시 클리어"""
        if self.cache_manager:
            self.cache_manager.clear()
        logger.info("RAG cache cleared")
