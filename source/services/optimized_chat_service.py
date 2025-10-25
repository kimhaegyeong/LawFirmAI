#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 ChatService
성능 최적화된 모델 관리, 검색 엔진, 캐싱 시스템 통합
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
import logging

# 프로젝트 모듈 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.utils.config import Config
from source.services.optimized_model_manager import model_manager
from source.services.optimized_hybrid_search_engine import OptimizedHybridSearchEngine, OptimizedSearchConfig
from source.services.integrated_cache_system import cache_system
from source.services.optimized_hybrid_classifier import OptimizedHybridQuestionClassifier
from source.services.improved_answer_generator import ImprovedAnswerGenerator
from source.services.answer_structure_enhancer import QuestionType

class OptimizedChatService:
    """최적화된 채팅 서비스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 성능 최적화 설정
        self.performance_config = {
            'enable_caching': True,
            'enable_parallel_search': True,
            'enable_model_preloading': True,
            'max_concurrent_requests': 10,
            'response_timeout': 15.0
        }
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 성능 통계
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0,
            'error_count': 0
        }
        
        self.logger.info("OptimizedChatService initialized with performance optimizations")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 질문 분류기 초기화
            self.question_classifier = OptimizedHybridQuestionClassifier()
            
            # 최적화된 검색 엔진 초기화
            search_config = OptimizedSearchConfig(
                max_results_per_type=5,
                parallel_search=self.performance_config['enable_parallel_search'],
                cache_enabled=self.performance_config['enable_caching'],
                timeout_seconds=3.0,
                min_score_threshold=0.3
            )
            self.search_engine = OptimizedHybridSearchEngine(search_config)
            
            # 답변 생성기 초기화
            self.answer_generator = ImprovedAnswerGenerator(self.config)
            
            # 모델 미리 로딩
            if self.performance_config['enable_model_preloading']:
                self._preload_models()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _preload_models(self):
        """모델 미리 로딩"""
        try:
            model_configs = {
                'jhgan/ko-sroberta-multitask': {
                    'device': 'cpu',
                    'enable_quantization': True
                }
            }
            
            model_manager.preload_models(model_configs)
            self.logger.info("Models preloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Model preloading failed: {e}")
    
    async def process_message(self, message: str, context: Optional[str] = None, 
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """최적화된 메시지 처리"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 입력 검증
            if not self._validate_input(message):
                return self._create_error_response("Invalid input", start_time)
            
            # 캐시 확인
            if self.performance_config['enable_caching']:
                cached_response = self._check_cache(message, context)
                if cached_response:
                    self.stats['cache_hits'] += 1
                    return cached_response
            
            # 비동기 처리 파이프라인
            response = await self._process_optimized_pipeline(message, context, start_time)
            
            # 캐시 저장
            if self.performance_config['enable_caching']:
                self._store_cache(message, context, response)
            
            return response
            
        except asyncio.TimeoutError:
            self.stats['error_count'] += 1
            return self._create_error_response("Request timeout", start_time)
        except Exception as e:
            self.stats['error_count'] += 1
            self.logger.error(f"Error processing message: {e}")
            return self._create_error_response(f"Processing error: {str(e)}", start_time)
    
    def _validate_input(self, message: str) -> bool:
        """입력 검증"""
        if not message or not isinstance(message, str):
            return False
        
        if len(message.strip()) == 0:
            return False
        
        if len(message) > 10000:  # 최대 길이 제한
            return False
        
        return True
    
    def _check_cache(self, message: str, context: Optional[str]) -> Optional[Dict[str, Any]]:
        """캐시 확인"""
        try:
            # 질문 분류 캐시 확인
            classification = cache_system.get_question_classification(message)
            if not classification:
                return None
            
            # 컨텍스트 해시 생성
            context_hash = self._generate_context_hash(context)
            
            # 답변 캐시 확인
            answer = cache_system.get_answer(message, classification.question_type.value, context_hash)
            if answer:
                return {
                    "response": answer,
                    "confidence": classification.confidence,
                    "question_type": classification.question_type.value,
                    "sources": [],
                    "cached": True,
                    "processing_time": 0.0
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None
    
    def _store_cache(self, message: str, context: Optional[str], response: Dict[str, Any]):
        """캐시 저장"""
        try:
            # 질문 분류 결과 저장
            if 'question_type' in response:
                cache_system.put_question_classification(message, response['question_type'])
            
            # 답변 저장
            if 'response' in response:
                context_hash = self._generate_context_hash(context)
                cache_system.put_answer(
                    message, 
                    response.get('question_type', 'unknown'), 
                    context_hash, 
                    response['response']
                )
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def _generate_context_hash(self, context: Optional[str]) -> str:
        """컨텍스트 해시 생성"""
        if not context:
            return "no_context"
        
        import hashlib
        return hashlib.md5(context.encode()).hexdigest()[:16]
    
    async def _process_optimized_pipeline(self, message: str, context: Optional[str], 
                                        start_time: float) -> Dict[str, Any]:
        """최적화된 처리 파이프라인"""
        
        # 1. 질문 분류 (캐시 우선)
        classification = cache_system.get_question_classification(message)
        if not classification:
            classification = self.question_classifier.classify(message)
            cache_system.put_question_classification(message, classification)
        
        # 2. 병렬 검색 실행
        search_results = await self._parallel_search(message, classification)
        
        # 3. 답변 생성
        answer_result = self.answer_generator.generate_answer(
            query=message,
            question_type=classification,
            context=context or "",
            sources=search_results,
            conversation_history=None
        )
        
        # 4. 응답 시간 계산
        processing_time = time.time() - start_time
        self._update_stats(processing_time)
        
        return {
            "response": answer_result.answer,
            "confidence": answer_result.confidence,
            "question_type": classification.question_type.value,
            "sources": self._format_sources(search_results),
            "processing_time": processing_time,
            "cached": False
        }
    
    async def _parallel_search(self, message: str, classification) -> Dict[str, List[Dict[str, Any]]]:
        """병렬 검색 실행"""
        try:
            search_results = await self.search_engine.search_with_question_type(
                query=message,
                question_type=classification.question_type,
                max_results=20
            )
            
            # 결과를 기존 형식으로 변환
            formatted_results = {
                'laws': [],
                'precedents': [],
                'general': []
            }
            
            for result in search_results:
                if result.source_type == 'law':
                    formatted_results['laws'].append({
                        'content': result.content,
                        'score': result.score,
                        'metadata': result.metadata
                    })
                elif result.source_type == 'precedent':
                    formatted_results['precedents'].append({
                        'content': result.content,
                        'score': result.score,
                        'metadata': result.metadata
                    })
                else:
                    formatted_results['general'].append({
                        'content': result.content,
                        'score': result.score,
                        'metadata': result.metadata
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {'laws': [], 'precedents': [], 'general': []}
    
    def _format_sources(self, search_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """검색 결과를 소스 리스트로 변환"""
        sources = []
        
        for category, results in search_results.items():
            for result in results[:3]:  # 각 카테고리별 최대 3개
                if 'metadata' in result and 'source' in result['metadata']:
                    sources.append(result['metadata']['source'])
        
        return sources[:5]  # 전체 최대 5개
    
    def _update_stats(self, processing_time: float):
        """통계 업데이트"""
        self.stats['total_response_time'] += processing_time
        self.stats['avg_response_time'] = (
            self.stats['total_response_time'] / self.stats['total_requests']
        )
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """에러 응답 생성"""
        processing_time = time.time() - start_time
        
        return {
            "response": f"죄송합니다. 처리 중 오류가 발생했습니다: {error_message}",
            "confidence": 0.0,
            "question_type": "error",
            "sources": [],
            "error": error_message,
            "processing_time": processing_time,
            "cached": False
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            "status": "operational",
            "components": {
                "question_classifier": "active",
                "search_engine": "active",
                "answer_generator": "active",
                "cache_system": "active" if self.performance_config['enable_caching'] else "disabled",
                "model_manager": "active"
            },
            "performance_config": self.performance_config,
            "stats": self.stats,
            "cache_stats": cache_system.get_stats(),
            "model_stats": model_manager.get_stats(),
            "search_stats": self.search_engine.get_stats()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )
        
        error_rate = (
            self.stats['error_count'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'cache_stats': cache_system.get_stats(),
            'model_stats': model_manager.get_stats(),
            'search_stats': self.search_engine.get_stats()
        }
    
    def clear_cache(self):
        """캐시 정리"""
        cache_system.clear_all()
        self.search_engine.clear_cache()
        self.logger.info("All caches cleared")
    
    def optimize_performance(self, config: Dict[str, Any]):
        """성능 설정 동적 조정"""
        self.performance_config.update(config)
        
        # 검색 엔진 설정 업데이트
        if 'enable_parallel_search' in config:
            self.search_engine.config.parallel_search = config['enable_parallel_search']
        
        if 'enable_caching' in config:
            self.search_engine.config.cache_enabled = config['enable_caching']
        
        self.logger.info(f"Performance configuration updated: {config}")
