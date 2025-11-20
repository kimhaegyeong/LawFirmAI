# -*- coding: utf-8 -*-
"""
향상된 캐싱 전략
검색 결과 및 컨텍스트 빌드 결과 캐싱 확대
"""

import logging
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EnhancedCacheStrategy:
    """향상된 캐싱 전략 클래스"""
    
    def __init__(self, base_cache: Any = None):
        """
        초기화
        
        Args:
            base_cache: 기본 캐시 인스턴스 (PerformanceCache 등)
        """
        self.base_cache = base_cache
        self.logger = logger
        
        # 인메모리 캐시 (빠른 접근용)
        self._context_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._search_results_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._query_optimization_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        
        # 캐시 통계
        self.stats = {
            'context_hits': 0,
            'context_misses': 0,
            'search_results_hits': 0,
            'search_results_misses': 0,
            'query_optimization_hits': 0,
            'query_optimization_misses': 0
        }
        
        # TTL 설정 (초)
        self.ttl = {
            'context': 3600.0,  # 1시간
            'search_results': 7200.0,  # 2시간
            'query_optimization': 1800.0  # 30분
        }
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        캐시 키 생성
        
        Args:
            prefix: 키 접두사
            *args: 위치 인자
            **kwargs: 키워드 인자
        
        Returns:
            MD5 해시된 캐시 키
        """
        key_parts = [prefix]
        
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(json.dumps(sorted_kwargs, sort_keys=True))
        
        key_string = ':'.join(key_parts)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_context(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                   query_type: str = "") -> Optional[Dict[str, Any]]:
        """
        컨텍스트 빌드 결과 가져오기
        
        Args:
            query: 질의
            retrieved_docs: 검색된 문서 리스트
            query_type: 질의 유형
        
        Returns:
            캐시된 컨텍스트 또는 None
        """
        # 문서 ID 리스트로 키 생성 (문서 내용이 변경되지 않았다고 가정)
        doc_ids = [doc.get('id') or doc.get('doc_id') or str(i) 
                  for i, doc in enumerate(retrieved_docs[:20])]  # 최대 20개만 사용
        
        cache_key = self._generate_cache_key(
            "context",
            query,
            query_type,
            doc_ids
        )
        
        if cache_key in self._context_cache:
            cached_data, timestamp = self._context_cache[cache_key]
            if time.time() - timestamp < self.ttl['context']:
                self.stats['context_hits'] += 1
                self.logger.debug(f"Context cache hit: {query[:50]}...")
                return cached_data
            else:
                del self._context_cache[cache_key]
        
        self.stats['context_misses'] += 1
        return None
    
    def put_context(self, query: str, retrieved_docs: List[Dict[str, Any]],
                   query_type: str, context: Dict[str, Any]) -> bool:
        """
        컨텍스트 빌드 결과 저장
        
        Args:
            query: 질의
            retrieved_docs: 검색된 문서 리스트
            query_type: 질의 유형
            context: 컨텍스트 딕셔너리
        
        Returns:
            저장 성공 여부
        """
        try:
            doc_ids = [doc.get('id') or doc.get('doc_id') or str(i) 
                      for i, doc in enumerate(retrieved_docs[:20])]
            
            cache_key = self._generate_cache_key(
                "context",
                query,
                query_type,
                doc_ids
            )
            
            # 캐시 크기 제한 (최대 100개)
            if len(self._context_cache) >= 100:
                oldest_key = min(
                    self._context_cache.keys(),
                    key=lambda k: self._context_cache[k][1]
                )
                del self._context_cache[oldest_key]
            
            self._context_cache[cache_key] = (context, time.time())
            self.logger.debug(f"Context cached: {query[:50]}...")
            
            # 기본 캐시에도 저장 (영구 저장용)
            if self.base_cache:
                try:
                    # base_cache에 context 저장 메서드가 있다면 사용
                    if hasattr(self.base_cache, 'cache_context'):
                        self.base_cache.cache_context(query, query_type, context)
                except Exception as e:
                    self.logger.debug(f"Failed to save context to base cache: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error caching context: {e}")
            return False
    
    def get_search_results(self, query: str, query_type: str,
                          search_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        검색 결과 가져오기
        
        Args:
            query: 질의
            query_type: 질의 유형
            search_params: 검색 파라미터
        
        Returns:
            캐시된 검색 결과 또는 None
        """
        cache_key = self._generate_cache_key(
            "search_results",
            query,
            query_type,
            search_params or {}
        )
        
        if cache_key in self._search_results_cache:
            cached_data, timestamp = self._search_results_cache[cache_key]
            if time.time() - timestamp < self.ttl['search_results']:
                self.stats['search_results_hits'] += 1
                self.logger.debug(f"Search results cache hit: {query[:50]}...")
                return cached_data
            else:
                del self._search_results_cache[cache_key]
        
        self.stats['search_results_misses'] += 1
        
        # 기본 캐시에서도 확인
        if self.base_cache:
            try:
                if hasattr(self.base_cache, 'get_cached_documents'):
                    cached_docs = self.base_cache.get_cached_documents(query, query_type)
                    if cached_docs:
                        result = {
                            'semantic_results': cached_docs,
                            'keyword_results': [],
                            'semantic_count': len(cached_docs),
                            'keyword_count': 0
                        }
                        self.stats['search_results_hits'] += 1
                        return result
            except Exception as e:
                self.logger.debug(f"Failed to get from base cache: {e}")
        
        return None
    
    def put_search_results(self, query: str, query_type: str,
                          search_params: Dict[str, Any],
                          results: Dict[str, Any]) -> bool:
        """
        검색 결과 저장
        
        Args:
            query: 질의
            query_type: 질의 유형
            search_params: 검색 파라미터
            results: 검색 결과 딕셔너리
        
        Returns:
            저장 성공 여부
        """
        try:
            cache_key = self._generate_cache_key(
                "search_results",
                query,
                query_type,
                search_params
            )
            
            # 캐시 크기 제한 (최대 200개)
            if len(self._search_results_cache) >= 200:
                oldest_key = min(
                    self._search_results_cache.keys(),
                    key=lambda k: self._search_results_cache[k][1]
                )
                del self._search_results_cache[oldest_key]
            
            self._search_results_cache[cache_key] = (results, time.time())
            self.logger.debug(f"Search results cached: {query[:50]}...")
            
            # 기본 캐시에도 저장
            if self.base_cache:
                try:
                    semantic_results = results.get('semantic_results', [])
                    if semantic_results and hasattr(self.base_cache, 'cache_documents'):
                        self.base_cache.cache_documents(query, query_type, semantic_results)
                except Exception as e:
                    self.logger.debug(f"Failed to save to base cache: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error caching search results: {e}")
            return False
    
    def get_query_optimization(self, query: str, query_type: str = "") -> Optional[Dict[str, Any]]:
        """
        쿼리 최적화 결과 가져오기
        
        Args:
            query: 질의
            query_type: 질의 유형
        
        Returns:
            캐시된 쿼리 최적화 결과 또는 None
        """
        cache_key = self._generate_cache_key(
            "query_optimization",
            query,
            query_type
        )
        
        if cache_key in self._query_optimization_cache:
            cached_data, timestamp = self._query_optimization_cache[cache_key]
            if time.time() - timestamp < self.ttl['query_optimization']:
                self.stats['query_optimization_hits'] += 1
                self.logger.debug(f"Query optimization cache hit: {query[:50]}...")
                return cached_data
            else:
                del self._query_optimization_cache[cache_key]
        
        self.stats['query_optimization_misses'] += 1
        return None
    
    def put_query_optimization(self, query: str, query_type: str,
                              optimization_result: Dict[str, Any]) -> bool:
        """
        쿼리 최적화 결과 저장
        
        Args:
            query: 질의
            query_type: 질의 유형
            optimization_result: 최적화 결과 딕셔너리
        
        Returns:
            저장 성공 여부
        """
        try:
            cache_key = self._generate_cache_key(
                "query_optimization",
                query,
                query_type
            )
            
            # 캐시 크기 제한 (최대 500개)
            if len(self._query_optimization_cache) >= 500:
                oldest_key = min(
                    self._query_optimization_cache.keys(),
                    key=lambda k: self._query_optimization_cache[k][1]
                )
                del self._query_optimization_cache[oldest_key]
            
            self._query_optimization_cache[cache_key] = (optimization_result, time.time())
            self.logger.debug(f"Query optimization cached: {query[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"Error caching query optimization: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환
        
        Returns:
            캐시 통계 딕셔너리
        """
        total_context_requests = self.stats['context_hits'] + self.stats['context_misses']
        total_search_requests = self.stats['search_results_hits'] + self.stats['search_results_misses']
        total_optimization_requests = (self.stats['query_optimization_hits'] + 
                                      self.stats['query_optimization_misses'])
        
        return {
            'context': {
                'hits': self.stats['context_hits'],
                'misses': self.stats['context_misses'],
                'hit_rate': (self.stats['context_hits'] / total_context_requests * 100 
                           if total_context_requests > 0 else 0.0),
                'cache_size': len(self._context_cache)
            },
            'search_results': {
                'hits': self.stats['search_results_hits'],
                'misses': self.stats['search_results_misses'],
                'hit_rate': (self.stats['search_results_hits'] / total_search_requests * 100 
                           if total_search_requests > 0 else 0.0),
                'cache_size': len(self._search_results_cache)
            },
            'query_optimization': {
                'hits': self.stats['query_optimization_hits'],
                'misses': self.stats['query_optimization_misses'],
                'hit_rate': (self.stats['query_optimization_hits'] / total_optimization_requests * 100 
                           if total_optimization_requests > 0 else 0.0),
                'cache_size': len(self._query_optimization_cache)
            }
        }
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        캐시 클리어
        
        Args:
            cache_type: 캐시 타입 ('context', 'search_results', 'query_optimization', None=전체)
        """
        if cache_type is None:
            self._context_cache.clear()
            self._search_results_cache.clear()
            self._query_optimization_cache.clear()
            self.logger.info("All caches cleared")
        elif cache_type == 'context':
            self._context_cache.clear()
            self.logger.info("Context cache cleared")
        elif cache_type == 'search_results':
            self._search_results_cache.clear()
            self.logger.info("Search results cache cleared")
        elif cache_type == 'query_optimization':
            self._query_optimization_cache.clear()
            self.logger.info("Query optimization cache cleared")

