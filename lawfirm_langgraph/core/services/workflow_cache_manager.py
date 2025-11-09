#!/usr/bin/env python3
"""
워크플로우 전용 캐시 관리자
prepare_search_query, execute_searches_parallel 등의 결과를 캐싱
"""

import hashlib
import time
from typing import Dict, List, Any, Optional
from lawfirm_langgraph.core.services.integrated_cache_system import (
    LRUCache, PersistentCache, IntegratedCacheSystem
)


class WorkflowCacheManager:
    """워크플로우 전용 캐시 관리자"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # L1: 메모리 캐시 (빠른 접근)
        self.l1_cache = LRUCache(
            max_size=self.config.get('l1_cache_size', 1000)
        )
        
        # L2: 영구 캐시 (디스크 기반)
        self.l2_cache = PersistentCache(
            cache_dir=self.config.get('cache_dir', 'cache/workflow'),
            max_size=self.config.get('l2_cache_size', 5000)
        )
        
        # 통합 캐시 시스템 (기존 시스템 활용)
        self.integrated_cache = IntegratedCacheSystem(config)
        
        # 성능 통계
        self.stats = {
            'query_preparation_hits': 0,
            'query_preparation_misses': 0,
            'search_results_hits': 0,
            'search_results_misses': 0,
            'query_optimization_hits': 0,
            'query_optimization_misses': 0,
        }
    
    def _generate_key(self, prefix: str, *args) -> str:
        """캐시 키 생성"""
        key_string = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_query_preparation(self, query: str, query_type: str = "") -> Optional[Dict[str, Any]]:
        """쿼리 준비 결과 가져오기"""
        cache_key = self._generate_key("query_prep", query, query_type)
        
        # L1 확인
        result = self.l1_cache.get(cache_key)
        if result:
            self.stats['query_preparation_hits'] += 1
            return result
        
        # L2 확인
        result = self.l2_cache.get(cache_key)
        if result:
            self.l1_cache.put(cache_key, result, ttl=1800)
            self.stats['query_preparation_hits'] += 1
            return result
        
        self.stats['query_preparation_misses'] += 1
        return None
    
    def put_query_preparation(self, query: str, query_type: str, 
                            result: Dict[str, Any], ttl: float = 3600.0):
        """쿼리 준비 결과 저장"""
        cache_key = self._generate_key("query_prep", query, query_type)
        
        # L1과 L2 모두에 저장
        self.l1_cache.put(cache_key, result, ttl=min(ttl, 1800))
        self.l2_cache.put(cache_key, result, ttl=ttl)
    
    def get_query_optimization(self, query: str, query_type: str, 
                              extracted_keywords: List[str] = None) -> Optional[Dict[str, Any]]:
        """쿼리 최적화 결과 가져오기"""
        keywords_str = ",".join(sorted(extracted_keywords or []))
        cache_key = self._generate_key("query_opt", query, query_type, keywords_str)
        
        # L1 확인
        result = self.l1_cache.get(cache_key)
        if result:
            self.stats['query_optimization_hits'] += 1
            return result
        
        # L2 확인
        result = self.l2_cache.get(cache_key)
        if result:
            self.l1_cache.put(cache_key, result, ttl=1800)
            self.stats['query_optimization_hits'] += 1
            return result
        
        self.stats['query_optimization_misses'] += 1
        return None
    
    def put_query_optimization(self, query: str, query_type: str, 
                              extracted_keywords: List[str], 
                              result: Dict[str, Any], ttl: float = 3600.0):
        """쿼리 최적화 결과 저장"""
        keywords_str = ",".join(sorted(extracted_keywords or []))
        cache_key = self._generate_key("query_opt", query, query_type, keywords_str)
        
        # L1과 L2 모두에 저장
        self.l1_cache.put(cache_key, result, ttl=min(ttl, 1800))
        self.l2_cache.put(cache_key, result, ttl=ttl)
    
    def get_search_results(self, query: str, query_type: str, 
                          search_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """검색 결과 가져오기"""
        params_str = str(sorted((search_params or {}).items()))
        cache_key = self._generate_key("search_results", query, query_type, params_str)
        
        # L1 확인
        result = self.l1_cache.get(cache_key)
        if result:
            self.stats['search_results_hits'] += 1
            return result
        
        # L2 확인
        result = self.l2_cache.get(cache_key)
        if result:
            self.l1_cache.put(cache_key, result, ttl=1800)
            self.stats['search_results_hits'] += 1
            return result
        
        self.stats['search_results_misses'] += 1
        return None
    
    def put_search_results(self, query: str, query_type: str, 
                          search_params: Dict[str, Any],
                          results: Dict[str, Any], ttl: float = 7200.0):
        """검색 결과 저장"""
        params_str = str(sorted((search_params or {}).items()))
        cache_key = self._generate_key("search_results", query, query_type, params_str)
        
        # L1과 L2 모두에 저장
        self.l1_cache.put(cache_key, results, ttl=min(ttl, 1800))
        self.l2_cache.put(cache_key, results, ttl=ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """전체 캐시 통계 반환"""
        total_query_prep = (self.stats['query_preparation_hits'] + 
                          self.stats['query_preparation_misses'])
        total_query_opt = (self.stats['query_optimization_hits'] + 
                          self.stats['query_optimization_misses'])
        total_search = (self.stats['search_results_hits'] + 
                       self.stats['search_results_misses'])
        
        return {
            'query_preparation': {
                'hit_rate': (self.stats['query_preparation_hits'] / total_query_prep 
                           if total_query_prep > 0 else 0),
                'hits': self.stats['query_preparation_hits'],
                'misses': self.stats['query_preparation_misses'],
                'l1_stats': self.l1_cache.get_stats()
            },
            'query_optimization': {
                'hit_rate': (self.stats['query_optimization_hits'] / total_query_opt 
                           if total_query_opt > 0 else 0),
                'hits': self.stats['query_optimization_hits'],
                'misses': self.stats['query_optimization_misses']
            },
            'search_results': {
                'hit_rate': (self.stats['search_results_hits'] / total_search 
                           if total_search > 0 else 0),
                'hits': self.stats['search_results_hits'],
                'misses': self.stats['search_results_misses']
            },
            'l2_stats': {
                'size': len(self.l2_cache.index)
            }
        }
    
    def clear_all(self):
        """모든 캐시 정리"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        
        # 통계 초기화
        self.stats = {
            'query_preparation_hits': 0,
            'query_preparation_misses': 0,
            'search_results_hits': 0,
            'search_results_misses': 0,
            'query_optimization_hits': 0,
            'query_optimization_misses': 0,
        }
    
    def get_hit_rate(self) -> float:
        """전체 캐시 히트율 반환"""
        total_hits = (self.stats['query_preparation_hits'] + 
                     self.stats['query_optimization_hits'] + 
                     self.stats['search_results_hits'])
        total_misses = (self.stats['query_preparation_misses'] + 
                       self.stats['query_optimization_misses'] + 
                       self.stats['search_results_misses'])
        total = total_hits + total_misses
        
        return total_hits / total if total > 0 else 0.0

