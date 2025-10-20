# -*- coding: utf-8 -*-
"""
최적화된 검색 엔진
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from functools import lru_cache
from pathlib import Path
import hashlib

from ..data.vector_store import LegalVectorStore
from ..services.hybrid_search_engine import HybridSearchEngine
from ..services.exact_search_engine import ExactSearchEngine
from ..services.semantic_search_engine import SemanticSearchEngine

logger = logging.getLogger(__name__)

class SearchCache:
    """검색 결과 캐시"""
    
    def __init__(self, cache_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.cache_size = cache_size
        self.ttl = ttl
        self.access_times = {}
    
    def _generate_key(self, query: str, search_type: str, top_k: int) -> str:
        """캐시 키 생성"""
        key_string = f"{query}:{search_type}:{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, search_type: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """캐시에서 결과 조회"""
        key = self._generate_key(query, search_type, top_k)
        
        if key in self.cache:
            # TTL 확인
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()
                logger.debug(f"Cache hit for query: {query}")
                return self.cache[key]
            else:
                # TTL 만료
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, query: str, search_type: str, top_k: int, results: List[Dict[str, Any]]):
        """캐시에 결과 저장"""
        key = self._generate_key(query, search_type, top_k)
        
        # 캐시 크기 제한
        if len(self.cache) >= self.cache_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = results
        self.access_times[key] = time.time()
        logger.debug(f"Cached results for query: {query}")
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Search cache cleared")

class OptimizedSearchEngine:
    """최적화된 검색 엔진"""
    
    def __init__(self, 
                 vector_store: LegalVectorStore,
                 hybrid_engine: HybridSearchEngine,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600):
        self.vector_store = vector_store
        self.hybrid_engine = hybrid_engine
        self.cache = SearchCache(cache_size, cache_ttl)
        self.logger = logging.getLogger(__name__)
        
        # 성능 통계
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_search_time': 0.0,
            'total_search_time': 0.0
        }
    
    def search(self, 
               query: str, 
               search_type: str = "hybrid",
               top_k: int = 10,
               use_cache: bool = True) -> Dict[str, Any]:
        """최적화된 검색 실행"""
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        # 캐시 확인
        if use_cache:
            cached_results = self.cache.get(query, search_type, top_k)
            if cached_results is not None:
                self.stats['cache_hits'] += 1
                search_time = time.time() - start_time
                self._update_stats(search_time)
                
                return {
                    'results': cached_results,
                    'search_time': search_time,
                    'cache_hit': True,
                    'total_results': len(cached_results)
                }
        
        # 캐시 미스
        self.stats['cache_misses'] += 1
        
        # 실제 검색 실행
        if search_type == "vector":
            results = self._vector_search(query, top_k)
        elif search_type == "hybrid":
            results = self._hybrid_search(query, top_k)
        else:
            results = self._vector_search(query, top_k)
        
        # 결과 후처리
        processed_results = self._post_process_results(results, query)
        
        # 캐시에 저장
        if use_cache:
            self.cache.set(query, search_type, top_k, processed_results)
        
        search_time = time.time() - start_time
        self._update_stats(search_time)
        
        return {
            'results': processed_results,
            'search_time': search_time,
            'cache_hit': False,
            'total_results': len(processed_results)
        }
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """벡터 검색 실행"""
        try:
            results = self.vector_store.search(query, top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """하이브리드 검색 실행"""
        try:
            # 하이브리드 검색 결과를 리스트로 변환
            hybrid_results = self.hybrid_engine.search(query, max_results=top_k)
            
            # 결과 형식 통일
            if isinstance(hybrid_results, dict):
                # 딕셔너리인 경우 results 키에서 추출
                results = hybrid_results.get('results', [])
                if isinstance(results, list):
                    return results
                else:
                    return []
            elif isinstance(hybrid_results, list):
                return hybrid_results
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _post_process_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """검색 결과 후처리"""
        if not results:
            return []
        
        # 점수 정규화 및 정렬
        processed_results = []
        
        for result in results:
            if isinstance(result, dict):
                # 점수 정규화 (0-1 범위로)
                score = result.get('score', 0.0)
                if score > 1.0:
                    score = score / 100.0  # 0-100 범위를 0-1로 변환
                
                # 관련성 점수 계산
                relevance_score = self._calculate_relevance_score(result, query)
                
                processed_result = {
                    'case_id': result.get('case_id', ''),
                    'case_name': result.get('case_name', ''),
                    'case_number': result.get('case_number', ''),
                    'decision_date': result.get('decision_date', ''),
                    'court': result.get('court', ''),
                    'category': result.get('category', ''),
                    'field': result.get('field', ''),
                    'content': result.get('content', ''),
                    'score': score,
                    'relevance_score': relevance_score,
                    'metadata': result.get('metadata', {})
                }
                
                processed_results.append(processed_result)
        
        # 관련성 점수로 정렬
        processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return processed_results
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """관련성 점수 계산"""
        base_score = result.get('score', 0.0)
        
        # 쿼리 키워드와의 매칭 점수
        query_lower = query.lower()
        case_name = result.get('case_name', '').lower()
        content = result.get('content', '').lower()
        
        # 정확한 매칭
        exact_matches = 0
        for word in query_lower.split():
            if word in case_name:
                exact_matches += 1
            if word in content:
                exact_matches += 1
        
        # 부분 매칭
        partial_matches = 0
        for word in query_lower.split():
            if any(word in case_name for word in query_lower.split()):
                partial_matches += 0.5
            if any(word in content for word in query_lower.split()):
                partial_matches += 0.3
        
        # 최종 관련성 점수
        relevance_score = base_score + (exact_matches * 0.1) + (partial_matches * 0.05)
        
        return min(relevance_score, 1.0)  # 1.0으로 제한
    
    def _update_stats(self, search_time: float):
        """통계 업데이트"""
        self.stats['total_search_time'] += search_time
        self.stats['avg_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
    
    def get_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        cache_hit_rate = 0.0
        if self.stats['total_searches'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_searches']
        
        return {
            'total_searches': self.stats['total_searches'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'avg_search_time': self.stats['avg_search_time'],
            'total_search_time': self.stats['total_search_time']
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        self.logger.info("Search cache cleared")
    
    def warm_up_cache(self, common_queries: List[str]):
        """캐시 워밍업"""
        self.logger.info(f"Warming up cache with {len(common_queries)} queries")
        
        for query in common_queries:
            try:
                self.search(query, use_cache=True)
            except Exception as e:
                self.logger.warning(f"Failed to warm up cache for query '{query}': {e}")
        
        self.logger.info("Cache warm-up completed")
