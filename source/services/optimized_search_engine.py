# -*- coding: utf-8 -*-
"""
Optimized Search Engine
최적화된 검색 엔진 모듈
"""

import logging
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import multiprocessing as mp

from source.services.cache_manager import QueryCache, get_cache_manager
from source.data.vector_store import LegalVectorStore
from source.services.exact_search_engine import ExactSearchEngine
from source.services.semantic_search_engine import SemanticSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    query: str
    results: List[Dict[str, Any]]
    search_time: float
    cache_hit: bool
    search_type: str
    total_results: int
    confidence: float


class OptimizedSearchEngine:
    """최적화된 검색 엔진 클래스"""
    
    def __init__(self, 
                 vector_store: LegalVectorStore,
                 exact_search_engine: ExactSearchEngine,
                 semantic_search_engine: SemanticSearchEngine,
                 max_workers: int = None,
                 enable_parallel_search: bool = True,
                 enable_caching: bool = True):
        """
        최적화된 검색 엔진 초기화
        
        Args:
            vector_store: 벡터 스토어
            exact_search_engine: 정확 검색 엔진
            semantic_search_engine: 의미 검색 엔진
            max_workers: 최대 워커 수
            enable_parallel_search: 병렬 검색 활성화
            enable_caching: 캐싱 활성화
        """
        self.vector_store = vector_store
        self.exact_search_engine = exact_search_engine
        self.semantic_search_engine = semantic_search_engine
        
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.enable_parallel_search = enable_parallel_search
        self.enable_caching = enable_caching
        
        # 캐시 매니저
        self.cache_manager = get_cache_manager() if enable_caching else None
        self.query_cache = QueryCache(self.cache_manager) if enable_caching else None
        
        # 스레드 풀
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 성능 통계
        self._stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_searches': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }
        
        logger.info(f"OptimizedSearchEngine initialized: workers={self.max_workers}, "
                   f"parallel={enable_parallel_search}, caching={enable_caching}")
    
    async def search(self, 
                    query: str, 
                    top_k: int = 10,
                    search_types: List[str] = None,
                    filters: Dict = None,
                    use_cache: bool = True) -> SearchResult:
        """
        최적화된 검색 수행
        
        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
            search_types: 검색 타입 리스트
            filters: 필터 조건
            use_cache: 캐시 사용 여부
            
        Returns:
            SearchResult: 검색 결과
        """
        start_time = time.time()
        
        # 기본 검색 타입 설정
        if search_types is None:
            search_types = ['vector', 'exact', 'semantic']
        
        # 캐시 확인
        if use_cache and self.query_cache:
            cached_result = self.query_cache.get_search_result(query, filters, top_k)
            if cached_result is not None:
                search_time = time.time() - start_time
                self._update_stats(search_time, cache_hit=True)
                
                return SearchResult(
                    query=query,
                    results=cached_result,
                    search_time=search_time,
                    cache_hit=True,
                    search_type='cached',
                    total_results=len(cached_result),
                    confidence=0.9
                )
        
        # 병렬 검색 수행
        if self.enable_parallel_search and len(search_types) > 1:
            results = await self._parallel_search(query, top_k, search_types, filters)
        else:
            results = await self._sequential_search(query, top_k, search_types, filters)
        
        # 결과 통합 및 정렬
        integrated_results = self._integrate_results(results, top_k)
        
        search_time = time.time() - start_time
        self._update_stats(search_time, cache_hit=False)
        
        # 캐시 저장
        if use_cache and self.query_cache:
            self.query_cache.set_search_result(query, integrated_results, filters, top_k)
        
        return SearchResult(
            query=query,
            results=integrated_results,
            search_time=search_time,
            cache_hit=False,
            search_type='integrated',
            total_results=len(integrated_results),
            confidence=self._calculate_confidence(integrated_results)
        )
    
    async def _parallel_search(self, 
                              query: str, 
                              top_k: int, 
                              search_types: List[str],
                              filters: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """병렬 검색 수행"""
        self._stats['parallel_searches'] += 1
        
        # 검색 태스크 생성
        tasks = []
        
        if 'vector' in search_types:
            task = asyncio.create_task(
                self._run_in_executor(self._vector_search, query, top_k, filters)
            )
            tasks.append(('vector', task))
        
        if 'exact' in search_types:
            task = asyncio.create_task(
                self._run_in_executor(self._exact_search, query, top_k, filters)
            )
            tasks.append(('exact', task))
        
        if 'semantic' in search_types:
            task = asyncio.create_task(
                self._run_in_executor(self._semantic_search, query, top_k, filters)
            )
            tasks.append(('semantic', task))
        
        # 모든 태스크 완료 대기
        results = {}
        for search_type, task in tasks:
            try:
                result = await task
                results[search_type] = result
            except Exception as e:
                logger.error(f"Parallel search failed for {search_type}: {e}")
                results[search_type] = []
        
        return results
    
    async def _sequential_search(self, 
                               query: str, 
                               top_k: int, 
                               search_types: List[str],
                               filters: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """순차 검색 수행"""
        results = {}
        
        for search_type in search_types:
            try:
                if search_type == 'vector':
                    result = await self._run_in_executor(self._vector_search, query, top_k, filters)
                elif search_type == 'exact':
                    result = await self._run_in_executor(self._exact_search, query, top_k, filters)
                elif search_type == 'semantic':
                    result = await self._run_in_executor(self._semantic_search, query, top_k, filters)
                else:
                    result = []
                
                results[search_type] = result
                
            except Exception as e:
                logger.error(f"Sequential search failed for {search_type}: {e}")
                results[search_type] = []
        
        return results
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """스레드 풀에서 함수 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def _vector_search(self, query: str, top_k: int, filters: Dict) -> List[Dict[str, Any]]:
        """벡터 검색 수행"""
        try:
            if self.query_cache:
                cached_embedding = self.query_cache.get_embedding(query)
                if cached_embedding is not None:
                    # 캐시된 임베딩 사용
                    pass  # 벡터 스토어에서 직접 검색
            
            results = self.vector_store.search(query, top_k, filters, enhanced=True)
            
            # 임베딩 캐싱
            if self.query_cache and results:
                # 임베딩은 벡터 스토어 내부에서 생성되므로 여기서는 결과만 캐싱
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _exact_search(self, query: str, top_k: int, filters: Dict) -> List[Dict[str, Any]]:
        """정확 검색 수행"""
        try:
            # ExactSearchEngine은 documents 매개변수가 필요하므로 빈 리스트 전달
            results = self.exact_search_engine.search(query, documents=[], top_k=top_k)
            
            # 결과 형식 통일
            formatted_results = []
            for result in results:
                formatted_result = {
                    'score': getattr(result, 'score', 0.8),
                    'text': getattr(result, 'text', ''),
                    'metadata': getattr(result, 'metadata', {}),
                    'search_type': 'exact'
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
            return []
    
    def _semantic_search(self, query: str, top_k: int, filters: Dict) -> List[Dict[str, Any]]:
        """의미 검색 수행"""
        try:
            # SemanticSearchEngine은 k 매개변수 사용
            results = self.semantic_search_engine.search(query, k=top_k)
            
            # 결과 형식 통일
            formatted_results = []
            for result in results:
                formatted_result = {
                    'score': result.get('score', 0.7),
                    'text': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'search_type': 'semantic'
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _integrate_results(self, 
                          results: Dict[str, List[Dict[str, Any]]], 
                          top_k: int) -> List[Dict[str, Any]]:
        """검색 결과 통합 및 정렬"""
        try:
            # 모든 결과 수집
            all_results = []
            
            for search_type, search_results in results.items():
                for result in search_results:
                    result['search_type'] = search_type
                    all_results.append(result)
            
            # 중복 제거 (텍스트 기준)
            unique_results = {}
            for result in all_results:
                text_key = result.get('text', '')[:100]  # 처음 100자로 중복 판단
                if text_key not in unique_results:
                    unique_results[text_key] = result
                else:
                    # 더 높은 점수로 업데이트
                    if result.get('score', 0) > unique_results[text_key].get('score', 0):
                        unique_results[text_key] = result
            
            # 점수 기준 정렬
            sorted_results = sorted(
                unique_results.values(),
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            
            return sorted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return []
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """검색 결과 신뢰도 계산"""
        if not results:
            return 0.0
        
        try:
            # 상위 결과들의 평균 점수
            top_scores = [result.get('score', 0) for result in results[:3]]
            avg_score = sum(top_scores) / len(top_scores)
            
            # 결과 수에 따른 보정
            count_factor = min(len(results) / 5, 1.0)
            
            confidence = avg_score * count_factor
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _update_stats(self, search_time: float, cache_hit: bool):
        """통계 업데이트"""
        self._stats['total_searches'] += 1
        self._stats['total_search_time'] += search_time
        self._stats['avg_search_time'] = (
            self._stats['total_search_time'] / self._stats['total_searches']
        )
        
        if cache_hit:
            self._stats['cache_hits'] += 1
        else:
            self._stats['cache_misses'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self._stats.copy()
        
        # 캐시 통계 추가
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats['cache_stats'] = cache_stats
        
        return stats
    
    def clear_cache(self):
        """캐시 정리"""
        if self.cache_manager:
            self.cache_manager.clear()
    
    def cleanup_expired_cache(self):
        """만료된 캐시 정리"""
        if self.cache_manager:
            self.cache_manager.cleanup_expired()
    
    def shutdown(self):
        """리소스 정리"""
        self.executor.shutdown(wait=True)
        logger.info("OptimizedSearchEngine shutdown completed")