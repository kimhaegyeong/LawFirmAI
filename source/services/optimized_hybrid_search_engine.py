#!/usr/bin/env python3
"""
최적화된 하이브리드 검색 엔진
병렬 처리, 캐싱, 결과 제한을 통한 성능 최적화
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from collections import defaultdict

# 프로젝트 모듈 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.services.exact_search_engine import ExactSearchEngine
from source.services.semantic_search_engine import SemanticSearchEngine
from source.services.precedent_search_engine import PrecedentSearchEngine
from source.services.answer_structure_enhancer import QuestionType

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any]
    search_method: str

@dataclass
class OptimizedSearchConfig:
    """최적화된 검색 설정"""
    max_results_per_type: int = 5  # 각 검색 타입별 최대 결과 수
    parallel_search: bool = True    # 병렬 검색 사용
    cache_enabled: bool = True     # 캐싱 사용
    timeout_seconds: float = 3.0   # 검색 타임아웃
    min_score_threshold: float = 0.3  # 최소 점수 임계값

class SearchCache:
    """검색 결과 캐시"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, query: str, question_type: str, max_results: int) -> str:
        """캐시 키 생성"""
        key_string = f"{query.lower().strip()}:{question_type}:{max_results}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, question_type: str, max_results: int) -> Optional[List[SearchResult]]:
        """캐시에서 결과 가져오기"""
        with self.lock:
            key = self._generate_key(query, question_type, max_results)
            if key in self.cache:
                results, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return results
                else:
                    del self.cache[key]
            return None
    
    def put(self, query: str, question_type: str, max_results: int, results: List[SearchResult]):
        """캐시에 결과 저장"""
        with self.lock:
            key = self._generate_key(query, question_type, max_results)
            
            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (results, time.time())

class OptimizedHybridSearchEngine:
    """최적화된 하이브리드 검색 엔진"""
    
    def __init__(self, config: Optional[OptimizedSearchConfig] = None):
        self.config = config or OptimizedSearchConfig()
        
        # 검색 엔진 초기화
        self.exact_search = ExactSearchEngine()
        self.semantic_search = SemanticSearchEngine()
        self.precedent_search = PrecedentSearchEngine()
        
        # 캐시 초기화
        self.cache = SearchCache() if self.config.cache_enabled else None
        
        # 스레드 풀 초기화
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 성능 통계
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'parallel_searches': 0,
            'avg_search_time': 0.0,
            'total_search_time': 0.0
        }
    
    async def search_with_question_type(self, query: str, question_type: QuestionType, 
                                      max_results: int = 20) -> List[SearchResult]:
        """질문 유형에 따른 최적화된 검색"""
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        # 캐시 확인
        if self.cache:
            cached_results = self.cache.get(query, question_type.value, max_results)
            if cached_results:
                self.stats['cache_hits'] += 1
                return cached_results
        
        # 병렬 검색 실행
        if self.config.parallel_search:
            results = await self._parallel_search(query, question_type, max_results)
        else:
            results = await self._sequential_search(query, question_type, max_results)
        
        # 결과 필터링 및 정렬
        filtered_results = self._filter_and_rank_results(results, max_results)
        
        # 캐시 저장
        if self.cache:
            self.cache.put(query, question_type.value, max_results, filtered_results)
        
        # 통계 업데이트
        search_time = time.time() - start_time
        self.stats['total_search_time'] += search_time
        self.stats['avg_search_time'] = (
            self.stats['total_search_time'] / self.stats['total_searches']
        )
        
        return filtered_results
    
    async def _parallel_search(self, query: str, question_type: QuestionType, 
                             max_results: int) -> List[SearchResult]:
        """병렬 검색 실행"""
        self.stats['parallel_searches'] += 1
        
        # 검색 작업 정의
        search_tasks = []
        
        # 정확 검색
        if question_type.law_weight > 0:
            search_tasks.append(
                self._run_search_task(
                    self._exact_search_laws, 
                    query, 
                    question_type.law_weight,
                    max_results // 3
                )
            )
        
        # 의미 검색
        if question_type.precedent_weight > 0:
            search_tasks.append(
                self._run_search_task(
                    self._semantic_search_precedents,
                    query,
                    question_type.precedent_weight,
                    max_results // 3
                )
            )
        
        # 판례 검색
        if question_type.precedent_weight > 0:
            search_tasks.append(
                self._run_search_task(
                    self._precedent_search,
                    query,
                    question_type.precedent_weight,
                    max_results // 3
                )
            )
        
        # 모든 검색 작업 병렬 실행
        all_results = []
        if search_tasks:
            results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            for results in results_list:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    print(f"Search task failed: {results}")
        
        return all_results
    
    async def _sequential_search(self, query: str, question_type: QuestionType, 
                               max_results: int) -> List[SearchResult]:
        """순차 검색 실행"""
        all_results = []
        
        # 정확 검색
        if question_type.law_weight > 0:
            try:
                law_results = await self._exact_search_laws(query, question_type.law_weight, max_results // 3)
                all_results.extend(law_results)
            except Exception as e:
                print(f"Exact search failed: {e}")
        
        # 의미 검색
        if question_type.precedent_weight > 0:
            try:
                semantic_results = await self._semantic_search_precedents(query, question_type.precedent_weight, max_results // 3)
                all_results.extend(semantic_results)
            except Exception as e:
                print(f"Semantic search failed: {e}")
        
        # 판례 검색
        if question_type.precedent_weight > 0:
            try:
                precedent_results = await self._precedent_search(query, question_type.precedent_weight, max_results // 3)
                all_results.extend(precedent_results)
            except Exception as e:
                print(f"Precedent search failed: {e}")
        
        return all_results
    
    async def _run_search_task(self, search_func, query: str, weight: float, max_results: int) -> List[SearchResult]:
        """검색 작업 실행 (타임아웃 적용)"""
        try:
            return await asyncio.wait_for(
                search_func(query, weight, max_results),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"Search task timed out: {search_func.__name__}")
            return []
        except Exception as e:
            print(f"Search task failed: {search_func.__name__} - {e}")
            return []
    
    async def _exact_search_laws(self, query: str, weight: float, max_results: int) -> List[SearchResult]:
        """정확 검색 실행"""
        try:
            results = self.exact_search.search(query, max_results=max_results)
            return [
                SearchResult(
                    content=result.get('content', ''),
                    score=result.get('score', 0.0) * weight,
                    source_type='law',
                    metadata=result.get('metadata', {}),
                    search_method='exact'
                )
                for result in results
            ]
        except Exception as e:
            print(f"Exact search error: {e}")
            return []
    
    async def _semantic_search_precedents(self, query: str, weight: float, max_results: int) -> List[SearchResult]:
        """의미 검색 실행"""
        try:
            results = self.semantic_search.search(query, max_results=max_results)
            return [
                SearchResult(
                    content=result.get('content', ''),
                    score=result.get('score', 0.0) * weight,
                    source_type='precedent',
                    metadata=result.get('metadata', {}),
                    search_method='semantic'
                )
                for result in results
            ]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    async def _precedent_search(self, query: str, weight: float, max_results: int) -> List[SearchResult]:
        """판례 검색 실행"""
        try:
            results = self.precedent_search.search_precedents(query, max_results=max_results)
            return [
                SearchResult(
                    content=result.get('content', ''),
                    score=result.get('score', 0.0) * weight,
                    source_type='precedent',
                    metadata=result.get('metadata', {}),
                    search_method='precedent'
                )
                for result in results
            ]
        except Exception as e:
            print(f"Precedent search error: {e}")
            return []
    
    def _filter_and_rank_results(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """결과 필터링 및 랭킹"""
        # 점수 임계값 필터링
        filtered_results = [
            result for result in results 
            if result.score >= self.config.min_score_threshold
        ]
        
        # 점수 기준 정렬
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # 결과 수 제한
        return filtered_results[:max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['total_searches']
            if self.stats['total_searches'] > 0 else 0
        )
        
        parallel_rate = (
            self.stats['parallel_searches'] / self.stats['total_searches']
            if self.stats['total_searches'] > 0 else 0
        )
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'parallel_search_rate': parallel_rate,
            'cache_size': len(self.cache.cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """캐시 정리"""
        if self.cache:
            self.cache.cache.clear()
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
