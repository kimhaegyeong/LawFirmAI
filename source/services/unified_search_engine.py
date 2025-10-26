# -*- coding: utf-8 -*-
"""
Unified Search Engine
모든 검색 기능을 통합한 단일 검색 엔진
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..data.vector_store import LegalVectorStore
from .search.precedent_search_engine import PrecedentSearchEngine
from .search.semantic_search_engine import SemanticSearchEngine

logger = logging.getLogger(__name__)


class SimpleCache:
    """간단한 캐시 구현"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def clear(self):
        self.cache.clear()


@dataclass
class UnifiedSearchResult:
    """통합 검색 결과"""
    query: str
    results: List[Dict[str, Any]]
    search_time: float
    search_types_used: List[str]
    total_results: int
    confidence: float
    cache_hit: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UnifiedSearchEngine:
    """통합 검색 엔진 클래스"""

    def __init__(self,
                 vector_store: LegalVectorStore,
                 exact_search_engine: Optional[Any] = None,
                 semantic_search_engine: Optional[SemanticSearchEngine] = None,
                 precedent_search_engine: Optional[PrecedentSearchEngine] = None,
                 current_law_search_engine: Optional[Any] = None,
                 enable_caching: bool = True):
        """
        통합 검색 엔진 초기화

        Args:
            vector_store: 벡터 스토어
            exact_search_engine: 정확 검색 엔진
            semantic_search_engine: 의미 검색 엔진
            precedent_search_engine: 판례 검색 엔진
            enable_caching: 캐싱 활성화
        """
        self.vector_store = vector_store
        self.exact_search_engine = exact_search_engine
        self.semantic_search_engine = semantic_search_engine
        self.precedent_search_engine = precedent_search_engine
        self.current_law_search_engine = current_law_search_engine

        # 캐시 매니저
        self.cache_manager = SimpleCache() if enable_caching else None

        # 성능 통계
        self._stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }

        logger.info("UnifiedSearchEngine initialized successfully")

    async def search(self,
                    query: str,
                    top_k: int = 10,
                    search_types: List[str] = None,
                    category: str = 'all',
                    use_cache: bool = True) -> UnifiedSearchResult:
        """
        통합 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
            search_types: 검색 타입 리스트
            category: 검색 카테고리
            use_cache: 캐시 사용 여부

        Returns:
            UnifiedSearchResult: 통합 검색 결과
        """
        start_time = time.time()

        # 기본 검색 타입 설정
        if search_types is None:
            search_types = ['vector', 'exact', 'semantic', 'precedent']

        # 캐시 확인
        if use_cache and self.cache_manager:
            cache_key = f"{query}:{category}:{top_k}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                search_time = time.time() - start_time
                self._update_stats(search_time, cache_hit=True)

                return UnifiedSearchResult(
                    query=query,
                    results=cached_result,
                    search_time=search_time,
                    search_types_used=['cached'],
                    total_results=len(cached_result),
                    confidence=0.9,
                    cache_hit=True
                )

        # 검색 실행
        all_results = []
        used_types = []

        # 벡터 검색
        if 'vector' in search_types and self.vector_store:
            try:
                vector_results = self._search_vector(query, top_k, category)
                all_results.extend(vector_results)
                used_types.append('vector')
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")

        # 정확 검색
        if 'exact' in search_types and self.exact_search_engine:
            try:
                exact_results = self._search_exact(query, top_k, category)
                all_results.extend(exact_results)
                used_types.append('exact')
            except Exception as e:
                logger.debug(f"Exact search failed: {e}")

        # 현행법령 검색
        if 'current_law' in search_types and hasattr(self, 'current_law_search_engine') and self.current_law_search_engine:
            try:
                current_law_results = self.current_law_search_engine.search_current_laws(
                    query, top_k=top_k
                )
                all_results.extend(self._format_current_law_results(current_law_results))
                used_types.append('current_law')
            except Exception as e:
                logger.debug(f"Current law search failed: {e}")

        # 의미 검색
        if 'semantic' in search_types and self.semantic_search_engine:
            try:
                semantic_results = self._search_semantic(query, top_k, category)
                all_results.extend(semantic_results)
                used_types.append('semantic')
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # 판례 검색
        if 'precedent' in search_types and self.precedent_search_engine:
            try:
                precedent_results = self._search_precedent(query, top_k, category)
                all_results.extend(precedent_results)
                used_types.append('precedent')
            except Exception as e:
                logger.debug(f"Precedent search failed: {e}")

        # 결과 통합 및 중복 제거
        integrated_results = self._integrate_and_deduplicate_results(all_results, top_k)

        search_time = time.time() - start_time
        self._update_stats(search_time, cache_hit=False)

        # 캐시 저장
        if use_cache and self.cache_manager:
            cache_key = f"{query}:{category}:{top_k}"
            self.cache_manager.set(cache_key, integrated_results)

        return UnifiedSearchResult(
            query=query,
            results=integrated_results,
            search_time=search_time,
            search_types_used=used_types,
            total_results=len(integrated_results),
            confidence=self._calculate_confidence(integrated_results)
        )

    def _search_vector(self, query: str, top_k: int, category: str) -> List[Dict[str, Any]]:
        """벡터 검색 - 개선된 버전"""
        try:
            # 벡터 스토어가 비어있는 경우 인덱스 로드 시도
            if not hasattr(self.vector_store, 'index') or self.vector_store.index is None:
                logger.debug("Vector index is empty, attempting to load...")
                try:
                    self.vector_store.load_index()
                    logger.info("Vector index loaded successfully")
                except Exception as load_error:
                    logger.debug(f"Failed to load vector index: {load_error}")
                    return []

            # 성능 최적화: 확장 검색 범위 축소
            expanded_k = min(top_k * 2, 20)  # 최대 20개까지 확장 검색 (기존 50개에서 축소)
            results = self.vector_store.search(query, top_k=expanded_k)

            if not results:
                logger.debug("No vector search results found")
                return []

            # 결과 필터링 및 포맷팅
            formatted_results = []
            for result in results:
                formatted_result = self._format_vector_result(result)
                # 임계값을 낮춰서 더 많은 결과 포함
                if formatted_result.get('score', 0.0) >= 0.3:  # 기존 0.5에서 0.3으로 낮춤
                    formatted_results.append(formatted_result)

            # 상위 k개만 반환
            formatted_results = formatted_results[:top_k]

            logger.debug(f"Vector search found {len(formatted_results)} results (from {len(results)} total)")
            return formatted_results
        except Exception as e:
            logger.debug(f"Vector search error: {e}")
            return []

    def _format_current_law_results(self, results) -> List[Dict[str, Any]]:
        """현행법령 검색 결과 포맷팅"""
        formatted_results = []
        for result in results:
            formatted_result = {
                'content': result.matched_content,
                'metadata': {
                    'law_id': result.law_id,
                    'law_name_korean': result.law_name_korean,
                    'law_name_abbreviation': result.law_name_abbreviation,
                    'promulgation_date': result.promulgation_date,
                    'promulgation_number': result.promulgation_number,
                    'amendment_type': result.amendment_type,
                    'ministry_name': result.ministry_name,
                    'law_type': result.law_type,
                    'effective_date': result.effective_date,
                    'law_detail_link': result.law_detail_link,
                    'document_type': 'current_law',
                    'search_type': result.search_type,
                    'article_content': getattr(result, 'article_content', None)
                },
                'score': result.similarity_score,
                'similarity': result.similarity_score
            }
            formatted_results.append(formatted_result)
        return formatted_results

    def _search_exact(self, query: str, top_k: int, category: str) -> List[Dict[str, Any]]:
        """정확 검색 - FTS 직접 사용"""
        try:
            # FTS 직접 검색으로 개선
            results = self._search_fts_direct(query, top_k)
            return results
        except Exception as e:
            logger.error(f"Exact search error: {e}")
            return []

    def _search_fts_direct(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """FTS 직접 검색"""
        try:
            import sqlite3

            # 데이터베이스 연결
            conn = sqlite3.connect('data/lawfirm.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            results = []

            # 1. current_laws_articles에서 FTS 검색
            try:
                cursor.execute("""
                    SELECT article_id, article_title, article_content,
                           snippet(fts_current_laws_articles, 2, '<b>', '</b>', '...', 32) as snippet
                    FROM fts_current_laws_articles
                    WHERE fts_current_laws_articles MATCH ?
                    LIMIT ?
                """, (query, top_k))

                rows = cursor.fetchall()
                for row in rows:
                    result = {
                        'content': row['article_content'],
                        'score': 0.9,  # FTS 검색은 높은 신뢰도
                        'source': 'fts_current_laws_articles',
                        'type': 'exact',
                        'metadata': {
                            'id': row['article_id'],
                            'title': row['article_title'],
                            'snippet': row['snippet']
                        }
                    }
                    results.append(result)

            except Exception as e:
                logger.debug(f"FTS current_laws_articles search error: {e}")

            # 2. assembly_laws에서 FTS 검색
            try:
                cursor.execute("""
                    SELECT id, law_name, full_text,
                           snippet(fts_assembly_laws, 2, '<b>', '</b>', '...', 32) as snippet
                    FROM fts_assembly_laws
                    WHERE fts_assembly_laws MATCH ?
                    LIMIT ?
                """, (query, top_k))

                rows = cursor.fetchall()
                for row in rows:
                    result = {
                        'content': row['full_text'],
                        'score': 0.8,  # 법률명 검색은 중간 신뢰도
                        'source': 'fts_assembly_laws',
                        'type': 'exact',
                        'metadata': {
                            'id': row['id'],
                            'law_name': row['law_name'],
                            'snippet': row['snippet']
                        }
                    }
                    results.append(result)

            except Exception as e:
                logger.debug(f"FTS assembly_laws search error: {e}")

            conn.close()

            # 상위 k개만 반환
            results = results[:top_k]
            logger.debug(f"FTS direct search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"FTS direct search error: {e}")
            return []

    def _search_semantic(self, query: str, top_k: int, category: str) -> List[Dict[str, Any]]:
        """의미 검색"""
        try:
            # SemanticSearchEngine은 k 매개변수 사용
            results = self.semantic_search_engine.search(query, k=top_k)
            return [self._format_semantic_result(result) for result in results]
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def _search_precedent(self, query: str, top_k: int, category: str) -> List[Dict[str, Any]]:
        """판례 검색"""
        try:
            precedent_category = 'civil' if category == 'all' else category
            results = self.precedent_search_engine.search_precedents(
                query, category=precedent_category, top_k=top_k
            )
            return [self._format_precedent_result(result) for result in results]
        except Exception as e:
            logger.error(f"Precedent search error: {e}")
            return []

    def _format_vector_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """벡터 검색 결과 포맷팅"""
        return {
            'content': result.get('text', result.get('content', '')),
            'score': result.get('score', 0.0),
            'source': result.get('source', 'vector'),
            'type': 'vector',
            'metadata': result.get('metadata', {})
        }

    def _format_exact_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """정확 검색 결과 포맷팅"""
        return {
            'content': result.get('content', ''),
            'score': result.get('score', 0.0),
            'source': result.get('source', 'exact'),
            'type': 'exact',
            'metadata': result.get('metadata', {})
        }

    def _format_semantic_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """의미 검색 결과 포맷팅"""
        return {
            'content': result.get('content', ''),
            'score': result.get('score', 0.0),
            'source': result.get('source', 'semantic'),
            'type': 'semantic',
            'metadata': result.get('metadata', {})
        }

    def _format_precedent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """판례 검색 결과 포맷팅"""
        return {
            'content': result.get('content', ''),
            'score': result.get('score', 0.0),
            'source': result.get('source', 'precedent'),
            'type': 'precedent',
            'metadata': result.get('metadata', {})
        }

    def _integrate_and_deduplicate_results(self, all_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """결과 통합 및 중복 제거"""
        try:
            # 중복 제거 (내용 기반)
            seen_contents = set()
            unique_results = []

            for result in all_results:
                content_hash = hash(result.get('content', ''))
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append(result)

            # 점수 기반 정렬
            sorted_results = sorted(unique_results, key=lambda x: x.get('score', 0.0), reverse=True)

            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return all_results[:top_k]

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """신뢰도 계산"""
        if not results:
            return 0.0

        # 평균 점수 기반 신뢰도 계산
        avg_score = sum(result.get('score', 0.0) for result in results) / len(results)
        return min(avg_score, 1.0)

    def _update_stats(self, search_time: float, cache_hit: bool = False):
        """통계 업데이트"""
        self._stats['total_searches'] += 1
        self._stats['total_search_time'] += search_time
        self._stats['avg_search_time'] = self._stats['total_search_time'] / self._stats['total_searches']

        if cache_hit:
            self._stats['cache_hits'] += 1
        else:
            self._stats['cache_misses'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return self._stats.copy()

    def clear_cache(self):
        """캐시 클리어"""
        if self.cache_manager:
            self.cache_manager.clear()
        logger.info("Search cache cleared")
