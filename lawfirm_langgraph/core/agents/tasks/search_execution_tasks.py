# -*- coding: utf-8 -*-
"""
검색 실행 병렬 Task 모듈
의미 검색과 키워드 검색을 병렬로 실행하는 Task
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchExecutionTasks:
    """검색 실행 병렬 Task 클래스"""

    @staticmethod
    async def execute_searches_parallel(
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        query_type_str: str,
        legal_field: str,
        extracted_keywords: List[str],
        original_query: str,
        execute_semantic_search_func: Optional[Callable] = None,
        execute_keyword_search_func: Optional[Callable] = None,
        timeout: float = 30.0
    ) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]], int]:
        """
        의미 검색과 키워드 검색을 병렬로 실행
        
        Args:
            optimized_queries: 최적화된 쿼리 딕셔너리
            search_params: 검색 파라미터
            query_type_str: 질의 유형 문자열
            legal_field: 법률 분야
            extracted_keywords: 추출된 키워드
            original_query: 원본 질의
            execute_semantic_search_func: 의미 검색 실행 함수
            execute_keyword_search_func: 키워드 검색 실행 함수
            timeout: 타임아웃 (초)
        
        Returns:
            (semantic_results, semantic_count, keyword_results, keyword_count)
        """
        semantic_results = []
        semantic_count = 0
        keyword_results = []
        keyword_count = 0

        async def run_semantic_search():
            """의미 검색 실행 (비동기 래퍼)"""
            if not execute_semantic_search_func:
                return [], 0
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    execute_semantic_search_func,
                    optimized_queries,
                    search_params,
                    original_query
                )
                try:
                    result, count = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: future.result(timeout=timeout)),
                        timeout=timeout
                    )
                    return result, count
                except (FutureTimeoutError, asyncio.TimeoutError) as e:
                    logger.warning(f"Semantic search timeout: {e}")
                    return [], 0
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")
                    return [], 0

        async def run_keyword_search():
            """키워드 검색 실행 (비동기 래퍼)"""
            if not execute_keyword_search_func:
                return [], 0
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    execute_keyword_search_func,
                    optimized_queries,
                    search_params,
                    query_type_str,
                    legal_field,
                    extracted_keywords,
                    original_query
                )
                try:
                    result, count = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: future.result(timeout=timeout)),
                        timeout=timeout
                    )
                    return result, count
                except (FutureTimeoutError, asyncio.TimeoutError) as e:
                    logger.warning(f"Keyword search timeout: {e}")
                    return [], 0
                except Exception as e:
                    logger.error(f"Keyword search failed: {e}")
                    return [], 0

        semantic_results, semantic_count = await run_semantic_search()
        keyword_results, keyword_count = await run_keyword_search()

        return semantic_results, semantic_count, keyword_results, keyword_count

    @staticmethod
    def execute_searches_sync(
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        query_type_str: str,
        legal_field: str,
        extracted_keywords: List[str],
        original_query: str,
        execute_semantic_search_func: Optional[Callable] = None,
        execute_keyword_search_func: Optional[Callable] = None,
        timeout: float = 30.0
    ) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]], int]:
        """
        의미 검색과 키워드 검색을 동기적으로 병렬 실행 (ThreadPoolExecutor 사용)
        
        Args:
            optimized_queries: 최적화된 쿼리 딕셔너리
            search_params: 검색 파라미터
            query_type_str: 질의 유형 문자열
            legal_field: 법률 분야
            extracted_keywords: 추출된 키워드
            original_query: 원본 질의
            execute_semantic_search_func: 의미 검색 실행 함수
            execute_keyword_search_func: 키워드 검색 실행 함수
            timeout: 타임아웃 (초)
        
        Returns:
            (semantic_results, semantic_count, keyword_results, keyword_count)
        """
        semantic_results = []
        semantic_count = 0
        keyword_results = []
        keyword_count = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = None
            keyword_future = None

            if execute_semantic_search_func:
                semantic_future = executor.submit(
                    execute_semantic_search_func,
                    optimized_queries,
                    search_params,
                    original_query
                )

            if execute_keyword_search_func:
                keyword_future = executor.submit(
                    execute_keyword_search_func,
                    optimized_queries,
                    search_params,
                    query_type_str,
                    legal_field,
                    extracted_keywords,
                    original_query
                )

            if semantic_future:
                try:
                    semantic_results, semantic_count = semantic_future.result(timeout=timeout)
                except FutureTimeoutError:
                    logger.warning(f"Semantic search timeout after {timeout}s")
                    semantic_results, semantic_count = [], 0
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")
                    semantic_results, semantic_count = [], 0

            if keyword_future:
                try:
                    keyword_results, keyword_count = keyword_future.result(timeout=timeout)
                except FutureTimeoutError:
                    logger.warning(f"Keyword search timeout after {timeout}s")
                    keyword_results, keyword_count = [], 0
                except Exception as e:
                    logger.error(f"Keyword search failed: {e}")
                    keyword_results, keyword_count = [], 0

        return semantic_results, semantic_count, keyword_results, keyword_count

