# -*- coding: utf-8 -*-
"""
비동기 배치 처리 유틸리티
네트워크 요청 및 LLM 호출을 배치로 처리하여 성능 최적화
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional, TypeVar, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncBatchProcessor:
    """비동기 배치 처리 클래스"""
    
    def __init__(self, batch_size: int = 5, timeout: float = 30.0):
        """
        Args:
            batch_size: 배치 크기 (동시 실행할 작업 수)
            timeout: 작업별 타임아웃 (초)
        """
        self.batch_size = batch_size
        self.timeout = timeout
    
    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[Any], Awaitable[T]],
        timeout: Optional[float] = None
    ) -> List[T]:
        """
        배치로 비동기 작업 처리
        
        Args:
            items: 처리할 아이템 리스트
            process_func: 각 아이템을 처리하는 비동기 함수
            timeout: 작업별 타임아웃 (None이면 self.timeout 사용)
        
        Returns:
            처리 결과 리스트
        """
        timeout = timeout or self.timeout
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            tasks = [
                asyncio.wait_for(process_func(item), timeout=timeout)
                for item in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for idx, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Batch item {i + idx} failed: {result}")
                        results.append(None)
                    else:
                        results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.extend([None] * len(batch))
        
        return results


def with_timeout(timeout: float = 30.0):
    """비동기 함수에 타임아웃을 추가하는 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout} seconds")
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
        return wrapper
    return decorator


async def batch_llm_calls(
    prompts: List[str],
    llm_call_func: Callable[[str], Awaitable[str]],
    batch_size: int = 5,
    timeout: float = 30.0
) -> List[str]:
    """
    여러 LLM 호출을 배치로 처리
    
    Args:
        prompts: 프롬프트 리스트
        llm_call_func: LLM 호출 함수 (비동기)
        batch_size: 배치 크기
        timeout: 각 호출의 타임아웃
    
    Returns:
        응답 리스트
    """
    processor = AsyncBatchProcessor(batch_size=batch_size, timeout=timeout)
    return await processor.process_batch(prompts, llm_call_func, timeout)

