# -*- coding: utf-8 -*-
"""
성능 측정 데코레이터
노드 메서드의 실행 시간을 측정하고 메트릭으로 저장
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def measure_performance(metric_name: Optional[str] = None):
    """
    성능 측정 데코레이터
    
    Args:
        metric_name: 메트릭 이름 (None이면 함수 이름 사용)
    
    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, state: Any, *args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(self, state, *args, **kwargs)
                elapsed = time.time() - start_time
                
                # 메트릭 저장
                metric_key = metric_name or func.__name__
                if hasattr(self, 'performance_metrics'):
                    if not isinstance(self.performance_metrics, dict):
                        self.performance_metrics = {}
                    self.performance_metrics[metric_key] = elapsed
                
                # 로깅
                node_logger = getattr(self, 'logger', logger)
                node_logger.debug(f"[PERFORMANCE] {metric_key}: {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                node_logger = getattr(self, 'logger', logger)
                node_logger.error(f"[PERFORMANCE] {metric_name or func.__name__} failed after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator

