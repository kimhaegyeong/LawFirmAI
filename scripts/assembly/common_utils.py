#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Collection Common Utilities
국회 수집 스크립트들의 공통 유틸리티 모듈

이 모듈은 모든 수집 스크립트에서 공통으로 사용되는 기능들을 제공합니다:
- 메모리 관리
- 에러 처리
- 로깅 설정
- 시그널 핸들링
- 재시도 로직
"""

import gc
import psutil
import logging
import signal
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
import json


class MemoryManager:
    """메모리 관리 유틸리티 클래스"""
    
    def __init__(self, memory_limit_mb: int = 600, cleanup_threshold: float = 0.7):
        """
        메모리 매니저 초기화
        
        Args:
            memory_limit_mb: 메모리 제한 (MB)
            cleanup_threshold: 정리 시작 임계값 (0.0-1.0)
        """
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_and_cleanup(self) -> bool:
        """
        메모리 사용량 체크 및 정리
        
        Returns:
            bool: 메모리 제한 내에 있으면 True, 초과하면 False
        """
        memory_mb = self.get_memory_usage()
        
        if memory_mb > self.memory_limit_mb * self.cleanup_threshold:
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB, cleaning up...")
            gc.collect()
            
            memory_after = self.get_memory_usage()
            self.logger.info(f"After cleanup: {memory_after:.1f}MB")
            
            if memory_after > self.memory_limit_mb:
                self.logger.error(f"Memory limit exceeded: {memory_after:.1f}MB")
                return False
        
        return True
    
    def force_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        self.logger.info("Forced memory cleanup completed")


class RetryManager:
    """재시도 관리 유틸리티 클래스"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        """
        재시도 매니저 초기화
        
        Args:
            max_retries: 최대 재시도 횟수
            base_delay: 기본 대기 시간 (초)
            backoff_factor: 지수 백오프 계수
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        함수 재시도 실행
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
            
        Returns:
            함수 실행 결과
            
        Raises:
            Exception: 최대 재시도 후에도 실패한 경우
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


class SignalHandler:
    """시그널 핸들링 유틸리티 클래스"""
    
    def __init__(self):
        """시그널 핸들러 초기화"""
        self.interrupted = False
        self.logger = logging.getLogger(__name__)
        self._setup_handlers()
    
    def _setup_handlers(self):
        """시그널 핸들러 설정"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.warning(f"Signal {signum} received. Initiating graceful shutdown...")
        self.interrupted = True
    
    def is_interrupted(self) -> bool:
        """중단 신호 확인"""
        return self.interrupted


class CollectionLogger:
    """수집 작업 로깅 유틸리티 클래스"""
    
    @staticmethod
    def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
        """
        로깅 설정
        
        Args:
            name: 로거 이름
            level: 로그 레벨
            
        Returns:
            설정된 로거
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            
            # 파일 핸들러
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_progress(logger: logging.Logger, current: int, total: int, item_type: str = "items"):
        """진행률 로깅"""
        if total > 0:
            percentage = (current / total) * 100
            logger.info(f"Progress: {current}/{total} {item_type} ({percentage:.1f}%)")
    
    @staticmethod
    def log_memory_usage(logger: logging.Logger, memory_mb: float, limit_mb: float):
        """메모리 사용량 로깅"""
        usage_percent = (memory_mb / limit_mb) * 100
        logger.info(f"Memory: {memory_mb:.1f}MB / {limit_mb}MB ({usage_percent:.1f}%)")


class DataOptimizer:
    """데이터 최적화 유틸리티 클래스"""
    
    # 대용량 필드 크기 제한 설정
    FIELD_LIMITS = {
        'content_html': 1_000_000,      # 1MB
        'precedent_content': 1_000_000,  # 1MB
        'law_content': 1_000_000,        # 1MB
        'full_text': 1_000_000,         # 1MB
        'structured_content': 500_000,   # 500KB
    }
    
    @classmethod
    def optimize_item(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        아이템 메모리 최적화
        
        Args:
            item: 최적화할 아이템
            
        Returns:
            최적화된 아이템
        """
        optimized = item.copy()
        
        # 대용량 필드 크기 제한
        for field, limit in cls.FIELD_LIMITS.items():
            if field in optimized and isinstance(optimized[field], str):
                if len(optimized[field]) > limit:
                    optimized[field] = optimized[field][:limit] + "... [TRUNCATED]"
        
        # structured_content 내부 필드도 최적화
        if 'structured_content' in optimized and isinstance(optimized['structured_content'], dict):
            structured = optimized['structured_content']
            for key, value in structured.items():
                if isinstance(value, str) and len(value) > 200_000:  # 200KB 제한
                    structured[key] = value[:200_000] + "... [TRUNCATED]"
        
        return optimized
    
    @classmethod
    def save_compressed_json(cls, data: Any, filepath: Path) -> bool:
        """
        압축된 JSON으로 저장
        
        Args:
            data: 저장할 데이터
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save compressed JSON: {e}")
            return False


class CollectionConfig:
    """수집 설정 관리 클래스"""
    
    # 기본 설정값들
    DEFAULT_CONFIG = {
        'memory_limit_mb': 600,
        'batch_size': 20,
        'page_size': 10,
        'rate_limit': 3.0,
        'timeout': 30000,
        'max_retries': 3,
        'cleanup_threshold': 0.7,
        'log_level': 'INFO'
    }
    
    def __init__(self, **kwargs):
        """
        설정 초기화
        
        Args:
            **kwargs: 설정 오버라이드
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """설정 업데이트"""
        self.config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return self.config.copy()


def memory_monitor(threshold_mb: float = 500.0):
    """
    메모리 모니터링 데코레이터
    
    Args:
        threshold_mb: 메모리 임계값 (MB)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = MemoryManager()
            
            # 함수 실행 전 메모리 체크
            if not memory_manager.check_and_cleanup():
                raise MemoryError(f"Memory limit exceeded before {func.__name__}")
            
            result = func(*args, **kwargs)
            
            # 함수 실행 후 메모리 체크
            memory_mb = memory_manager.get_memory_usage()
            if memory_mb > threshold_mb:
                logging.getLogger(__name__).warning(
                    f"High memory usage after {func.__name__}: {memory_mb:.1f}MB"
                )
                memory_manager.force_cleanup()
            
            return result
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    실패 시 재시도 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수
        delay: 재시도 간격 (초)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = RetryManager(max_retries=max_retries, base_delay=delay)
            return retry_manager.retry(func, *args, **kwargs)
        return wrapper
    return decorator


# 편의 함수들
def get_system_memory_info() -> Dict[str, float]:
    """시스템 메모리 정보 반환"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    except Exception:
        return {'total_gb': 0, 'available_gb': 0, 'used_percent': 0, 'free_gb': 0}


def check_system_requirements(min_memory_gb: float = 2.0) -> bool:
    """
    시스템 요구사항 확인
    
    Args:
        min_memory_gb: 최소 필요 메모리 (GB)
        
    Returns:
        요구사항 충족 여부
    """
    memory_info = get_system_memory_info()
    
    if memory_info['available_gb'] < min_memory_gb:
        logging.getLogger(__name__).warning(
            f"Low available memory: {memory_info['available_gb']:.1f}GB "
            f"(minimum required: {min_memory_gb}GB)"
        )
        return False
    
    if memory_info['used_percent'] > 80:
        logging.getLogger(__name__).warning(
            f"High memory usage: {memory_info['used_percent']:.1f}%"
        )
        return False
    
    return True
