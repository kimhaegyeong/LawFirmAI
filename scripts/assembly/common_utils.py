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
    """메모리 관리 유틸리티 클래스 (최적화된 버전)"""
    
    def __init__(self, memory_limit_mb: int = 600, cleanup_threshold: float = 0.6):  # 400MB → 600MB로 조정
        """
        메모리 매니저 초기화
        
        Args:
            memory_limit_mb: 메모리 제한 (MB) - 기본값을 600MB로 설정
            cleanup_threshold: 정리 시작 임계값 (0.0-1.0) - 60%에서 시작
        """
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.warning_threshold = 0.5  # 50%에서 경고
        self.critical_threshold = 0.8  # 80%에서 중단
        self.logger = logging.getLogger(__name__)
        self.cleanup_count = 0
        
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
        메모리 사용량 체크 및 정리 (강화된 버전)
        
        Returns:
            bool: 메모리 제한 내에 있으면 True, 초과하면 False
        """
        memory_mb = self.get_memory_usage()
        
        # 50%에서 경고
        if memory_mb > self.memory_limit_mb * self.warning_threshold:
            self.logger.warning(f"Memory usage warning: {memory_mb:.1f}MB ({memory_mb/self.memory_limit_mb*100:.1f}%)")
        
        # 60%에서 정리 시작
        if memory_mb > self.memory_limit_mb * self.cleanup_threshold:
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB, cleaning up...")
            self.aggressive_cleanup()
            
            memory_after = self.get_memory_usage()
            self.logger.info(f"After cleanup: {memory_after:.1f}MB")
            
            if memory_after > self.memory_limit_mb:
                self.logger.error(f"Memory limit exceeded: {memory_after:.1f}MB")
                return False
        
        # 80%에서 중단
        if memory_mb > self.memory_limit_mb * self.critical_threshold:
            self.logger.error(f"Critical memory usage: {memory_mb:.1f}MB - stopping")
            return False
        
        return True
    
    def aggressive_cleanup(self):
        """강화된 메모리 정리"""
        self.cleanup_count += 1
        
        # 강제 가비지 컬렉션
        gc.collect()
        
        # 메모리 사용량 재확인
        current_mb = self.get_memory_usage()
        if current_mb > self.memory_limit_mb * 0.8:
            if self.cleanup_count > 3:
                raise MemoryError(f"Memory cleanup failed after {self.cleanup_count} attempts: {current_mb:.1f}MB")
        
        self.logger.info(f"Aggressive cleanup completed (attempt {self.cleanup_count})")
    
    def force_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        self.logger.info("Forced memory cleanup completed")
    
    def adaptive_batch_size(self, current_memory_mb: float, base_batch_size: int = 10) -> int:
        """메모리 사용량에 따른 배치 크기 동적 조정"""
        if current_memory_mb > self.memory_limit_mb * 0.7:  # 70% 초과시
            return max(5, base_batch_size - 3)
        elif current_memory_mb > self.memory_limit_mb * 0.5:  # 50% 초과시
            return max(8, base_batch_size - 2)
        else:
            return min(15, base_batch_size + 1)


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
    """데이터 최적화 유틸리티 클래스 (최적화된 버전)"""
    
    # 대용량 필드 크기 제한 설정 (더 엄격하게)
    FIELD_LIMITS = {
        'content_html': 500_000,        # 1MB → 500KB로 감소
        'precedent_content': 500_000,   # 1MB → 500KB로 감소
        'law_content': 500_000,          # 1MB → 500KB로 감소
        'full_text': 500_000,           # 1MB → 500KB로 감소
        'structured_content': 300_000,  # 500KB → 300KB로 감소
        'case_summary': 50_000,          # 새로 추가: 케이스 요약 50KB 제한
        'legal_sections': 100_000,       # 새로 추가: 법조문 100KB 제한
    }
    
    # 필수 필드만 유지하는 설정 (완전한 판례 데이터 유지)
    ESSENTIAL_FIELDS = {
        'precedent': [
            'row_number', 'case_name', 'case_number', 'decision_date', 'field', 'court',
            'detail_url', 'structured_content', 'precedent_content', 'content_html',
            'full_text_length', 'extracted_content_length', 'params', 'collected_at',
            'source_url', 'category', 'category_code'
        ],
        'law': [
            'law_name', 'law_number', 'enactment_date', 
            'summary', 'main_content', 'category'
        ]
    }
    
    @classmethod
    def optimize_item(cls, item: Dict[str, Any], data_type: str = 'precedent') -> Dict[str, Any]:
        """
        아이템 메모리 최적화 (완전한 데이터 유지 버전)
        
        Args:
            item: 최적화할 아이템
            data_type: 데이터 타입 ('precedent' 또는 'law')
            
        Returns:
            최적화된 아이템
        """
        # 원본 데이터가 비어있거나 유효하지 않으면 그대로 반환
        if not item or not isinstance(item, dict):
            return item
        
        # 판례 데이터는 완전한 구조를 유지하되 메모리만 최적화
        if data_type == 'precedent':
            optimized = item.copy()
        else:
            # 법률 데이터만 필수 필드 추출
            if data_type in cls.ESSENTIAL_FIELDS and item:
                optimized = {}
                for field in cls.ESSENTIAL_FIELDS[data_type]:
                    if field in item and item[field] is not None:
                        optimized[field] = item[field]
                
                # 필수 필드가 하나도 없으면 원본 데이터 유지
                if not optimized:
                    optimized = item.copy()
            else:
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
                if isinstance(value, str) and len(value) > 100_000:  # 200KB → 100KB로 감소
                    structured[key] = value[:100_000] + "... [TRUNCATED]"
        
        # 불필요한 빈 필드 제거 (판례 데이터는 제외)
        if data_type != 'precedent':
            optimized = {k: v for k, v in optimized.items() if v is not None and v != ""}
        
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
    
    # 기본 설정값들 (최적화된 버전)
    DEFAULT_CONFIG = {
        'memory_limit_mb': 600,  # 400MB → 600MB로 조정
        'batch_size': 10,        # 20 → 10으로 감소
        'page_size': 10,
        'rate_limit': 3.0,
        'timeout': 30000,
        'max_retries': 3,
        'cleanup_threshold': 0.6,  # 0.7 → 0.6으로 더 보수적으로
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


def memory_monitor(threshold_mb: float = 500.0):  # 300MB → 500MB로 조정
    """
    메모리 모니터링 데코레이터 (최적화된 버전)
    
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
                memory_manager.aggressive_cleanup()
            
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
