#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
네트워크 오류 처리 유틸리티

네트워크 오류 발생 시 재시도 로직과 오류 처리를 담당하는 모듈입니다.
- 네트워크 오류 감지
- 지수 백오프 재시도
- 연결 상태 모니터링
- 오류 통계 수집
"""

import time
import logging
import requests
from typing import Dict, Any, Optional, Callable, Tuple, List
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class NetworkErrorHandler:
    """네트워크 오류 처리기"""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 progressive_delays: List[float] = None):
        """
        네트워크 오류 처리기 초기화
        
        Args:
            max_retries: 최대 재시도 횟수
            base_delay: 기본 지연 시간 (초)
            max_delay: 최대 지연 시간 (초)
            backoff_factor: 백오프 배수
            progressive_delays: 점진적 지연 시간 목록 (초) - [180, 300, 600] = [3분, 5분, 10분]
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.progressive_delays = progressive_delays or [180, 300, 600]  # 3분, 5분, 10분
        
        # 오류 통계
        self.error_stats = {
            "total_errors": 0,
            "network_errors": 0,
            "timeout_errors": 0,
            "connection_errors": 0,
            "http_errors": 0,
            "retry_successes": 0,
            "retry_failures": 0,
            "last_error_time": None,
            "error_history": []
        }
        
        logger.info(f"NetworkErrorHandler 초기화 완료 - 최대 재시도: {max_retries}회")
    
    def is_network_error(self, exception: Exception) -> bool:
        """
        네트워크 오류 여부 판단
        
        Args:
            exception: 발생한 예외
            
        Returns:
            네트워크 오류 여부
        """
        network_error_types = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
            ConnectionError,
            TimeoutError
        )
        
        return isinstance(exception, network_error_types)
    
    def classify_error(self, exception: Exception) -> str:
        """
        오류 분류
        
        Args:
            exception: 발생한 예외
            
        Returns:
            오류 분류
        """
        if isinstance(exception, requests.exceptions.ConnectionError):
            return "connection_error"
        elif isinstance(exception, requests.exceptions.Timeout):
            return "timeout_error"
        elif isinstance(exception, requests.exceptions.HTTPError):
            return "http_error"
        elif isinstance(exception, requests.exceptions.RequestException):
            return "network_error"
        else:
            return "unknown_error"
    
    def calculate_delay(self, attempt: int) -> float:
        """
        재시도 지연 시간 계산 (지수 백오프)
        
        Args:
            attempt: 현재 시도 횟수
            
        Returns:
            지연 시간 (초)
        """
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)
    
    def record_error(self, exception: Exception, attempt: int) -> None:
        """
        오류 기록
        
        Args:
            exception: 발생한 예외
            attempt: 시도 횟수
        """
        error_type = self.classify_error(exception)
        error_time = datetime.now()
        
        # 통계 업데이트
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = error_time.isoformat()
        
        if error_type == "connection_error":
            self.error_stats["connection_errors"] += 1
        elif error_type == "timeout_error":
            self.error_stats["timeout_errors"] += 1
        elif error_type == "http_error":
            self.error_stats["http_errors"] += 1
        elif error_type == "network_error":
            self.error_stats["network_errors"] += 1
        
        # 오류 히스토리 기록
        error_record = {
            "timestamp": error_time.isoformat(),
            "error_type": error_type,
            "error_message": str(exception),
            "attempt": attempt,
            "retry_delay": self.calculate_delay(attempt)
        }
        
        self.error_stats["error_history"].append(error_record)
        
        # 최근 100개 오류만 유지
        if len(self.error_stats["error_history"]) > 100:
            self.error_stats["error_history"] = self.error_stats["error_history"][-100:]
        
        logger.warning(f"네트워크 오류 기록: {error_type} (시도 {attempt}/{self.max_retries})")
    
    def record_success(self, attempt: int) -> None:
        """
        재시도 성공 기록
        
        Args:
            attempt: 성공한 시도 횟수
        """
        if attempt > 1:
            self.error_stats["retry_successes"] += 1
            logger.info(f"재시도 성공: {attempt}번째 시도에서 성공")
    
    def record_failure(self) -> None:
        """재시도 실패 기록"""
        self.error_stats["retry_failures"] += 1
        logger.error("모든 재시도 실패")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        지수 백오프를 사용한 재시도 실행
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
            
        Returns:
            함수 실행 결과
            
        Raises:
            Exception: 모든 재시도 실패 시 마지막 예외
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    self.record_success(attempt)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.is_network_error(e):
                    # 네트워크 오류가 아닌 경우 즉시 실패
                    logger.error(f"네트워크 오류가 아닌 예외 발생: {e}")
                    raise e
                
                self.record_error(e, attempt)
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(f"재시도 {attempt}/{self.max_retries} - {delay:.1f}초 후 재시도")
                    time.sleep(delay)
                else:
                    logger.error(f"최대 재시도 횟수({self.max_retries}) 초과")
        
        self.record_failure()
        raise last_exception
    
    def retry_with_progressive_delays(self, func: Callable, *args, **kwargs) -> Any:
        """
        점진적 지연 시간을 사용한 재시도 실행 (3분, 5분, 10분)
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
            
        Returns:
            함수 실행 결과
            
        Raises:
            Exception: 모든 재시도 실패 시 마지막 예외
        """
        last_exception = None
        total_attempts = len(self.progressive_delays) + 1  # 초기 시도 + 점진적 지연 시도들
        
        # 초기 시도
        try:
            result = func(*args, **kwargs)
            logger.info("초기 시도 성공")
            return result
        except Exception as e:
            last_exception = e
            
            if not self.is_network_error(e):
                logger.error(f"네트워크 오류가 아닌 예외 발생: {e}")
                raise e
            
            logger.warning(f"초기 시도 실패: {e}")
            self.record_error(e, 1)
        
        # 점진적 지연 시간으로 재시도
        for attempt, delay in enumerate(self.progressive_delays, 2):
            try:
                logger.warning(f"재시도 {attempt}/{total_attempts} - {delay/60:.1f}분 후 재시도")
                print(f"⏳ 네트워크 지연으로 인한 대기: {delay/60:.1f}분 후 재시도...")
                
                time.sleep(delay)
                
                result = func(*args, **kwargs)
                self.record_success(attempt)
                logger.info(f"재시도 {attempt} 성공")
                print(f"✅ 재시도 {attempt} 성공!")
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.is_network_error(e):
                    logger.error(f"네트워크 오류가 아닌 예외 발생: {e}")
                    raise e
                
                self.record_error(e, attempt)
                logger.warning(f"재시도 {attempt} 실패: {e}")
                
                if attempt < total_attempts:
                    print(f"❌ 재시도 {attempt} 실패: {e}")
                else:
                    logger.error(f"모든 점진적 재시도 실패")
                    print(f"❌ 모든 재시도 실패 - 프로그램을 중지합니다")
        
        self.record_failure()
        raise last_exception
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        오류 통계 조회
        
        Returns:
            오류 통계 정보
        """
        stats = self.error_stats.copy()
        
        # 성공률 계산
        total_retries = stats["retry_successes"] + stats["retry_failures"]
        if total_retries > 0:
            stats["retry_success_rate"] = (stats["retry_successes"] / total_retries) * 100
        else:
            stats["retry_success_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """오류 통계 초기화"""
        self.error_stats = {
            "total_errors": 0,
            "network_errors": 0,
            "timeout_errors": 0,
            "connection_errors": 0,
            "http_errors": 0,
            "retry_successes": 0,
            "retry_failures": 0,
            "last_error_time": None,
            "error_history": []
        }
        logger.info("오류 통계 초기화 완료")
    
    def test_connection(self, url: str, timeout: int = 10) -> bool:
        """
        연결 테스트
        
        Args:
            url: 테스트할 URL
            timeout: 타임아웃 (초)
            
        Returns:
            연결 성공 여부
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"연결 테스트 실패: {e}")
            return False


def retry_on_network_error(max_retries: int = 5, base_delay: float = 1.0, 
                          max_delay: float = 60.0, backoff_factor: float = 2.0):
    """
    네트워크 오류 시 재시도하는 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수
        base_delay: 기본 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
        backoff_factor: 백오프 배수
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = NetworkErrorHandler(max_retries, base_delay, max_delay, backoff_factor)
            return handler.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator


def create_network_error_handler(max_retries: int = 5, base_delay: float = 1.0, 
                                max_delay: float = 60.0, backoff_factor: float = 2.0) -> NetworkErrorHandler:
    """
    네트워크 오류 처리기 생성
    
    Args:
        max_retries: 최대 재시도 횟수
        base_delay: 기본 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
        backoff_factor: 백오프 배수
        
    Returns:
        네트워크 오류 처리기 인스턴스
    """
    return NetworkErrorHandler(max_retries, base_delay, max_delay, backoff_factor)
