# -*- coding: utf-8 -*-
"""
에러 핸들링 유틸리티
HuggingFace Spaces 환경에서 에러를 효과적으로 처리합니다.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(Enum):
    """에러 유형"""
    INITIALIZATION = "initialization"
    MEMORY = "memory"
    NETWORK = "network"
    MODEL = "model"
    DATABASE = "database"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """에러 정보 데이터 클래스"""
    timestamp: datetime
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recoverable: bool = True

class ErrorHandler:
    """에러 핸들링 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        self.error_counts = {}
        self.recovery_attempts = {}
        
    def handle_error(self, 
                    error: Exception, 
                    error_type: ErrorType = ErrorType.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    recoverable: bool = True) -> ErrorInfo:
        """에러 처리"""
        
        # 에러 정보 생성
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            message=str(error),
            details=traceback.format_exc(),
            context=context,
            recoverable=recoverable
        )
        
        # 에러 기록
        self.error_history.append(error_info)
        self.error_counts[error_type.value] = self.error_counts.get(error_type.value, 0) + 1
        
        # 로깅
        self._log_error(error_info)
        
        # 복구 시도
        if recoverable:
            self._attempt_recovery(error_info)
        
        return error_info
    
    def _log_error(self, error_info: ErrorInfo):
        """에러 로깅"""
        log_message = f"Error [{error_info.error_type.value}] [{error_info.severity.value}]: {error_info.message}"
        
        if error_info.context:
            log_message += f" Context: {error_info.context}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # 디버그 모드에서 상세 정보 로깅
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Error details: {error_info.details}")
    
    def _attempt_recovery(self, error_info: ErrorInfo):
        """복구 시도"""
        error_key = f"{error_info.error_type.value}_{error_info.message}"
        self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
        
        # 복구 시도 횟수 제한
        if self.recovery_attempts[error_key] > 3:
            self.logger.error(f"Too many recovery attempts for {error_key}")
            return
        
        # 에러 유형별 복구 전략
        if error_info.error_type == ErrorType.MEMORY:
            self._recover_memory_error()
        elif error_info.error_type == ErrorType.NETWORK:
            self._recover_network_error()
        elif error_info.error_type == ErrorType.MODEL:
            self._recover_model_error()
        elif error_info.error_type == ErrorType.DATABASE:
            self._recover_database_error()
        else:
            self._recover_generic_error()
    
    def _recover_memory_error(self):
        """메모리 에러 복구"""
        try:
            import gc
            import torch
            
            # 가비지 컬렉션 실행
            collected = gc.collect()
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Memory recovery: collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
    
    def _recover_network_error(self):
        """네트워크 에러 복구"""
        try:
            import requests
            import time
            
            # 간단한 네트워크 연결 테스트
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                self.logger.info("Network recovery: connection restored")
            else:
                self.logger.warning("Network recovery: connection test failed")
                
        except Exception as e:
            self.logger.error(f"Network recovery failed: {e}")
    
    def _recover_model_error(self):
        """모델 에러 복구"""
        try:
            # 모델 재로딩 시도 (실제 구현에서는 모델 매니저 사용)
            self.logger.info("Model recovery: attempting model reload")
            
        except Exception as e:
            self.logger.error(f"Model recovery failed: {e}")
    
    def _recover_database_error(self):
        """데이터베이스 에러 복구"""
        try:
            # 데이터베이스 연결 재시도 (실제 구현에서는 DB 매니저 사용)
            self.logger.info("Database recovery: attempting reconnection")
            
        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
    
    def _recover_generic_error(self):
        """일반 에러 복구"""
        try:
            # 일반적인 복구 시도
            self.logger.info("Generic recovery: attempting cleanup")
            
        except Exception as e:
            self.logger.error(f"Generic recovery failed: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "error_counts": {},
                "recovery_attempts": {},
                "recent_errors": []
            }
        
        # 최근 에러 (최근 10개)
        recent_errors = [
            {
                "timestamp": error.timestamp.isoformat(),
                "type": error.error_type.value,
                "severity": error.severity.value,
                "message": error.message,
                "recoverable": error.recoverable
            }
            for error in self.error_history[-10:]
        ]
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recovery_attempts": self.recovery_attempts,
            "recent_errors": recent_errors
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 반환"""
        if not self.error_history:
            return {
                "status": "healthy",
                "error_rate": 0.0,
                "critical_errors": 0,
                "recoverable_errors": 0
            }
        
        # 최근 1시간 내 에러 분석
        recent_errors = [
            error for error in self.error_history
            if (datetime.now() - error.timestamp).total_seconds() < 3600
        ]
        
        critical_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL])
        recoverable_errors = len([e for e in recent_errors if e.recoverable])
        
        # 상태 결정
        if critical_errors > 0:
            status = "critical"
        elif len(recent_errors) > 10:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "error_rate": len(recent_errors) / 3600,  # 에러/시간
            "critical_errors": critical_errors,
            "recoverable_errors": recoverable_errors,
            "total_recent_errors": len(recent_errors)
        }

# 전역 에러 핸들러 인스턴스
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """에러 핸들러 인스턴스 반환"""
    return error_handler

def handle_errors(error_type: ErrorType = ErrorType.UNKNOWN,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  recoverable: bool = True):
    """에러 핸들링 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    error=e,
                    error_type=error_type,
                    severity=severity,
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)},
                    recoverable=recoverable
                )
                raise
        return wrapper
    return decorator

def safe_execute(func: Callable, 
                 error_type: ErrorType = ErrorType.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 default_return: Any = None,
                 context: Optional[Dict[str, Any]] = None) -> Any:
    """안전한 함수 실행"""
    try:
        return func()
    except Exception as e:
        error_handler.handle_error(
            error=e,
            error_type=error_type,
            severity=severity,
            context=context,
            recoverable=True
        )
        return default_return
