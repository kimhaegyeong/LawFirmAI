# -*- coding: utf-8 -*-
"""
Logging Configuration
로깅 설정 및 관리
"""

import logging
import threading
from typing import Optional

# structlog 선택적 import
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

from .config import Config

# TRACE 레벨 추가 (DEBUG보다 낮은 레벨, 값: 5)
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")

# Logger 클래스에 trace 메서드 추가
def trace(self, message, *args, **kwargs):
    """TRACE 레벨 로그"""
    if self.isEnabledFor(logging.TRACE):
        self._log(logging.TRACE, message, args, **kwargs)

logging.Logger.trace = trace

# 스레드 안전을 위한 락
_logging_lock = threading.Lock()


def _safe_remove_handlers(logger: logging.Logger) -> None:
    """
    안전하게 로거 핸들러 제거 (멀티스레딩 안전)
    
    Args:
        logger: 제거할 핸들러가 있는 로거
    """
    try:
        # 핸들러를 제거하기 전에 플러시하고 닫기
        for handler in logger.handlers[:]:
            try:
                handler.flush()
            except (ValueError, AttributeError):
                pass
            try:
                handler.close()
            except (ValueError, AttributeError):
                pass
            try:
                logger.removeHandler(handler)
            except (ValueError, AttributeError):
                pass
    except (ValueError, AttributeError, RuntimeError):
        # 핸들러 제거 중 오류 발생 시 무시 (안전한 실패)
        pass


def configure_global_logging(log_level: Optional[str] = None) -> None:
    """
    전역 로깅 설정 - 루트 로거와 모든 하위 로거 레벨 설정
    
    이 함수는 환경 변수 LOG_LEVEL을 읽어서 루트 로거와 lawfirm_langgraph 네임스페이스 로거의
    레벨을 설정합니다. 모든 하위 로거는 루트 로거로 전파되므로 전역적으로 로그 레벨이 관리됩니다.
    
    Args:
        log_level: 로그 레벨 (CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE)
                  None이면 환경 변수 LOG_LEVEL 사용 (기본값: INFO)
    
    사용 예시:
        # 환경 변수 사용
        configure_global_logging()  # LOG_LEVEL 환경 변수 자동 읽기
        
        # 직접 설정
        configure_global_logging("DEBUG")
    """
    import os
    
    # 스레드 락을 사용하여 안전하게 로깅 설정
    with _logging_lock:
        # 로그 레벨 결정
        if log_level is None:
            log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        else:
            log_level_str = log_level.upper()
        
        log_level_map = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "TRACE": logging.TRACE,
        }
        log_level_value = log_level_map.get(log_level_str, logging.INFO)
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_value)
        root_logger.disabled = False
        root_logger.propagate = True
        
        # 핸들러가 없으면 추가
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level_value)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            root_logger.addHandler(handler)
        else:
            # 기존 핸들러 레벨도 업데이트
            for handler in root_logger.handlers:
                handler.setLevel(log_level_value)
        
        # lawfirm_langgraph 네임스페이스 로거 설정
        langgraph_logger = logging.getLogger("lawfirm_langgraph")
        langgraph_logger.setLevel(log_level_value)
        langgraph_logger.propagate = True
        langgraph_logger.disabled = False
        
        # 외부 라이브러리 로거는 CRITICAL로 유지 (기존 동작 유지)
        external_loggers = [
            "faiss", "sentence_transformers", "transformers", "torch",
            "numpy", "scipy", "sklearn", "matplotlib", "pandas",
            "requests", "urllib3", "httpx", "aiohttp", "uvicorn", "fastapi",
            "tqdm", "tokenizers", "datasets", "accelerate", "bitsandbytes"
        ]
        
        for logger_name in external_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
            except (ValueError, AttributeError, RuntimeError):
                pass
        
        # asyncio 로거는 TRACE 레벨로 설정 (proactor 로그 과다 방지)
        try:
            asyncio_logger = logging.getLogger("asyncio")
            asyncio_logger.setLevel(logging.TRACE)
            asyncio_logger.propagate = False
        except (ValueError, AttributeError, RuntimeError):
            pass


def setup_logging(config: Optional[Config] = None) -> None:
    """로깅 설정 - 환경 변수 레벨 완전 차단 (멀티스레딩 안전)"""
    if config is None:
        config = Config()
    
    # 스레드 락을 사용하여 안전하게 로깅 설정
    with _logging_lock:
        # 환경 변수로 로깅 완전 차단
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # 모든 로깅 완전 비활성화
        import logging
        import sys
        import warnings
        
        # 경고 메시지 완전 차단
        warnings.filterwarnings('ignore')
        warnings.simplefilter('ignore')
        
        # 로깅 시스템 완전 차단 (핸들러 제거 전에 먼저 비활성화)
        logging.disable(logging.CRITICAL)
        
        # 모든 로거 비활성화 (핸들러 제거 대신 레벨 조정 우선)
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logger = logging.getLogger(logger_name)
                logger.disabled = True
                logger.propagate = False
                logger.setLevel(logging.CRITICAL)
                # 핸들러는 안전하게 제거
                _safe_remove_handlers(logger)
            except (ValueError, AttributeError, RuntimeError):
                # 로거 처리 중 오류 발생 시 무시
                pass
        
        # 루트 로거 완전 비활성화
        try:
            root_logger = logging.getLogger()
            root_logger.disabled = True
            root_logger.propagate = False
            root_logger.setLevel(logging.CRITICAL)
            # 핸들러는 안전하게 제거
            _safe_remove_handlers(root_logger)
        except (ValueError, AttributeError, RuntimeError):
            pass
        
        # 외부 라이브러리 로깅 완전 차단
        external_loggers = [
            "faiss", "sentence_transformers", "transformers", "torch", 
            "numpy", "scipy", "sklearn", "matplotlib", "pandas",
            "requests", "urllib3", "httpx", "aiohttp", "uvicorn", "fastapi",
            "tqdm", "tokenizers", "datasets", "accelerate", "bitsandbytes"
        ]
        
        for logger_name in external_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.disabled = True
                logger.propagate = False
                logger.setLevel(logging.CRITICAL)
                # 핸들러는 안전하게 제거
                _safe_remove_handlers(logger)
            except (ValueError, AttributeError, RuntimeError):
                pass
        
        # structlog 완전 비활성화
        if STRUCTLOG_AVAILABLE and structlog:
            try:
                structlog.configure(
                    processors=[],
                    logger_factory=lambda *args, **kwargs: None,
                    wrapper_class=lambda *args, **kwargs: None,
                    cache_logger_on_first_use=False,
                )
            except Exception:
                pass
        
        # 루트 로거 핸들러 안전하게 제거
        try:
            _safe_remove_handlers(logging.root)
        except (ValueError, AttributeError, RuntimeError):
            pass
        
        # 로깅 레벨 최고로 설정
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # 로깅 모듈 자체를 비활성화 (핸들러 없이)
        try:
            logging.basicConfig(level=logging.CRITICAL, handlers=[], force=True)
        except Exception:
            pass
        
        # sys.stdout/sys.stderr 리다이렉션은 제거 (멀티스레딩에서 문제 발생 가능)
        # 대신 로깅 레벨로만 제어
        
        # Windows 환경에서 로깅 버퍼 분리 문제 해결
        if sys.platform == "win32":
            try:
                # sys.stdout 버퍼링 비활성화
                import io
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    write_through=True
                )
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    write_through=True
                )
            except Exception:
                # 버퍼링 비활성화 실패 시 무시
                pass


def get_logger(name: str):
    """
    로거 반환 - 환경 변수 LOG_LEVEL을 읽어서 로거 레벨 설정
    루트 로거도 함께 설정하여 전역 로그 레벨 관리
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거
    
    Note:
        이 함수는 최초 호출 시 루트 로거 레벨도 함께 설정합니다.
        이후 호출에서는 루트 로거가 이미 설정되어 있으면 그대로 사용합니다.
    """
    import os
    
    # 환경 변수에서 LOG_LEVEL 읽기 (기본값: INFO)
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "TRACE": logging.TRACE,
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    
    # 루트 로거 설정 (최초 호출 시에만 또는 레벨이 다를 때)
    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET or root_logger.level != log_level:
        # 전역 로깅 설정 호출 (스레드 안전)
        configure_global_logging(log_level_str)
    
    # 개별 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.disabled = False
    logger.propagate = True  # 루트 로거로 전파
    
    return logger


class DeduplicationLogger:
    """중복 로그 방지를 위한 로거 래퍼"""
    
    def __init__(self, logger: logging.Logger, dedup_window: float = 2.0):
        """
        Args:
            logger: 기본 로거
            dedup_window: 중복 방지 시간 창 (초)
        """
        self.logger = logger
        self.dedup_window = dedup_window
        self._recent_logs: dict = {}  # {log_key: timestamp}
        import time
        self._last_cleanup = time.time()
        self._cleanup_interval = dedup_window * 2  # 정리 주기
    
    def _get_log_key(self, level: str, message: str) -> str:
        """로그 메시지의 고유 키 생성"""
        # 태그 부분만 추출하여 키로 사용
        import re
        tag_match = re.search(r'\[([^\]]+)\]', message)
        if tag_match:
            tag = tag_match.group(1)
            # 태그와 메시지의 핵심 부분만 사용
            message_preview = message[:100].replace('\n', ' ').strip()
            return f"{level}:{tag}:{hash(message_preview) % 10000}"
        # 태그가 없으면 메시지의 처음 50자 사용
        message_preview = message[:50].replace('\n', ' ').strip()
        return f"{level}:{hash(message_preview) % 10000}"
    
    def _should_log(self, level: str, message: str) -> bool:
        """중복 로그인지 확인"""
        import time
        log_key = self._get_log_key(level, message)
        current_time = time.time()
        
        # 주기적으로 오래된 로그 정리 (메모리 관리)
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._recent_logs = {
                k: v for k, v in self._recent_logs.items()
                if current_time - v < self._cleanup_interval
            }
            self._last_cleanup = current_time
        
        if log_key in self._recent_logs:
            last_time = self._recent_logs[log_key]
            if current_time - last_time < self.dedup_window:
                return False
        
        self._recent_logs[log_key] = current_time
        return True
    
    def info(self, message: str, *args, **kwargs):
        """INFO 레벨 로그 (중복 방지)"""
        if self._should_log("INFO", message):
            self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """WARNING 레벨 로그 (중복 방지)"""
        if self._should_log("WARNING", message):
            self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """ERROR 레벨 로그 (중복 방지)"""
        if self._should_log("ERROR", message):
            self.logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """DEBUG 레벨 로그 (중복 방지)"""
        if self._should_log("DEBUG", message):
            self.logger.debug(message, *args, **kwargs)
    
    def trace(self, message: str, *args, **kwargs):
        """TRACE 레벨 로그 (중복 방지)"""
        if self._should_log("TRACE", message):
            self.logger.trace(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """예외 로그 (항상 출력)"""
        self.logger.exception(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """CRITICAL 레벨 로그 (항상 출력)"""
        self.logger.critical(message, *args, **kwargs)