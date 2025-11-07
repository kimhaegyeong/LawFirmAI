# -*- coding: utf-8 -*-
"""
Logging Configuration
로깅 설정 및 관리
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# structlog 선택적 import
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

from .config import Config

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
    """로거 반환 - 로거 문제 해결을 위해 기본 로거 사용"""
    import logging
    return logging.getLogger(name)
