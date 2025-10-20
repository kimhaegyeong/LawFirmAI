# -*- coding: utf-8 -*-
"""
Logging Configuration
로깅 설정 및 관리
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from .config import Config


def setup_logging(config: Optional[Config] = None) -> None:
    """로깅 설정 - 환경 변수 레벨 완전 차단"""
    if config is None:
        config = Config()
    
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
    
    # 로깅 시스템 완전 차단
    logging.disable(logging.CRITICAL)
    
    # 모든 로거 비활성화
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False
        logger.handlers.clear()
        logger.setLevel(logging.CRITICAL)
    
    # 루트 로거 완전 비활성화
    root_logger = logging.getLogger()
    root_logger.disabled = True
    root_logger.handlers.clear()
    root_logger.propagate = False
    root_logger.setLevel(logging.CRITICAL)
    
    # 외부 라이브러리 로깅 완전 차단
    external_loggers = [
        "faiss", "sentence_transformers", "transformers", "torch", 
        "numpy", "scipy", "sklearn", "matplotlib", "pandas",
        "requests", "urllib3", "httpx", "aiohttp", "uvicorn", "fastapi",
        "tqdm", "tokenizers", "datasets", "accelerate", "bitsandbytes"
    ]
    
    for logger_name in external_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False
        logger.handlers.clear()
        logger.setLevel(logging.CRITICAL)
    
    # structlog 완전 비활성화
    try:
        import structlog
        structlog.configure(
            processors=[],
            logger_factory=lambda *args, **kwargs: None,
            wrapper_class=lambda *args, **kwargs: None,
            cache_logger_on_first_use=False,
        )
    except Exception:
        pass
    
    # 로깅 핸들러 완전 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로깅 레벨 최고로 설정
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # 로깅 모듈 자체를 비활성화
    logging.basicConfig(level=logging.CRITICAL, handlers=[])
    
    # sys.stdout/sys.stderr 리다이렉션으로 로깅 완전 차단
    try:
        import io
        devnull = io.StringIO()
        sys.stdout = devnull
        sys.stderr = devnull
    except Exception:
        pass
    
    print("로깅 시스템이 환경 변수 레벨에서 완전히 차단되었습니다")


def get_logger(name: str):
    """로거 반환 - 로거 문제 해결을 위해 기본 로거 사용"""
    import logging
    return logging.getLogger(name)
