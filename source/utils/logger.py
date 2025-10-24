# -*- coding: utf-8 -*-
"""
Logging Configuration
로깅 설정 및 관리
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import Config


def setup_logging(config: Optional[Config] = None) -> None:
    """로깅 설정 - 보안 감사 로그와 일반 로그 분리"""
    if config is None:
        config = Config()
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    
    # 기존 핸들러 제거
    root_logger.handlers.clear()
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    log_file = log_dir / "lawfirm_ai.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 외부 라이브러리 로깅 레벨 조정 (필요시)
    external_loggers = {
        "faiss": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "requests": logging.WARNING,
        "urllib3": logging.WARNING,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)
    
    print(f"✅ 로깅 시스템 초기화 완료 - 로그 파일: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)