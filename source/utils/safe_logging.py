#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
안전한 로깅 설정 유틸리티
로깅 스트림 문제를 해결하기 위한 안전한 로깅 설정 함수들
"""

import logging
import sys
import os
from typing import Optional


def setup_safe_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    안전한 로깅 설정을 구성합니다.
    
    Args:
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: 로그 포맷 문자열
        log_file: 로그 파일 경로 (선택사항)
    
    Returns:
        설정된 로거
    """
    # 기본 포맷 설정
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 기존 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새로운 핸들러 설정
    handlers = []
    
    # 콘솔 핸들러 (안전한 설정)
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    except Exception as e:
        print(f"Warning: Could not setup console logging: {e}")
    
    # 파일 핸들러 (선택사항)
    if log_file:
        try:
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # 루트 로거 설정
    root_logger.setLevel(getattr(logging, level.upper()))
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 특정 라이브러리의 로깅 레벨 조정
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def get_safe_logger(name: str) -> logging.Logger:
    """
    안전한 로거를 반환합니다.
    
    Args:
        name: 로거 이름
    
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)


def disable_external_logging():
    """
    외부 라이브러리의 로깅을 비활성화합니다.
    """
    # FAISS 로깅 비활성화
    logging.getLogger('faiss').disabled = True
    logging.getLogger('faiss.loader').disabled = True
    
    # Transformers 로깅 비활성화
    logging.getLogger('transformers').disabled = True
    logging.getLogger('transformers.tokenization_utils').disabled = True
    
    # PyTorch 로깅 비활성화
    logging.getLogger('torch').disabled = True
    
    # 기타 외부 라이브러리 로깅 비활성화
    logging.getLogger('urllib3').disabled = True
    logging.getLogger('requests').disabled = True


def setup_script_logging(script_name: str) -> logging.Logger:
    """
    스크립트용 로깅을 설정합니다.
    
    Args:
        script_name: 스크립트 이름
    
    Returns:
        설정된 로거
    """
    # 로그 파일 경로 설정
    log_file = f"logs/{script_name}.log"
    
    # 안전한 로깅 설정
    logger = setup_safe_logging(
        level="INFO",
        log_file=log_file
    )
    
    # 외부 라이브러리 로깅 비활성화
    disable_external_logging()
    
    return logger


if __name__ == "__main__":
    # 테스트
    logger = setup_script_logging("test_safe_logging")
    logger.info("안전한 로깅 테스트 성공!")
    logger.warning("경고 메시지 테스트")
    logger.error("오류 메시지 테스트")
