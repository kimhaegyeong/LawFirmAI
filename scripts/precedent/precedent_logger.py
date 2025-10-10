#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 수집 관련 로깅 유틸리티
"""

import logging
import sys
import io
import os
from datetime import datetime
from pathlib import Path

# 한글 출력을 위한 인코딩 설정
if sys.platform == "win32":
    # Windows에서 한글 출력을 위한 설정
    try:
        import codecs
        # Windows 콘솔에서 UTF-8 지원 확인 후 설정
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        else:
            # detach()가 지원되지 않는 경우 다른 방법 사용
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        # 인코딩 설정 실패 시 기본 설정 유지
        pass
    
    # 환경변수 인코딩 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """로깅 설정을 초기화하고 로거를 반환합니다."""
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 포맷 설정 (이모지 제거)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 파일 핸들러 (일별 로테이션, UTF-8 인코딩)
    log_file = log_dir / f"collect_precedents_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 콘솔 핸들러 설정 (Windows 한글 출력 개선)
    if sys.platform == "win32":
        try:
            # Windows 콘솔 인코딩을 UTF-8로 설정
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            
            # Windows에서 한글 출력을 위한 특별한 스트림 핸들러
            import io
            console_handler = logging.StreamHandler(
                io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            )
        except Exception:
            # 실패 시 기본 핸들러 사용
            console_handler = logging.StreamHandler(sys.stdout)
    else:
        # Linux/Mac에서는 기본 핸들러 사용
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 중복 핸들러 방지
    logger.propagate = False
    
    return logger
