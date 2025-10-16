# -*- coding: utf-8 -*-
"""
Assembly Logger
로깅 설정 모듈

구조화된 로깅을 제공합니다.
- 파일 및 콘솔 출력
- JSON 구조화 로그
- 로그 레벨 설정
- 로그 회전
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON 형태의 로그 포맷터"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 예외 정보 추가
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 추가 필드 추가
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """컬러가 적용된 콘솔 포맷터"""
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 자주색
        'RESET': '\033[0m'      # 리셋
    }
    
    def format(self, record):
        # 색상 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 포맷 문자열에 색상 추가
        colored_format = f"{color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}"
        
        formatter = logging.Formatter(colored_format)
        return formatter.format(record)


def setup_logging(
    log_name: str = "assembly_collection",
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_name: 로거 이름
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일 저장 디렉토리
        console_output: 콘솔 출력 여부
        file_output: 파일 출력 여부
        json_format: JSON 형태 로그 여부
        max_file_size: 로그 파일 최대 크기 (바이트)
        backup_count: 백업 파일 개수
    
    Returns:
        logging.Logger: 설정된 로거
    """
    
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 콘솔 핸들러
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter()
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 파일 핸들러
    if file_output:
        # 일반 텍스트 로그 파일
        log_file = log_path / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # JSON 로그 파일 (선택적)
        if json_format:
            json_log_file = log_path / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(getattr(logging, log_level.upper()))
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
    
    # 로그 설정 완료 메시지 (간단히)
    print(f"✅ Logging configured: {log_name} ({log_level})")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기존 로거 가져오기
    
    Args:
        name: 로거 이름
    
    Returns:
        logging.Logger: 로거
    """
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    추가 필드와 함께 로그 기록
    
    Args:
        logger: 로거
        level: 로그 레벨
        message: 로그 메시지
        **kwargs: 추가 필드
    """
    extra_fields = kwargs
    getattr(logger, level.lower())(message, extra={'extra_fields': extra_fields})


# 편의 함수들
def log_progress(logger: logging.Logger, current: int, total: int, item_name: str = "items"):
    """진행률 로그"""
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"📊 Progress: {current}/{total} {item_name} ({percentage:.1f}%)")


def log_memory_usage(logger: logging.Logger, memory_mb: float, limit_mb: float):
    """메모리 사용량 로그"""
    percentage = (memory_mb / limit_mb) * 100 if limit_mb > 0 else 0
    status = "⚠️" if percentage > 80 else "✅"
    logger.info(f"{status} Memory: {memory_mb:.1f}MB / {limit_mb}MB ({percentage:.1f}%)")


def log_collection_stats(logger: logging.Logger, collected: int, failed: int, total: int):
    """수집 통계 로그"""
    success_rate = (collected / total) * 100 if total > 0 else 0
    logger.info(f"📈 Collection stats: {collected} collected, {failed} failed, {success_rate:.1f}% success rate")


def log_checkpoint_info(logger: logging.Logger, checkpoint_data: dict):
    """체크포인트 정보 로그"""
    logger.info(f"💾 Checkpoint info:")
    logger.info(f"   Data type: {checkpoint_data.get('data_type', 'unknown')}")
    logger.info(f"   Category: {checkpoint_data.get('category', 'None')}")
    logger.info(f"   Page: {checkpoint_data.get('current_page', 0)}/{checkpoint_data.get('total_pages', 0)}")
    logger.info(f"   Collected: {checkpoint_data.get('collected_count', 0)} items")
    logger.info(f"   Memory: {checkpoint_data.get('memory_usage_mb', 0):.1f}MB")
