#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로깅 유틸리티

법령용어 수집 시스템의 로깅을 관리하는 모듈입니다.
- 파일 기반 로깅 설정
- 로그 로테이션 관리
- 로그 레벨 설정
- 로그 포맷팅
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def setup_collection_logger(name: str, log_dir: str = "logs/legal_term_collection", 
                           level: str = "INFO", max_file_size_mb: int = 10, 
                           backup_count: int = 5) -> logging.Logger:
    """
    수집 로거 설정
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리
        level: 로그 레벨
        max_file_size_mb: 최대 파일 크기 (MB)
        backup_count: 백업 파일 수
        
    Returns:
        설정된 로거
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 파일 핸들러 (로테이션 지원) - Windows 권한 문제 해결
    import time
    timestamp = int(time.time())
    log_file = log_path / f"collection_{datetime.now().strftime('%Y%m%d')}_{timestamp}.log"
    
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size_mb * 1024 * 1024,  # MB를 바이트로 변환
            backupCount=backup_count,
            encoding='utf-8',
            delay=True  # 파일을 실제로 사용할 때까지 열기 지연
        )
    except (PermissionError, OSError) as e:
        # 권한 문제 시 간단한 파일 핸들러 사용
        print(f"⚠️ 로그 파일 권한 문제로 간단한 파일 핸들러 사용: {e}")
        try:
            file_handler = logging.FileHandler(
                log_file,
                encoding='utf-8',
                mode='a'  # 추가 모드
            )
        except (PermissionError, OSError) as e2:
            # 파일 핸들러도 실패하면 콘솔만 사용
            print(f"⚠️ 파일 핸들러도 실패하여 콘솔만 사용: {e2}")
            file_handler = None
    
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_scheduler_logger(name: str, log_dir: str = "logs/legal_term_collection",
                          level: str = "INFO") -> logging.Logger:
    """
    스케줄러 로거 설정
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리
        level: 로그 레벨
        
    Returns:
        설정된 로거
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 파일 핸들러 - Windows 권한 문제 해결
    log_file = log_path / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    except (PermissionError, OSError) as e:
        # 권한 문제 시 콘솔만 사용
        print(f"⚠️ 스케줄러 로그 파일 권한 문제로 콘솔만 사용: {e}")
        file_handler = None
    
    if file_handler:
        file_handler.setLevel(getattr(logging, level.upper()))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if file_handler:
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_error_logger(name: str, log_dir: str = "logs/legal_term_collection") -> logging.Logger:
    """
    에러 전용 로거 설정
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리
        
    Returns:
        설정된 로거
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 에러 로그 파일 - Windows 권한 문제 해결
    error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    
    try:
        file_handler = logging.FileHandler(error_log_file, encoding='utf-8', mode='a')
    except (PermissionError, OSError) as e:
        # 권한 문제 시 콘솔만 사용
        print(f"⚠️ 에러 로그 파일 권한 문제로 콘솔만 사용: {e}")
        file_handler = None
    
    if file_handler:
        file_handler.setLevel(logging.ERROR)
    
    # 콘솔 핸들러 (에러만)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if file_handler:
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class CollectionLogger:
    """수집 작업 전용 로거"""
    
    def __init__(self, name: str, log_dir: str = "logs/legal_term_collection"):
        self.logger = setup_collection_logger(name, log_dir)
        self.error_logger = setup_error_logger(f"{name}_error", log_dir)
    
    def info(self, message: str):
        """정보 로그"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """경고 로그"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """에러 로그"""
        self.logger.error(message)
        self.error_logger.error(message)
    
    def debug(self, message: str):
        """디버그 로그"""
        self.logger.debug(message)
    
    def log_collection_start(self, data_type: str, mode: str = "incremental"):
        """수집 시작 로그"""
        self.info(f"수집 시작 - 타입: {data_type}, 모드: {mode}")
    
    def log_collection_end(self, data_type: str, result: Dict[str, Any]):
        """수집 완료 로그"""
        summary = result.get("summary", {})
        self.info(f"수집 완료 - 타입: {data_type}, "
                 f"새로운 레코드: {summary.get('new_count', 0)}개, "
                 f"업데이트: {summary.get('updated_count', 0)}개, "
                 f"삭제: {summary.get('deleted_count', 0)}개")
    
    def log_collection_error(self, data_type: str, error: Exception):
        """수집 에러 로그"""
        self.error(f"수집 실패 - 타입: {data_type}, 에러: {str(error)}")
    
    def log_api_request(self, api_name: str, success: bool, duration: float = None):
        """API 요청 로그"""
        status = "성공" if success else "실패"
        duration_str = f", 소요시간: {duration:.2f}초" if duration else ""
        self.info(f"API 요청 - {api_name}: {status}{duration_str}")
    
    def log_progress(self, current: int, total: int, item_type: str = "items"):
        """진행률 로그"""
        if total > 0:
            percentage = (current / total) * 100
            self.info(f"진행률: {current}/{total} {item_type} ({percentage:.1f}%)")


def get_log_files(log_dir: str = "logs/legal_term_collection") -> Dict[str, list]:
    """
    로그 파일 목록 조회
    
    Args:
        log_dir: 로그 디렉토리
        
    Returns:
        로그 파일 목록 (타입별로 분류)
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return {"collection": [], "scheduler": [], "errors": []}
    
    log_files = {
        "collection": [],
        "scheduler": [],
        "errors": []
    }
    
    for log_file in log_path.glob("*.log"):
        file_name = log_file.name
        
        if file_name.startswith("collection_"):
            log_files["collection"].append(str(log_file))
        elif file_name.startswith("scheduler_"):
            log_files["scheduler"].append(str(log_file))
        elif file_name.startswith("errors_"):
            log_files["errors"].append(str(log_file))
    
    # 각 타입별로 최신순 정렬
    for log_type in log_files:
        log_files[log_type].sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    return log_files


def cleanup_old_logs(log_dir: str = "logs/legal_term_collection", days: int = 30):
    """
    오래된 로그 파일 정리
    
    Args:
        log_dir: 로그 디렉토리
        days: 보관할 일수
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in log_path.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"로그 파일 삭제 실패: {log_file}, 에러: {e}")
    
    if deleted_count > 0:
        print(f"오래된 로그 파일 {deleted_count}개 삭제 완료")


if __name__ == "__main__":
    # 테스트 실행
    print("로깅 유틸리티 테스트")
    print("=" * 40)
    
    # 로거 설정
    logger = setup_collection_logger("test_logger", "test_logs")
    
    # 로그 테스트
    logger.info("정보 로그 테스트")
    logger.warning("경고 로그 테스트")
    logger.error("에러 로그 테스트")
    logger.debug("디버그 로그 테스트")
    
    # 수집 로거 테스트
    collection_logger = CollectionLogger("test_collection", "test_logs")
    collection_logger.log_collection_start("legal_terms", "incremental")
    collection_logger.log_api_request("get_legal_term_list", True, 1.5)
    collection_logger.log_progress(50, 100, "용어")
    
    result = {
        "summary": {
            "new_count": 5,
            "updated_count": 3,
            "deleted_count": 1
        }
    }
    collection_logger.log_collection_end("legal_terms", result)
    
    # 로그 파일 목록 조회
    log_files = get_log_files("test_logs")
    print(f"로그 파일 목록: {log_files}")
    
    print("✅ 로깅 유틸리티 테스트 완료")




