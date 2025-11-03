# -*- coding: utf-8 -*-
"""
워크플로우 전용 로거 설정
검색 관련 상세 로그를 파일로 저장
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkflowFileLogger:
    """워크플로우 전용 파일 로거"""

    def __init__(self, log_dir: str = "logs", log_prefix: str = "workflow"):
        """
        워크플로우 파일 로거 초기화

        Args:
            log_dir: 로그 파일 저장 디렉토리
            log_prefix: 로그 파일 이름 접두사
        """
        self.log_dir = Path(log_dir)
        self.log_prefix = log_prefix
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_prefix}_{timestamp}.log"

        # 파일 핸들러 설정
        self.file_handler = None
        self._setup_file_handler()

    def _setup_file_handler(self):
        """파일 핸들러 설정"""
        if self.file_handler:
            return

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.file_handler = logging.FileHandler(
            self.log_file,
            encoding='utf-8',
            mode='a'
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)

    def attach_to_logger(self, logger_name: str):
        """
        특정 로거에 파일 핸들러 연결

        Args:
            logger_name: 로거 이름 (예: 'core.agents.legal_workflow_enhanced')
        """
        logger = logging.getLogger(logger_name)
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(f"File logger attached: {self.log_file}")

    def attach_to_search_loggers(self):
        """검색 관련 주요 로거에 파일 핸들러 연결"""
        search_loggers = [
            'core.agents.legal_workflow_enhanced',
            'core.agents.legal_data_connector_v2',
            'source.services.semantic_search_engine_v2',
            'source.services.exact_search_engine_v2',
            'source.services.hybrid_search_engine_v2',
        ]

        for logger_name in search_loggers:
            self.attach_to_logger(logger_name)

    def get_log_file_path(self) -> Path:
        """로그 파일 경로 반환"""
        return self.log_file

    def close(self):
        """파일 핸들러 닫기"""
        if self.file_handler:
            self.file_handler.close()


def setup_workflow_file_logging(
    log_dir: str = "logs",
    log_prefix: str = "workflow",
    attach_search_loggers: bool = True
) -> WorkflowFileLogger:
    """
    워크플로우 파일 로깅 설정

    Args:
        log_dir: 로그 디렉토리
        log_prefix: 로그 파일 접두사
        attach_search_loggers: 검색 관련 로거에 자동 연결 여부

    Returns:
        WorkflowFileLogger 인스턴스
    """
    file_logger = WorkflowFileLogger(log_dir=log_dir, log_prefix=log_prefix)

    if attach_search_loggers:
        file_logger.attach_to_search_loggers()

    return file_logger
