# -*- coding: utf-8 -*-
"""
Logger 테스트
유틸리티 logger 모듈 단위 테스트
"""

import pytest
import logging
import os
from unittest.mock import patch, MagicMock

from lawfirm_langgraph.core.utils.logger import get_logger, setup_logging
from lawfirm_langgraph.core.utils.config import Config


class TestLogger:
    """Logger 테스트"""
    
    def test_get_logger_default(self):
        """기본 로거 생성 테스트"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_with_env_level(self, monkeypatch):
        """환경 변수로 로깅 레벨 설정 테스트"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        logger = get_logger("test_module")
        
        assert logger.level == logging.DEBUG
    
    def test_get_logger_with_different_levels(self, monkeypatch):
        """다양한 로깅 레벨 테스트"""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            monkeypatch.setenv("LOG_LEVEL", level)
            logger = get_logger(f"test_module_{level}")
            
            assert isinstance(logger, logging.Logger)
            assert logger.level == getattr(logging, level)
    
    def test_get_logger_invalid_level(self, monkeypatch):
        """잘못된 로깅 레벨 테스트"""
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
    
    def test_get_logger_propagate(self):
        """로거 전파 테스트"""
        logger = get_logger("test_module")
        
        assert logger.propagate is True
    
    def test_setup_logging(self):
        """로깅 설정 테스트"""
        config = Config()
        
        setup_logging(config)
        
        assert True
    
    def test_setup_logging_with_none_config(self):
        """None 설정으로 로깅 설정 테스트"""
        setup_logging(None)
        
        assert True
    
    def test_logger_logging(self):
        """로거 로깅 테스트"""
        logger = get_logger("test_module")
        
        with patch.object(logger, 'info') as mock_info:
            logger.info("테스트 메시지")
            mock_info.assert_called_once_with("테스트 메시지")
    
    def test_logger_error_logging(self):
        """로거 에러 로깅 테스트"""
        logger = get_logger("test_module")
        
        with patch.object(logger, 'error') as mock_error:
            logger.error("테스트 에러")
            mock_error.assert_called_once_with("테스트 에러")
    
    def test_logger_warning_logging(self):
        """로거 경고 로깅 테스트"""
        logger = get_logger("test_module")
        
        with patch.object(logger, 'warning') as mock_warning:
            logger.warning("테스트 경고")
            mock_warning.assert_called_once_with("테스트 경고")

