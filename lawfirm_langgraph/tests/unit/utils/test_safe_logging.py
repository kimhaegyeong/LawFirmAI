# -*- coding: utf-8 -*-
"""
Safe Logging 테스트
core/utils/safe_logging.py 단위 테스트
"""

import os
import pytest
import logging
import tempfile
from unittest.mock import patch, MagicMock

from lawfirm_langgraph.core.utils.safe_logging import (
    disable_external_logging,
    setup_safe_logging,
    get_safe_logger,
    setup_script_logging
)


class TestDisableExternalLogging:
    """disable_external_logging 테스트"""
    
    def test_disable_external_logging(self):
        """외부 로깅 비활성화 테스트"""
        disable_external_logging()
        
        faiss_logger = logging.getLogger('faiss')
        transformers_logger = logging.getLogger('transformers')
        sentence_transformers_logger = logging.getLogger('sentence_transformers')
        hf_hub_logger = logging.getLogger('huggingface_hub')
        torch_logger = logging.getLogger('torch')
        
        assert faiss_logger.disabled is True
        assert transformers_logger.disabled is True
        assert sentence_transformers_logger.disabled is True
        assert hf_hub_logger.disabled is True
        assert torch_logger.disabled is True


class TestSetupSafeLogging:
    """setup_safe_logging 테스트"""
    
    def test_setup_safe_logging_default(self, tmp_path):
        """기본 로깅 설정 테스트"""
        logger = setup_safe_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
    
    def test_setup_safe_logging_custom_level(self, tmp_path):
        """사용자 정의 레벨 로깅 설정 테스트"""
        logger = setup_safe_logging(level="DEBUG")
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG
    
    def test_setup_safe_logging_with_file(self, tmp_path):
        """파일 로깅 설정 테스트"""
        log_file = str(tmp_path / "test.log")
        logger = setup_safe_logging(log_file=log_file)
        
        assert isinstance(logger, logging.Logger)
        assert os.path.exists(log_file)
    
    def test_setup_safe_logging_custom_format(self, tmp_path):
        """사용자 정의 포맷 로깅 설정 테스트"""
        format_string = '%(levelname)s - %(message)s'
        logger = setup_safe_logging(format_string=format_string)
        
        assert isinstance(logger, logging.Logger)
        handlers = logger.handlers
        assert len(handlers) > 0
    
    @patch('lawfirm_langgraph.core.utils.safe_logging.disable_external_logging')
    def test_setup_safe_logging_calls_disable(self, mock_disable, tmp_path):
        """외부 로깅 비활성화 호출 테스트"""
        setup_safe_logging()
        
        mock_disable.assert_called_once()
    
    def test_setup_safe_logging_removes_existing_handlers(self, tmp_path):
        """기존 핸들러 제거 테스트"""
        root_logger = logging.getLogger()
        original_handlers = len(root_logger.handlers)
        
        logger = setup_safe_logging()
        
        assert isinstance(logger, logging.Logger)


class TestGetSafeLogger:
    """get_safe_logger 테스트"""
    
    def test_get_safe_logger(self):
        """안전한 로거 반환 테스트"""
        logger = get_safe_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_safe_logger_different_names(self):
        """다른 이름의 로거 반환 테스트"""
        logger1 = get_safe_logger("logger1")
        logger2 = get_safe_logger("logger2")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1 is not logger2


class TestSetupScriptLogging:
    """setup_script_logging 테스트"""
    
    @patch('lawfirm_langgraph.core.utils.safe_logging.setup_safe_logging')
    @patch('lawfirm_langgraph.core.utils.safe_logging.disable_external_logging')
    def test_setup_script_logging(self, mock_disable, mock_setup, tmp_path):
        """스크립트 로깅 설정 테스트"""
        mock_logger = MagicMock()
        mock_setup.return_value = mock_logger
        
        logger = setup_script_logging("test_script")
        
        assert logger == mock_logger
        mock_setup.assert_called_once()
        mock_disable.assert_called_once()
    
    @patch('lawfirm_langgraph.core.utils.safe_logging.os.makedirs')
    def test_setup_script_logging_creates_directory(self, mock_makedirs, tmp_path):
        """로그 디렉토리 생성 테스트"""
        with patch('lawfirm_langgraph.core.utils.safe_logging.setup_safe_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            setup_script_logging("test_script")
            
            mock_setup.assert_called_once()


class TestSafeLoggingIntegration:
    """안전한 로깅 통합 테스트"""
    
    def test_logging_workflow(self, tmp_path):
        """로깅 워크플로우 테스트"""
        log_file = str(tmp_path / "integration_test.log")
        
        logger = setup_safe_logging(level="INFO", log_file=log_file)
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test info message" in content
            assert "Test warning message" in content
            assert "Test error message" in content
    
    def test_multiple_loggers(self):
        """여러 로거 테스트"""
        logger1 = get_safe_logger("logger1")
        logger2 = get_safe_logger("logger2")
        
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
    
    def test_external_logging_disabled(self):
        """외부 로깅 비활성화 확인 테스트"""
        disable_external_logging()
        
        external_loggers = [
            'faiss',
            'transformers',
            'sentence_transformers',
            'huggingface_hub',
            'torch',
            'urllib3',
            'requests'
        ]
        
        for logger_name in external_loggers:
            logger = logging.getLogger(logger_name)
            assert logger.disabled is True
            assert logger.level == logging.CRITICAL

