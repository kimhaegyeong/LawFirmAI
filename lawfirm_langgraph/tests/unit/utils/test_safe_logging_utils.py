# -*- coding: utf-8 -*-
"""
Safe Logging Utils 테스트
core/utils/safe_logging_utils.py 단위 테스트
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch

from lawfirm_langgraph.core.utils.safe_logging_utils import (
    _is_handler_valid,
    _has_valid_handlers,
    safe_log_debug,
    safe_log_info,
    safe_log_warning,
    safe_log_error
)


class TestIsHandlerValid:
    """_is_handler_valid 테스트"""
    
    def test_is_handler_valid_none(self):
        """None 핸들러 테스트"""
        assert _is_handler_valid(None) is False
    
    def test_is_handler_valid_stream_handler_valid(self):
        """유효한 StreamHandler 테스트"""
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        
        assert _is_handler_valid(handler) is True
    
    def test_is_handler_valid_stream_handler_closed(self):
        """닫힌 StreamHandler 테스트"""
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = True
        
        assert _is_handler_valid(handler) is False
    
    def test_is_handler_valid_stream_handler_not_writable(self):
        """쓰기 불가능한 StreamHandler 테스트"""
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=False)
        
        assert _is_handler_valid(handler) is False
    
    def test_is_handler_valid_file_handler_valid(self):
        """유효한 FileHandler 테스트"""
        handler = logging.FileHandler('/dev/null')
        handler.stream = Mock()
        handler.stream.closed = False
        
        assert _is_handler_valid(handler) is True
    
    def test_is_handler_valid_file_handler_closed(self):
        """닫힌 FileHandler 테스트"""
        handler = logging.FileHandler('/dev/null')
        handler.stream = Mock()
        handler.stream.closed = True
        
        assert _is_handler_valid(handler) is False
    
    def test_is_handler_valid_exception_handling(self):
        """예외 처리 테스트"""
        handler = Mock()
        handler.stream = Mock()
        handler.stream.writable = Mock(side_effect=ValueError("Test error"))
        
        assert _is_handler_valid(handler) is False


class TestHasValidHandlers:
    """_has_valid_handlers 테스트"""
    
    def test_has_valid_handlers_with_valid_handler(self):
        """유효한 핸들러가 있는 경우 테스트"""
        logger = logging.getLogger("test_logger")
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        logger.addHandler(handler)
        
        assert _has_valid_handlers(logger) is True
    
    def test_has_valid_handlers_without_handler(self):
        """핸들러가 없는 경우 테스트"""
        logger = logging.getLogger("test_logger_no_handler")
        logger.handlers = []
        
        assert _has_valid_handlers(logger) is False
    
    def test_has_valid_handlers_with_invalid_handler(self):
        """유효하지 않은 핸들러만 있는 경우 테스트"""
        logger = logging.getLogger("test_logger_invalid")
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = True
        logger.addHandler(handler)
        
        assert _has_valid_handlers(logger) is False
    
    def test_has_valid_handlers_exception_handling(self):
        """예외 처리 테스트"""
        logger = Mock()
        logger.handlers = [Mock()]
        logger.parent = None
        logger.handlers[0].stream = Mock(side_effect=ValueError("Test error"))
        
        assert _has_valid_handlers(logger) is False


class TestSafeLogDebug:
    """safe_log_debug 테스트"""
    
    def test_safe_log_debug_enabled(self):
        """디버그 로깅 활성화 테스트"""
        logger = logging.getLogger("test_debug")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        logger.addHandler(handler)
        
        with patch.object(logger, 'debug') as mock_debug:
            safe_log_debug(logger, "Test debug message")
            mock_debug.assert_called_once_with("Test debug message")
    
    def test_safe_log_debug_disabled(self):
        """디버그 로깅 비활성화 테스트"""
        logger = logging.getLogger("test_debug_disabled")
        logger.setLevel(logging.INFO)
        
        with patch.object(logger, 'debug') as mock_debug:
            safe_log_debug(logger, "Test debug message")
            mock_debug.assert_not_called()
    
    def test_safe_log_debug_no_handler(self):
        """핸들러가 없는 경우 테스트"""
        logger = logging.getLogger("test_debug_no_handler")
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        
        with patch.object(logger, 'debug') as mock_debug:
            safe_log_debug(logger, "Test debug message")
            mock_debug.assert_not_called()
    
    def test_safe_log_debug_exception_handling(self):
        """예외 처리 테스트"""
        logger = logging.getLogger("test_debug_exception")
        logger.setLevel(logging.DEBUG)
        handler = Mock()
        handler.stream = Mock(side_effect=ValueError("Test error"))
        logger.addHandler(handler)
        
        safe_log_debug(logger, "Test debug message")


class TestSafeLogInfo:
    """safe_log_info 테스트"""
    
    def test_safe_log_info_enabled(self):
        """정보 로깅 활성화 테스트"""
        logger = logging.getLogger("test_info")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        logger.addHandler(handler)
        
        with patch.object(logger, 'info') as mock_info:
            safe_log_info(logger, "Test info message")
            mock_info.assert_called_once_with("Test info message")
    
    def test_safe_log_info_disabled(self):
        """정보 로깅 비활성화 테스트"""
        logger = logging.getLogger("test_info_disabled")
        logger.setLevel(logging.WARNING)
        
        with patch.object(logger, 'info') as mock_info:
            safe_log_info(logger, "Test info message")
            mock_info.assert_not_called()
    
    def test_safe_log_info_exception_handling(self):
        """예외 처리 테스트"""
        logger = logging.getLogger("test_info_exception")
        logger.setLevel(logging.INFO)
        handler = Mock()
        handler.stream = Mock(side_effect=ValueError("Test error"))
        logger.addHandler(handler)
        
        safe_log_info(logger, "Test info message")


class TestSafeLogWarning:
    """safe_log_warning 테스트"""
    
    def test_safe_log_warning_enabled(self):
        """경고 로깅 활성화 테스트"""
        logger = logging.getLogger("test_warning")
        logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        logger.addHandler(handler)
        
        with patch.object(logger, 'warning') as mock_warning:
            safe_log_warning(logger, "Test warning message")
            mock_warning.assert_called_once_with("Test warning message")
    
    def test_safe_log_warning_disabled(self):
        """경고 로깅 비활성화 테스트"""
        logger = logging.getLogger("test_warning_disabled")
        logger.setLevel(logging.ERROR)
        
        with patch.object(logger, 'warning') as mock_warning:
            safe_log_warning(logger, "Test warning message")
            mock_warning.assert_not_called()
    
    def test_safe_log_warning_exception_handling(self):
        """예외 처리 테스트"""
        logger = logging.getLogger("test_warning_exception")
        logger.setLevel(logging.WARNING)
        handler = Mock()
        handler.stream = Mock(side_effect=ValueError("Test error"))
        logger.addHandler(handler)
        
        safe_log_warning(logger, "Test warning message")


class TestSafeLogError:
    """safe_log_error 테스트"""
    
    def test_safe_log_error_enabled(self):
        """오류 로깅 활성화 테스트"""
        logger = logging.getLogger("test_error")
        logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler()
        handler.stream = Mock()
        handler.stream.closed = False
        handler.stream.writable = Mock(return_value=True)
        logger.addHandler(handler)
        
        with patch.object(logger, 'error') as mock_error:
            safe_log_error(logger, "Test error message")
            mock_error.assert_called_once_with("Test error message")
    
    def test_safe_log_error_disabled(self):
        """오류 로깅 비활성화 테스트"""
        logger = logging.getLogger("test_error_disabled")
        logger.setLevel(logging.CRITICAL)
        
        with patch.object(logger, 'error') as mock_error:
            safe_log_error(logger, "Test error message")
            mock_error.assert_not_called()
    
    def test_safe_log_error_exception_handling(self):
        """예외 처리 테스트"""
        logger = logging.getLogger("test_error_exception")
        logger.setLevel(logging.ERROR)
        handler = Mock()
        handler.stream = Mock(side_effect=ValueError("Test error"))
        logger.addHandler(handler)
        
        safe_log_error(logger, "Test error message")

