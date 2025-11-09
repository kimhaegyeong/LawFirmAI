# -*- coding: utf-8 -*-
"""
안전한 로깅 유틸리티 (멀티스레딩 안전)
"""
import logging
from typing import Optional


def _is_handler_valid(handler: logging.Handler) -> bool:
    """
    핸들러가 유효한지 확인 (스트림 버퍼 분리 상태 체크)
    
    Args:
        handler: 확인할 핸들러
        
    Returns:
        핸들러가 유효하면 True, 아니면 False
    """
    try:
        if not handler:
            return False
        
        # StreamHandler인 경우 스트림 상태 확인
        if isinstance(handler, logging.StreamHandler):
            stream = handler.stream
            if stream is None:
                return False
            # 스트림이 닫혔는지 확인
            if hasattr(stream, 'closed') and stream.closed:
                return False
            # 버퍼가 분리되었는지 확인 (write 메서드로 테스트)
            try:
                # 실제로 쓰지 않고 버퍼 상태만 확인
                if hasattr(stream, 'writable'):
                    if not stream.writable():
                        return False
            except (ValueError, AttributeError, OSError):
                return False
        
        # FileHandler인 경우 파일 상태 확인
        if isinstance(handler, logging.FileHandler):
            if hasattr(handler, 'stream') and handler.stream:
                if hasattr(handler.stream, 'closed') and handler.stream.closed:
                    return False
        
        return True
    except (ValueError, AttributeError, RuntimeError, OSError):
        return False


def _has_valid_handlers(logger: logging.Logger) -> bool:
    """
    로거에 유효한 핸들러가 있는지 확인
    
    Args:
        logger: 확인할 로거
        
    Returns:
        유효한 핸들러가 있으면 True, 없으면 False
    """
    try:
        # 로거 자체의 핸들러 확인
        if logger.handlers:
            for handler in logger.handlers:
                if _is_handler_valid(handler):
                    return True
        
        # 부모 로거의 핸들러 확인
        if logger.parent and logger.parent.handlers:
            for handler in logger.parent.handlers:
                if _is_handler_valid(handler):
                    return True
        
        # 루트 로거의 핸들러 확인
        if logging.root.handlers:
            for handler in logging.root.handlers:
                if _is_handler_valid(handler):
                    return True
        
        return False
    except (ValueError, AttributeError, RuntimeError):
        return False


def safe_log_debug(logger: logging.Logger, message: str) -> None:
    """
    안전한 디버그 로깅 (멀티스레딩 안전)
    
    Args:
        logger: 로거 인스턴스
        message: 로그 메시지
    """
    try:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        # 유효한 핸들러가 있는지 확인
        if not _has_valid_handlers(logger):
            return
        
        # 실제 로깅 시도 (추가 예외 처리)
        try:
            logger.debug(message)
        except (ValueError, AttributeError, RuntimeError, OSError) as e:
            # 버퍼 분리 오류 등은 무시
            if "detached" not in str(e).lower() and "buffer" not in str(e).lower():
                # 다른 종류의 오류는 한 번만 로그 (무한 루프 방지)
                pass
    except (ValueError, AttributeError, RuntimeError, OSError):
        # 로깅 실패 시 무시 (안전한 실패)
        pass


def safe_log_info(logger: logging.Logger, message: str) -> None:
    """
    안전한 정보 로깅 (멀티스레딩 안전)
    
    Args:
        logger: 로거 인스턴스
        message: 로그 메시지
    """
    try:
        if not logger.isEnabledFor(logging.INFO):
            return
        
        # 유효한 핸들러가 있는지 확인
        if not _has_valid_handlers(logger):
            return
        
        # 실제 로깅 시도
        try:
            logger.info(message)
        except (ValueError, AttributeError, RuntimeError, OSError) as e:
            if "detached" not in str(e).lower() and "buffer" not in str(e).lower():
                pass
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass


def safe_log_warning(logger: logging.Logger, message: str) -> None:
    """
    안전한 경고 로깅 (멀티스레딩 안전)
    
    Args:
        logger: 로거 인스턴스
        message: 로그 메시지
    """
    try:
        if not logger.isEnabledFor(logging.WARNING):
            return
        
        # 유효한 핸들러가 있는지 확인
        if not _has_valid_handlers(logger):
            return
        
        # 실제 로깅 시도
        try:
            logger.warning(message)
        except (ValueError, AttributeError, RuntimeError, OSError) as e:
            if "detached" not in str(e).lower() and "buffer" not in str(e).lower():
                pass
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass


def safe_log_error(logger: logging.Logger, message: str) -> None:
    """
    안전한 오류 로깅 (멀티스레딩 안전)
    
    Args:
        logger: 로거 인스턴스
        message: 로그 메시지
    """
    try:
        if not logger.isEnabledFor(logging.ERROR):
            return
        
        # 유효한 핸들러가 있는지 확인
        if not _has_valid_handlers(logger):
            return
        
        # 실제 로깅 시도
        try:
            logger.error(message)
        except (ValueError, AttributeError, RuntimeError, OSError) as e:
            if "detached" not in str(e).lower() and "buffer" not in str(e).lower():
                pass
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

