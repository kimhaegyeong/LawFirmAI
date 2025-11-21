# -*- coding: utf-8 -*-
"""
워크플로우 에러 처리 데코레이터
노드 메서드의 에러를 통일된 방식으로 처리
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from functools import wraps
from typing import Any, Callable, Optional

from core.agents.state_definitions import LegalWorkflowState
from core.workflow.utils.workflow_utils import WorkflowUtils

logger = get_logger(__name__)


def handle_workflow_errors(
    fallback_return_value: Optional[Any] = None,
    error_message: Optional[str] = None,
    log_level: str = "error"
):
    """
    워크플로우 노드 에러 처리 데코레이터
    
    Args:
        fallback_return_value: 에러 발생 시 반환할 값 (None이면 state 반환)
        error_message: 커스텀 에러 메시지
        log_level: 로그 레벨 ("debug", "info", "warning", "error", "critical")
    
    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, state: LegalWorkflowState) -> LegalWorkflowState:
            try:
                return func(self, state)
            except Exception as e:
                # 로거 가져오기
                node_logger = getattr(self, 'logger', logger)
                
                # 로그 레벨에 따라 로깅
                log_message = error_message or f"Error in {func.__name__}: {str(e)}"
                
                if log_level == "debug":
                    node_logger.debug(log_message, exc_info=True)
                elif log_level == "info":
                    node_logger.info(log_message, exc_info=True)
                elif log_level == "warning":
                    node_logger.warning(log_message, exc_info=True)
                elif log_level == "critical":
                    node_logger.critical(log_message, exc_info=True)
                else:
                    node_logger.error(log_message, exc_info=True)
                
                # 에러 처리
                if hasattr(self, '_handle_error'):
                    self._handle_error(state, str(e), f"{func.__name__} 실행 중 오류 발생")
                else:
                    WorkflowUtils.handle_error(state, str(e), f"{func.__name__} 실행 중 오류 발생", node_logger)
                
                # 폴백 값 반환
                if fallback_return_value is not None:
                    return fallback_return_value
                
                # 기본 폴백: state 반환
                return state
        
        return wrapper
    return decorator

