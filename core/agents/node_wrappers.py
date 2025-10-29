# -*- coding: utf-8 -*-
"""
노드 함수 래퍼
State Reduction과 Adapter를 자동으로 적용하는 데코레이터 및 헬퍼 함수
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict

from .state_adapter import (
    StateAdapter,
    validate_state_for_node,
)
from .state_reduction import StateReducer

logger = logging.getLogger(__name__)


def with_state_optimization(node_name: str, enable_reduction: bool = True):
    """
    State 최적화를 적용하는 데코레이터

    적용 기능:
    1. Input 검증
    2. State 자동 변환 (flat ↔ nested)
    3. State Reduction (선택적)

    Args:
        node_name: 노드 이름
        enable_reduction: State Reduction 활성화 여부

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                # 0. 인자 처리: 첫 번째 인자가 self인지 확인
                # 바운드 메서드의 경우 self가 이미 바인딩되어 있으므로,
                # LangGraph는 state만 전달해야 하지만, 혹시 모를 상황 대비
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # 첫 번째 인자가 dict가 아닌 경우 (self가 전달된 것으로 간주)
                if len(args) > 1:
                    # args[0]은 self, args[1]은 state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]은 state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. State가 딕셔너리인지 확인
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without optimization."
                    )
                    logger.error(error_msg)
                    # 원본 함수 직접 호출 (최소한의 처리)
                    if len(args) > 1:
                        return func(*args, **kwargs)
                    else:
                        return func(state, *rest_args, **kwargs)

                # 1. Input 검증 및 자동 변환
                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                if not is_valid:
                    logger.warning(f"Input validation failed for {node_name}: {error}")

                # 2. State Reduction (활성화된 경우)
                if enable_reduction:
                    reducer = StateReducer(aggressive_reduction=True)
                    working_state = reducer.reduce_state_for_node(converted_state, node_name)

                    # Reduction 결과가 비어있으면 원본 사용
                    if not working_state:
                        logger.warning(f"State reduction returned empty dict for {node_name}, using converted_state")
                        working_state = converted_state

                    # 상태 크기 로깅
                    if logger.isEnabledFor(logging.DEBUG):
                        original_size = _estimate_state_size(state)
                        reduced_size = _estimate_state_size(working_state)
                        reduction_pct = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0
                        logger.debug(
                            f"State reduction for {node_name}: "
                            f"{reduction_pct:.1f}% reduction "
                            f"({original_size:.0f} → {reduced_size:.0f} bytes)"
                        )
                else:
                    working_state = converted_state

                # 3. 원본 함수 호출
                if len(args) > 1:
                    # self가 있는 경우
                    result = func(args[0], working_state, *rest_args, **kwargs)
                else:
                    # self가 없는 경우
                    result = func(working_state, *rest_args, **kwargs)

                # 4. 결과를 원본 State에 병합
                if isinstance(result, dict) and isinstance(state, dict):
                    # Nested 구조면 그대로 반환
                    if "input" in state and isinstance(state.get("input"), dict):
                        return result
                    else:
                        # Flat 구조면 병합
                        state.update(result)
                        return state

                return result

            except Exception as e:
                logger.error(f"Error in state optimization wrapper for {node_name}: {e}", exc_info=True)
                # 에러 발생 시 원본 함수 실행 (최소한의 처리)
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function call also failed for {node_name}: {fallback_error}",
                        exc_info=True
                    )
                    # 마지막 수단: 빈 딕셔너리 반환
                    return {}

        return wrapper
    return decorator


def _estimate_state_size(state: Dict[str, Any]) -> int:
    """State 크기 추정"""
    import sys
    try:
        return sys.getsizeof(str(state))
    except:
        return len(str(state))


def with_input_validation(node_name: str):
    """
    Input 검증만 적용하는 데코레이터

    State Reduction 없이 Input 검증만 수행
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                # 0. 인자 처리: 첫 번째 인자가 self인지 확인
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # 첫 번째 인자가 dict가 아닌 경우 (self가 전달된 것으로 간주)
                if len(args) > 1:
                    # args[0]은 self, args[1]은 state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]은 state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. State가 딕셔너리인지 확인
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without validation."
                    )
                    logger.error(error_msg)
                    return func(*args, **kwargs)

                # Input 검증 및 자동 변환
                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                if not is_valid:
                    logger.warning(f"Input validation failed for {node_name}: {error}")

                # 원본 함수 호출
                if len(args) > 1:
                    # self가 있는 경우
                    result = func(args[0], converted_state, *rest_args, **kwargs)
                else:
                    # self가 없는 경우
                    result = func(converted_state, *rest_args, **kwargs)

                # 결과 반환
                return result

            except Exception as e:
                logger.error(f"Error in input validation wrapper for {node_name}: {e}", exc_info=True)
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function call also failed for {node_name}: {fallback_error}",
                        exc_info=True
                    )
                    return {}

        return wrapper
    return decorator


def adapt_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    State를 필요시 자동 변환 (편의 함수)

    Args:
        state: State 객체 (flat 또는 nested)

    Returns:
        변환된 State 객체
    """
    return StateAdapter.to_nested(state)


def flatten_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    State를 Flat 구조로 변환 (편의 함수)

    Args:
        state: State 객체 (nested)

    Returns:
        Flat 구조의 State
    """
    return StateAdapter.to_flat(state)
