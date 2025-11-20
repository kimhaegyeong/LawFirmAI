# -*- coding: utf-8 -*-
"""
지연 로딩 유틸리티 모듈
모듈 import 최적화를 위한 지연 로딩 헬퍼
"""

import sys
from typing import Any, Optional, Callable

# 모듈 캐시
_module_cache: dict[str, Any] = {}


def lazy_import(
    module_path: str,
    fallback_path: Optional[str] = None,
    default: Any = None,
    cache_key: Optional[str] = None
) -> Any:
    """
    지연 로딩을 위한 모듈 import 헬퍼
    
    Args:
        module_path: 메인 모듈 경로
        fallback_path: 폴백 모듈 경로 (선택사항)
        default: import 실패 시 기본값
        cache_key: 캐시 키 (None이면 module_path 사용)
    
    Returns:
        로드된 모듈 또는 default 값
    """
    cache_key = cache_key or module_path
    
    if cache_key in _module_cache:
        return _module_cache[cache_key]
    
    # 메인 경로 시도
    try:
        module = __import__(module_path, fromlist=[''])
        _module_cache[cache_key] = module
        return module
    except ImportError:
        pass
    
    # 폴백 경로 시도
    if fallback_path:
        try:
            module = __import__(fallback_path, fromlist=[''])
            _module_cache[cache_key] = module
            return module
        except ImportError:
            pass
    
    # 모두 실패 시 기본값 반환
    _module_cache[cache_key] = default
    return default


def lazy_getattr(module: Any, attr_name: str, default: Any = None) -> Any:
    """
    모듈에서 속성을 지연 로딩으로 가져오기
    
    Args:
        module: 모듈 객체
        attr_name: 속성 이름
        default: 기본값
    
    Returns:
        속성 값 또는 default
    """
    if module is None:
        return default
    
    try:
        return getattr(module, attr_name, default)
    except AttributeError:
        return default


def clear_cache():
    """모듈 캐시 초기화"""
    global _module_cache
    _module_cache.clear()

