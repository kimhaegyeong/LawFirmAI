# -*- coding: utf-8 -*-
"""
Shared Module
공유 유틸리티 모듈
"""

from .utils import logger, config

# 순환 import 방지를 위해 node_wrappers는 lazy import
# from .wrappers import node_wrappers  # 주석 처리

try:
    from .utils import date_utils
    __all__ = [
        "logger",
        "config",
        "date_utils",
        # "node_wrappers",  # 주석 처리
    ]
except ImportError:
    __all__ = [
        "logger",
        "config",
        # "node_wrappers",  # 주석 처리
    ]

# node_wrappers는 필요시 직접 import하도록 변경
def get_node_wrappers():
    """node_wrappers를 lazy import하는 함수"""
    from .wrappers import node_wrappers
    return node_wrappers

