# -*- coding: utf-8 -*-
"""
Utils Module
유틸리티 모듈

참고: 대부분의 기능이 core/shared/utils/로 이동했습니다.
이 모듈은 호환성을 위해 re-export만 제공합니다.
"""

# core/shared/utils/로 re-export
# 순환 import 방지를 위해 로컬 모듈을 우선 사용
try:
    from .config import Config
    from .logger import setup_logging, get_logger
    from .korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    # Fallback: core.shared.utils 경로
    try:
        from core.shared.utils.config import Config
        from core.shared.utils.logger import setup_logging, get_logger
        from .korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        # 최종 Fallback: 직접 정의
        try:
            from .korean_stopword_processor import KoreanStopwordProcessor
        except ImportError:
            pass

__all__ = [
    "Config",
    "setup_logging",
    "get_logger",
    "KoreanStopwordProcessor"
]
