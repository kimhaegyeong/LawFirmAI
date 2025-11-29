# -*- coding: utf-8 -*-
"""
KoNLPy Okt 싱글톤 유틸리티
Okt 인스턴스를 싱글톤으로 관리하여 중복 초기화를 방지합니다.
"""

from typing import Optional, Any

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)

# 전역 Okt 인스턴스
_global_okt_instance: Optional[Any] = None
_okt_initialized: bool = False


def get_okt_instance() -> Optional[Any]:
    """
    Okt 싱글톤 인스턴스 가져오기
    
    Returns:
        Okt 인스턴스 (KoNLPy가 없으면 None)
    """
    global _global_okt_instance, _okt_initialized
    
    if _okt_initialized:
        return _global_okt_instance
    
    if _global_okt_instance is not None:
        return _global_okt_instance
    
    try:
        from konlpy.tag import Okt
        _global_okt_instance = Okt()
        _okt_initialized = True
        # 최초 초기화 시에만 로그 출력
        logger.debug("KoNLPy Okt initialized successfully (singleton)")
        return _global_okt_instance
    except ImportError as e:
        logger.debug(f"KoNLPy not available (ImportError: {e}), will use fallback method")
        _okt_initialized = True  # 초기화 시도 완료 표시
        return None
    except Exception as e:
        error_msg = str(e)
        # Java 관련 에러인지 확인
        if "java" in error_msg.lower() or "jvm" in error_msg.lower():
            logger.warning(
                f"KoNLPy 초기화 실패 (Java 관련): {e}\n"
                "💡 Java JDK가 설치되어 있는지 확인하세요.\n"
                "   Windows: https://adoptium.net/ 에서 JDK 다운로드\n"
                "   환경 변수 JAVA_HOME이 설정되어 있는지 확인하세요."
            )
        else:
            logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")
        _okt_initialized = True  # 초기화 시도 완료 표시
        return None

