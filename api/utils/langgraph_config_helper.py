"""
LangGraph Config 생성 유틸리티
최신 LangGraph API와 하위 호환성을 유지하면서 config를 생성합니다.
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def create_langgraph_config(
    session_id: str,
    enable_checkpoint: bool = True,
    callbacks: Optional[List[Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    LangGraph config 생성 (최신 API 호환)
    
    Args:
        session_id: 세션 ID (thread_id로 사용)
        enable_checkpoint: 체크포인트 사용 여부
        callbacks: 콜백 핸들러 리스트
        **kwargs: 추가 config 옵션
    
    Returns:
        LangGraph config 딕셔너리
    """
    config = {
        "configurable": {
            "thread_id": session_id  # 최신 LangGraph API에서도 thread_id 사용
        }
    }
    
    # Checkpoint 비활성화 (최신 API에 맞게)
    if not enable_checkpoint:
        # 최신 API: checkpoint_ns를 None으로 설정하여 비활성화
        config["configurable"]["checkpoint_ns"] = None
    
    # 콜백 추가 (있는 경우)
    if callbacks:
        config["callbacks"] = callbacks
    
    # 추가 옵션 병합
    if kwargs:
        # configurable에 추가 옵션 병합
        if "configurable" in kwargs:
            config["configurable"].update(kwargs.pop("configurable"))
        # 최상위 레벨 옵션 병합
        config.update(kwargs)
    
    return config


def create_langgraph_config_with_callbacks(
    session_id: str,
    callbacks: Optional[List[Any]] = None,
    enable_checkpoint: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    콜백이 포함된 LangGraph config 생성
    
    Args:
        session_id: 세션 ID
        callbacks: 콜백 핸들러 리스트
        enable_checkpoint: 체크포인트 사용 여부
        **kwargs: 추가 config 옵션
    
    Returns:
        LangGraph config 딕셔너리
    """
    return create_langgraph_config(
        session_id=session_id,
        enable_checkpoint=enable_checkpoint,
        callbacks=callbacks,
        **kwargs
    )


def validate_langgraph_config(config: Dict[str, Any]) -> bool:
    """
    LangGraph config 유효성 검증
    
    Args:
        config: 검증할 config 딕셔너리
    
    Returns:
        유효하면 True, 그렇지 않으면 False
    """
    if not isinstance(config, dict):
        logger.warning("Config must be a dictionary")
        return False
    
    if "configurable" not in config:
        logger.warning("Config must have 'configurable' key")
        return False
    
    if not isinstance(config["configurable"], dict):
        logger.warning("Config 'configurable' must be a dictionary")
        return False
    
    if "thread_id" not in config["configurable"]:
        logger.warning("Config 'configurable' must have 'thread_id' key")
        return False
    
    return True


def get_thread_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Config에서 thread_id 추출
    
    Args:
        config: LangGraph config 딕셔너리
    
    Returns:
        thread_id 또는 None
    """
    try:
        return config.get("configurable", {}).get("thread_id")
    except (AttributeError, TypeError):
        return None

