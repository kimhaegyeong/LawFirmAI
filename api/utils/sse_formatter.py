"""
SSE 형식 변환 유틸리티
Server-Sent Events 형식으로 이벤트를 포맷팅하는 유틸리티 함수
"""
import json
from typing import Dict, Any


def format_sse_event(event_data: Dict[str, Any]) -> str:
    """
    SSE 형식으로 이벤트 포맷팅
    
    Args:
        event_data: 이벤트 데이터 딕셔너리
    
    Returns:
        SSE 형식 문자열: "data: {json}\n\n"
    
    Example:
        >>> event = {"type": "stream", "content": "안녕하세요"}
        >>> format_sse_event(event)
        'data: {"type":"stream","content":"안녕하세요"}\\n\\n'
    """
    try:
        # JSON 직렬화 전에 데이터 정리
        cleaned_data = _clean_for_json_serialization(event_data)
        return f"data: {json.dumps(cleaned_data, ensure_ascii=False)}\n\n"
    except (TypeError, ValueError) as e:
        # JSON 직렬화 실패 시 최소한의 정보만 포함
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[format_sse_event] JSON serialization failed: {e}", exc_info=True)
        # 안전한 기본 이벤트 반환
        safe_event = {
            "type": event_data.get("type", "error"),
            "error": "Failed to serialize event data"
        }
        return f"data: {json.dumps(safe_event, ensure_ascii=False)}\n\n"


def _clean_for_json_serialization(obj: Any) -> Any:
    """JSON 직렬화를 위해 데이터 정리"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            try:
                # 키는 문자열로 변환
                str_key = str(key)
                # 값은 재귀적으로 정리
                cleaned[str_key] = _clean_for_json_serialization(value)
            except Exception:
                # 변환 실패 시 해당 항목 건너뛰기
                continue
        return cleaned
    if isinstance(obj, (list, tuple)):
        cleaned = []
        for item in obj:
            try:
                cleaned.append(_clean_for_json_serialization(item))
            except Exception:
                # 변환 실패 시 해당 항목 건너뛰기
                continue
        return cleaned
    # 기타 타입은 문자열로 변환 시도
    try:
        return str(obj)
    except Exception:
        return None

