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
    return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

