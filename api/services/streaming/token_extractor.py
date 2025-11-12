"""
이벤트에서 토큰 추출 전용 클래스
"""
from typing import Dict, Any, Optional


class TokenExtractor:
    """이벤트에서 토큰 추출 전용 클래스"""
    
    @staticmethod
    def extract_from_event(event_data: Dict[str, Any]) -> Optional[str]:
        """이벤트에서 토큰 추출"""
        chunk_obj = event_data.get("chunk")
        token = None
        
        if chunk_obj:
            if hasattr(chunk_obj, "content"):
                content = chunk_obj.content
                if isinstance(content, str):
                    token = content
                elif isinstance(content, list) and len(content) > 0:
                    token = content[0] if isinstance(content[0], str) else str(content[0])
                else:
                    token = str(content) if content else None
            elif isinstance(chunk_obj, str):
                token = chunk_obj
            elif isinstance(chunk_obj, dict):
                token = chunk_obj.get("content") or chunk_obj.get("text")
            elif hasattr(chunk_obj, "text"):
                token = chunk_obj.text
            elif hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
                try:
                    content = getattr(chunk_obj, "content", None)
                    if isinstance(content, str):
                        token = content
                    elif isinstance(content, list) and len(content) > 0:
                        token = content[0] if isinstance(content[0], str) else str(content[0])
                    elif content is not None:
                        token = str(content)
                except Exception:
                    token = None
            else:
                token = str(chunk_obj) if chunk_obj else None
        
        if not token and "delta" in event_data:
            delta = event_data["delta"]
            if isinstance(delta, dict):
                token = delta.get("content") or delta.get("text")
            elif isinstance(delta, str):
                token = delta
        
        return token if isinstance(token, str) and len(token) > 0 else None

