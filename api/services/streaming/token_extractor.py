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
                    # 🔥 개선: AIMessageChunk의 content 추출 로직 강화
                    # 1. content 속성 직접 접근
                    if hasattr(chunk_obj, "content"):
                        content = chunk_obj.content
                        if isinstance(content, str) and len(content) > 0:
                            token = content
                        elif isinstance(content, list) and len(content) > 0:
                            # 리스트인 경우 첫 번째 요소 사용
                            first_item = content[0]
                            if isinstance(first_item, str):
                                token = first_item
                            else:
                                token = str(first_item) if first_item else None
                        elif content is not None:
                            token = str(content) if str(content) else None
                    
                    # 2. content 속성이 없거나 실패한 경우, 다른 속성 시도
                    if not token:
                        # response_metadata에서 content 추출 시도
                        if hasattr(chunk_obj, "response_metadata"):
                            response_metadata = chunk_obj.response_metadata
                            if isinstance(response_metadata, dict):
                                token = response_metadata.get("content") or response_metadata.get("text")
                        
                        # additional_kwargs에서 content 추출 시도
                        if not token and hasattr(chunk_obj, "additional_kwargs"):
                            additional_kwargs = chunk_obj.additional_kwargs
                            if isinstance(additional_kwargs, dict):
                                token = additional_kwargs.get("content") or additional_kwargs.get("text")
                        
                        # __str__ 또는 __repr__ 사용 (마지막 수단)
                        if not token:
                            try:
                                token_str = str(chunk_obj)
                                if token_str and token_str != str(type(chunk_obj)):
                                    token = token_str
                            except Exception:
                                pass
                except Exception as e:
                    # 예외 발생 시 None 반환 (로그는 상위에서 처리)
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

