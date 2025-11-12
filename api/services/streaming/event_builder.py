"""
스트림 이벤트 생성 유틸리티
"""
from datetime import datetime
from typing import Dict, Any, Optional


class StreamEventBuilder:
    """스트림 이벤트 생성 유틸리티"""
    
    @staticmethod
    def create_stream_event(
        content: str, 
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """스트림 이벤트 생성"""
        event = {
            "type": "stream",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if source:
            event["source"] = source
        return event
    
    @staticmethod
    def create_error_event(
        content: str, 
        error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """에러 이벤트 생성"""
        event = {
            "type": "error",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if error_type:
            event["error_type"] = error_type
        return event
    
    @staticmethod
    def create_progress_event(content: str) -> Dict[str, Any]:
        """진행 상황 이벤트 생성"""
        return {
            "type": "progress",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_final_event(
        content: str,
        metadata: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """최종 이벤트 생성"""
        if message_id:
            metadata = {**metadata, "message_id": message_id}
        return {
            "type": "final",
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_validation_event(validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """검증 이벤트 생성"""
        return {
            "type": "validation",
            "content": "답변 품질 검증 완료",
            "metadata": {
                "quality_score": validation_result.get("quality_score", 0.0),
                "is_valid": validation_result.get("is_valid", False),
                "needs_regeneration": validation_result.get("needs_regeneration", False),
                "regeneration_reason": validation_result.get("regeneration_reason"),
                "issues": validation_result.get("issues", []),
                "strengths": validation_result.get("strengths", [])
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_done_event(
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """완료 이벤트 생성"""
        return {
            "type": "done",
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

