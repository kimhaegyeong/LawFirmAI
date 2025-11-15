"""
SSE 포맷터 유틸리티 테스트
"""
import pytest
import sys
import json
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.utils.sse_formatter import format_sse_event


class TestSSEFormatter:
    """SSE 포맷터 테스트"""
    
    def test_format_sse_event_basic(self):
        """기본 SSE 이벤트 포맷팅 테스트"""
        event_data = {"type": "stream", "content": "안녕하세요"}
        result = format_sse_event(event_data)
        
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        
        json_part = result[6:-2]
        parsed = json.loads(json_part)
        assert parsed == event_data
    
    def test_format_sse_event_complex(self):
        """복잡한 데이터 구조 SSE 포맷팅 테스트"""
        event_data = {
            "type": "message",
            "content": "테스트 메시지",
            "metadata": {
                "timestamp": "2024-01-01T00:00:00",
                "user_id": "123"
            }
        }
        result = format_sse_event(event_data)
        
        json_part = result[6:-2]
        parsed = json.loads(json_part)
        assert parsed == event_data
    
    def test_format_sse_event_unicode(self):
        """유니코드 문자 포함 SSE 포맷팅 테스트"""
        event_data = {"content": "한글 테스트 🎉"}
        result = format_sse_event(event_data)
        
        assert "한글" in result
        assert "🎉" in result
        
        json_part = result[6:-2]
        parsed = json.loads(json_part)
        assert parsed["content"] == "한글 테스트 🎉"
    
    def test_format_sse_event_empty(self):
        """빈 데이터 SSE 포맷팅 테스트"""
        event_data = {}
        result = format_sse_event(event_data)
        
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        
        json_part = result[6:-2]
        parsed = json.loads(json_part)
        assert parsed == {}

