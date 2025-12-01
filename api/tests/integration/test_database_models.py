"""
데이터베이스 모델 테스트
"""
import pytest
from datetime import datetime
from api.database.models import Session, Message


class TestDatabaseModels:
    """데이터베이스 모델 테스트"""
    
    def test_session_to_dict(self):
        """Session to_dict 테스트"""
        session = Session(
            session_id="test-session",
            title="테스트 세션",
            message_count=5,
            user_id="user123",
            ip_address="127.0.0.1"
        )
        session.created_at = datetime.now()
        session.updated_at = datetime.now()
        
        result = session.to_dict()
        
        assert result["session_id"] == "test-session"
        assert result["title"] == "테스트 세션"
        assert result["message_count"] == 5
        assert result["user_id"] == "user123"
        assert result["ip_address"] == "127.0.0.1"
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_session_to_dict_none_dates(self):
        """날짜가 None인 Session to_dict 테스트"""
        session = Session(
            session_id="test-session",
            title="테스트 세션"
        )
        session.created_at = None
        session.updated_at = None
        
        result = session.to_dict()
        
        assert result["created_at"] is None
        assert result["updated_at"] is None
    
    def test_message_to_dict(self):
        """Message to_dict 테스트"""
        message = Message(
            message_id="test-message",
            session_id="test-session",
            role="user",
            content="테스트 메시지",
            metadata={"key": "value"}
        )
        message.timestamp = datetime.now()
        
        result = message.to_dict()
        
        assert result["message_id"] == "test-message"
        assert result["session_id"] == "test-session"
        assert result["role"] == "user"
        assert result["content"] == "테스트 메시지"
        assert result["metadata"] == {"key": "value"}
        assert "timestamp" in result
    
    def test_message_to_dict_none_timestamp(self):
        """타임스탬프가 None인 Message to_dict 테스트"""
        message = Message(
            message_id="test-message",
            session_id="test-session",
            role="user",
            content="테스트 메시지"
        )
        message.timestamp = None
        
        result = message.to_dict()
        
        assert result["timestamp"] is None

