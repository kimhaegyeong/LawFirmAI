"""
데이터베이스 모델 정의
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.database.connection import Base


class Session(Base):
    """세션 모델"""
    __tablename__ = "sessions"
    
    session_id = Column(String(255), primary_key=True)
    title = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    message_count = Column(Integer, default=0)
    user_id = Column(String(255))
    ip_address = Column(String(45))
    
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
        }


class Message(Base):
    """메시지 모델"""
    __tablename__ = "messages"
    
    message_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    metadata = Column(JSON)
    
    session = relationship("Session", back_populates="messages")
    
    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

