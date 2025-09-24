"""
LawFirmAI - 법률 AI 어시스턴트
HuggingFace Spaces 배포용 법률 AI 시스템
"""

__version__ = "1.0.0"
__author__ = "LawFirmAI Team"
__description__ = "법률 AI 어시스턴트 - 판례, 법령, Q&A 데이터베이스 기반 AI 시스템"

# 주요 모듈 import
from .models import ModelManager
from .services import ChatService
from .data import DatabaseManager, VectorStore
from .utils import Config

__all__ = [
    "ModelManager",
    "ChatService", 
    "DatabaseManager",
    "VectorStore",
    "Config"
]