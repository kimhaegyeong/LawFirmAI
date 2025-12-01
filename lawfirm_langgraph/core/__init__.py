# -*- coding: utf-8 -*-
"""
LawFirmAI - 법률 AI 어시스턴트
HuggingFace Spaces 배포용 법률 AI 시스템
"""

__version__ = "1.0.0"
__author__ = "LawFirmAI Team"
__description__ = "법률 AI 어시스턴트 - 판례, 법령, Q&A 데이터베이스 기반 AI 시스템"

# 주요 모듈 import (선택적)
# 모델 관련 import는 필요시에만 개별적으로 수행
try:
    from .models.model_manager import LegalModelManager
except ImportError:
    LegalModelManager = None

LegalModelFineTuner = None

try:
    from .services.chat_service import ChatService
except ImportError:
    ChatService = None

try:
    from .data.vector_store import VectorStore
except ImportError:
    VectorStore = None

try:
    from .utils.config import Config
except ImportError:
    Config = None

# TASK 3.2 하이브리드 검색 시스템 모듈들
try:
    from .search.processors.result_merger import ResultMerger, ResultRanker
except ImportError:
    ResultMerger = None
    ResultRanker = None

try:
    from .data.legal_term_normalizer import LegalTermNormalizer
except ImportError:
    LegalTermNormalizer = None

__all__ = [
    "LegalModelManager",
    "LegalModelFineTuner",
    "ChatService", 
    "VectorStore",
    "Config",
    "ResultMerger",
    "ResultRanker",
    "LegalTermNormalizer"
]