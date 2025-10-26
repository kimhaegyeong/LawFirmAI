# -*- coding: utf-8 -*-
"""
Services Module
비즈니스 로직 서비스 모듈

이 모듈은 법률 AI 어시스턴트의 핵심 서비스들을 기능별로 구성합니다.
- 채팅 서비스
- 검색 서비스
- 분석 서비스
- 검증 서비스
- 워크플로우 서비스
- 통합 서비스
"""

# 기능별 서비스 모듈 import
from .chat import *
from .search import *
from .analysis import *
from .validation import *
from .workflow import *
from .integration import *

__all__ = [
    # 채팅 서비스
    "ChatService",
    "EnhancedChatService",
    "ConversationManager",
    "MultiTurnQuestionHandler",

    # 검색 서비스
    "SearchService",
    "RAGService",
    "HybridSearchEngine",
    "SemanticSearchEngine",
    "PrecedentSearchEngine",

    # 분석 서비스
    "AnalysisService",
    "DocumentProcessor",
    "LegalTermExtractor",
    "BERTClassifier",

    # 검증 서비스
    "ResponseValidationSystem",
    "QualityValidator",
    "LegalBasisValidator",
    "ConfidenceCalculator",

    # 통합 서비스
    "AKLSProcessor",
    "LangfuseClient"
]
