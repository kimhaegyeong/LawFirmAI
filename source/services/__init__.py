# -*- coding: utf-8 -*-
"""
Services Module
비즈니스 로직 서비스 모듈
"""

# 선택적 import로 의존성 문제 해결
try:
    from .chat_service import ChatService
except ImportError as e:
    print(f"Warning: Could not import ChatService: {e}")
    ChatService = None

try:
    from .rag_service import RAGService
except ImportError as e:
    print(f"Warning: Could not import RAGService: {e}")
    RAGService = None

try:
    from .search_service import SearchService
except ImportError as e:
    print(f"Warning: Could not import SearchService: {e}")
    SearchService = None

try:
    from .analysis_service import AnalysisService
except ImportError as e:
    print(f"Warning: Could not import AnalysisService: {e}")
    AnalysisService = None

# TASK 3.2 하이브리드 검색 시스템 모듈들
try:
    from .exact_search_engine import ExactSearchEngine
except ImportError as e:
    print(f"Warning: Could not import ExactSearchEngine: {e}")
    ExactSearchEngine = None

try:
    from .semantic_search_engine import SemanticSearchEngine
except ImportError as e:
    print(f"Warning: Could not import SemanticSearchEngine: {e}")
    SemanticSearchEngine = None

try:
    from .result_merger import ResultMerger, ResultRanker
except ImportError as e:
    print(f"Warning: Could not import ResultMerger/ResultRanker: {e}")
    ResultMerger = None
    ResultRanker = None

try:
    from .hybrid_search_engine import HybridSearchEngine
except ImportError as e:
    print(f"Warning: Could not import HybridSearchEngine: {e}")
    HybridSearchEngine = None

__all__ = [
    "ChatService",
    "RAGService", 
    "SearchService",
    "AnalysisService",
    "ExactSearchEngine",
    "SemanticSearchEngine",
    "ResultMerger",
    "ResultRanker",
    "HybridSearchEngine"
]