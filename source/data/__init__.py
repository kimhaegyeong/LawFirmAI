# -*- coding: utf-8 -*-
"""
Data Module
데이터 처리 및 관리 모듈
"""

# 선택적 import로 의존성 문제 해결
try:
    from .database import DatabaseManager
except ImportError as e:
    print(f"Warning: Could not import DatabaseManager: {e}")
    DatabaseManager = None

try:
    from .vector_store import LegalVectorStore as VectorStore
except ImportError as e:
    print(f"Warning: Could not import LegalVectorStore: {e}")
    VectorStore = None

try:
    from .data_processor import LegalDataProcessor as DataProcessor
except ImportError as e:
    print(f"Warning: Could not import LegalDataProcessor: {e}")
    DataProcessor = None

try:
    from .legal_term_normalizer import LegalTermNormalizer
except ImportError as e:
    print(f"Warning: Could not import LegalTermNormalizer: {e}")
    LegalTermNormalizer = None

__all__ = [
    "DatabaseManager",
    "VectorStore", 
    "DataProcessor",
    "LegalTermNormalizer"
]
