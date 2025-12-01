# -*- coding: utf-8 -*-
"""
Data Module
데이터 처리 및 관리 모듈
"""

# DatabaseManager는 deprecated되었습니다. LegalDataConnectorV2를 사용하세요.
# from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2

try:
    from .vector_store import LegalVectorStore as VectorStore
except ImportError as e:
    # psutil 관련 경고는 무시 (선택적 의존성)
    if "psutil" not in str(e).lower():
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

try:
    from .db_adapter import DatabaseAdapter
except ImportError as e:
    print(f"Warning: Could not import DatabaseAdapter: {e}")
    DatabaseAdapter = None

try:
    from .sql_adapter import SQLAdapter
except ImportError as e:
    print(f"Warning: Could not import SQLAdapter: {e}")
    SQLAdapter = None

__all__ = [
    "VectorStore", 
    "DataProcessor",
    "LegalTermNormalizer",
    "DatabaseAdapter",
    "SQLAdapter",
]
