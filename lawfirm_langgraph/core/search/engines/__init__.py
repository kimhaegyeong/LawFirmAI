# -*- coding: utf-8 -*-
"""
Search Engines Module
검색 엔진 모듈
"""

from .semantic_search_engine_v2 import SemanticSearchEngineV2
from .hybrid_search_engine_v2 import HybridSearchEngineV2
from .keyword_search_engine import KeywordSearchEngine
from .precedent_search_engine import PrecedentSearchEngine

try:
    from .vector_search_adapter import (
        VectorSearchAdapter,
        PgVectorAdapter,
        FaissAdapter,
        VectorSearchFactory,
    )
except ImportError as e:
    print(f"Warning: Could not import vector_search_adapter: {e}")
    VectorSearchAdapter = None
    PgVectorAdapter = None
    FaissAdapter = None
    VectorSearchFactory = None

__all__ = [
    "SemanticSearchEngineV2",
    "HybridSearchEngineV2",
    "KeywordSearchEngine",
    "PrecedentSearchEngine",
    "VectorSearchAdapter",
    "PgVectorAdapter",
    "FaissAdapter",
    "VectorSearchFactory",
]

