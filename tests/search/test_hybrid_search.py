# -*- coding: utf-8 -*-
"""
Hybrid search unit/integration-like tests for SemanticSearchEngine
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _make_engine_with_metadata():
    from source.services.semantic_search_engine import SemanticSearchEngine
    engine = SemanticSearchEngine()
    # Prevent loading actual FAISS/model by overriding components
    engine.index = None
    engine.model = None
    engine.metadata = [
        {"id": 1, "text": "민법 ??50�?불법?�위 ?�해배상 규정", "type": "law", "source": "민법"},
        {"id": 2, "text": "?�혼 ?�차?� ?�건???�???�설", "type": "guide", "source": "가?�드"},
        {"id": 3, "text": "계약 ?��? ???�의?�항 �??��? ?��?", "type": "guide", "source": "가?�드"},
    ]
    return engine


def test_query_rewrite_generates_variants():
    engine = _make_engine_with_metadata()
    variants = engine._rewrite_query("민법 ??50�?조문")
    assert isinstance(variants, list)
    assert any("?�?��?�?민법" in v or "??750 �? in v for v in variants)


def test_keyword_rank_returns_results():
    engine = _make_engine_with_metadata()
    res = engine._keyword_rank("?�해배상", k=5)
    assert isinstance(res, list)
    assert len(res) > 0
    assert all(isinstance(r, dict) for r in res)


class _FakeIndex:
    def search(self, vec, topk):
        # Return indices [0, 2] as top results with dummy scores
        scores = np.array([[0.9, 0.8] + [-1] * (topk - 2)], dtype=np.float32)
        idx = np.array([[0, 2] + [-1] * (topk - 2)], dtype=np.int64)
        return scores, idx


def test_hybrid_prefers_vector_when_available(monkeypatch):
    engine = _make_engine_with_metadata()
    # enable vector path
    engine.index = _FakeIndex()
    class _FakeModel:
        def encode(self, arr):
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    engine.model = _FakeModel()

    results = engine.search("민법 ?�해배상", k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    # Ensure hybrid score computed
    assert any('search_type' in r for r in results)


def test_min_recall_fallback_uses_db_when_no_results(monkeypatch):
    engine = _make_engine_with_metadata()
    # Force no keyword and no vector results
    engine.metadata = []
    engine.index = None
    engine.model = None

    # Monkeypatch DB fallback to return stub
    stub = [{
        'text': 'DB: 민법 개요', 'score': 0.5, 'metadata': {'id': 'db_1'}, 'search_type': 'database_fallback'
    }]
    monkeypatch.setattr(engine, "_database_fallback_search", lambda q, k: stub)

    results = engine.search("민법", k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].get('search_type') in ( 'database_fallback', 'keyword_fallback')
