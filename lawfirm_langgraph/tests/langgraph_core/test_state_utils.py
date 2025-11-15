# -*- coding: utf-8 -*-
"""
State Utils 테스트
langgraph_core/state/state_utils.py 단위 테스트
"""

import pytest
from typing import Dict, Any, List

from lawfirm_langgraph.langgraph_core.state.state_utils import (
    MAX_RETRIEVED_DOCS,
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_CONVERSATION_HISTORY,
    MAX_PROCESSING_STEPS,
    summarize_document,
    prune_retrieved_docs,
    prune_processing_steps
)


class TestStateConstants:
    """State 상수 테스트"""
    
    def test_max_constants(self):
        """최대값 상수 테스트"""
        assert isinstance(MAX_RETRIEVED_DOCS, int)
        assert MAX_RETRIEVED_DOCS > 0
        assert isinstance(MAX_DOCUMENT_CONTENT_LENGTH, int)
        assert MAX_DOCUMENT_CONTENT_LENGTH > 0
        assert isinstance(MAX_CONVERSATION_HISTORY, int)
        assert MAX_CONVERSATION_HISTORY > 0
        assert isinstance(MAX_PROCESSING_STEPS, int)
        assert MAX_PROCESSING_STEPS > 0


class TestSummarizeDocument:
    """summarize_document 테스트"""
    
    def test_summarize_document_short(self):
        """짧은 문서 요약 테스트"""
        doc = {"content": "Short content"}
        result = summarize_document(doc, max_content_length=500)
        
        assert result["content"] == "Short content"
        assert result.get("is_summarized", False) is False
    
    def test_summarize_document_long(self):
        """긴 문서 요약 테스트"""
        long_content = "A" * 1000
        doc = {"content": long_content}
        result = summarize_document(doc, max_content_length=500)
        
        assert len(result["content"]) <= 500
        assert result.get("is_summarized", False) is True
        assert result.get("original_content_length") == 1000
    
    def test_summarize_document_no_content(self):
        """내용이 없는 문서 요약 테스트"""
        doc = {"metadata": "test"}
        result = summarize_document(doc)
        
        assert result == doc
    
    def test_summarize_document_not_dict(self):
        """딕셔너리가 아닌 입력 테스트"""
        doc = "not a dict"
        result = summarize_document(doc)
        
        assert result == doc


class TestPruneRetrievedDocs:
    """prune_retrieved_docs 테스트"""
    
    def test_prune_retrieved_docs_basic(self):
        """기본 문서 정제 테스트"""
        docs = [
            {"content": "Doc 1", "relevance_score": 0.9},
            {"content": "Doc 2", "relevance_score": 0.8},
            {"content": "Doc 3", "relevance_score": 0.7}
        ]
        
        result = prune_retrieved_docs(docs, max_items=2)
        
        assert len(result) == 2
        assert result[0]["relevance_score"] >= result[1]["relevance_score"]
    
    def test_prune_retrieved_docs_empty(self):
        """빈 문서 목록 정제 테스트"""
        docs = []
        result = prune_retrieved_docs(docs)
        
        assert result == []
    
    def test_prune_retrieved_docs_with_score_field(self):
        """score 필드 사용 문서 정제 테스트"""
        docs = [
            {"content": "Doc 1", "score": 0.9},
            {"content": "Doc 2", "score": 0.8}
        ]
        
        result = prune_retrieved_docs(docs, max_items=1)
        
        assert len(result) == 1
        assert result[0]["score"] == 0.9
    
    def test_prune_retrieved_docs_no_score(self):
        """점수가 없는 문서 정제 테스트"""
        docs = [
            {"content": "Doc 1"},
            {"content": "Doc 2"}
        ]
        
        result = prune_retrieved_docs(docs, max_items=1)
        
        assert len(result) == 1


class TestPruneProcessingSteps:
    """prune_processing_steps 테스트"""
    
    def test_prune_processing_steps_basic(self):
        """기본 처리 단계 정제 테스트"""
        steps = [
            "step1",
            "step2",
            "step3"
        ]
        
        result = prune_processing_steps(steps, max_items=2)
        
        assert len(result) <= 2
        assert isinstance(result, list)
    
    def test_prune_processing_steps_empty(self):
        """빈 처리 단계 정제 테스트"""
        steps = []
        result = prune_processing_steps(steps)
        
        assert isinstance(result, list)

