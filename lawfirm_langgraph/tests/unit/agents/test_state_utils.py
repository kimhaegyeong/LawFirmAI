# -*- coding: utf-8 -*-
"""
State Utils 테스트
에이전트 state_utils 모듈 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.workflow.state.state_utils import (
    summarize_document,
    prune_retrieved_docs,
    prune_conversation_history,
    prune_processing_steps,
    consolidate_metadata
)


class TestStateUtils:
    """State Utils 테스트"""
    
    def test_summarize_document_short(self):
        """짧은 문서 요약 테스트"""
        doc = {"content": "short text", "metadata": "test"}
        result = summarize_document(doc)
        
        assert result["content"] == "short text"
        assert result["is_summarized"] is False
    
    def test_summarize_document_long(self):
        """긴 문서 요약 테스트"""
        long_content = "a" * 1000
        doc = {"content": long_content, "metadata": "test"}
        result = summarize_document(doc, max_content_length=500)
        
        assert len(result["content"]) <= 500
        assert result["is_summarized"] is True
        assert result["original_content_length"] == 1000
    
    def test_summarize_document_no_content(self):
        """content 없는 문서 테스트"""
        doc = {"metadata": "test"}
        result = summarize_document(doc)
        
        assert "content" not in result
        assert "is_summarized" not in result
    
    def test_summarize_document_not_dict(self):
        """딕셔너리가 아닌 입력 테스트"""
        result = summarize_document("not a dict")
        assert result == "not a dict"
    
    def test_summarize_document_very_short_max(self):
        """매우 짧은 max_content_length 테스트"""
        long_content = "a" * 100
        doc = {"content": long_content}
        result = summarize_document(doc, max_content_length=50)
        
        assert len(result["content"]) <= 50
        assert result["is_summarized"] is True
    
    def test_prune_retrieved_docs_empty(self):
        """빈 문서 목록 정제 테스트"""
        result = prune_retrieved_docs([])
        assert result == []
    
    def test_prune_retrieved_docs_basic(self):
        """기본 문서 목록 정제 테스트"""
        docs = [
            {"content": "doc1", "relevance_score": 0.9},
            {"content": "doc2", "relevance_score": 0.8},
            {"content": "doc3", "relevance_score": 0.7},
        ]
        result = prune_retrieved_docs(docs, max_items=2)
        
        assert len(result) == 2
        assert result[0]["relevance_score"] == 0.9
        assert result[1]["relevance_score"] == 0.8
    
    def test_prune_retrieved_docs_with_score_key(self):
        """score 키 사용 문서 정제 테스트"""
        docs = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
        ]
        result = prune_retrieved_docs(docs, max_items=2)
        
        assert len(result) == 2
        assert result[0]["score"] == 0.9
    
    def test_prune_retrieved_docs_no_score(self):
        """점수 없는 문서 정제 테스트"""
        docs = [
            {"content": "doc1"},
            {"content": "doc2"},
        ]
        result = prune_retrieved_docs(docs, max_items=2)
        
        assert len(result) == 2
    
    def test_prune_retrieved_docs_content_summarization(self):
        """문서 내용 요약 포함 정제 테스트"""
        long_content = "a" * 1000
        docs = [
            {"content": long_content, "relevance_score": 0.9},
        ]
        result = prune_retrieved_docs(docs, max_items=10, max_content_per_doc=500)
        
        assert len(result) == 1
        assert len(result[0]["content"]) <= 500
    
    def test_prune_conversation_history_empty(self):
        """빈 대화 이력 정제 테스트"""
        result = prune_conversation_history([])
        assert result == []
    
    def test_prune_conversation_history_short(self):
        """짧은 대화 이력 정제 테스트"""
        history = [{"turn": 1}, {"turn": 2}]
        result = prune_conversation_history(history, max_items=5)
        
        assert len(result) == 2
        assert result == history
    
    def test_prune_conversation_history_long(self):
        """긴 대화 이력 정제 테스트"""
        history = [{"turn": i} for i in range(10)]
        result = prune_conversation_history(history, max_items=5)
        
        assert len(result) == 5
        assert result[0]["turn"] == 5
        assert result[-1]["turn"] == 9
    
    def test_prune_processing_steps_empty(self):
        """빈 처리 단계 정제 테스트"""
        result = prune_processing_steps([])
        assert result == []
    
    def test_prune_processing_steps_short(self):
        """짧은 처리 단계 정제 테스트"""
        steps = ["step1", "step2"]
        result = prune_processing_steps(steps, max_items=5)
        
        assert len(result) == 2
        assert result == steps
    
    def test_prune_processing_steps_long(self):
        """긴 처리 단계 정제 테스트"""
        steps = [f"step{i}" for i in range(25)]
        result = prune_processing_steps(steps, max_items=20)
        
        assert len(result) == 20
        assert result[0] == "step5"
        assert result[-1] == "step24"
    
    def test_consolidate_metadata_empty(self):
        """빈 상태 메타데이터 통합 테스트"""
        state = {}
        result = consolidate_metadata(state)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_consolidate_metadata_basic(self):
        """기본 메타데이터 통합 테스트"""
        state = {
            "metadata": {"key1": "value1"}
        }
        result = consolidate_metadata(state)
        
        assert result["key1"] == "value1"
    
    def test_consolidate_metadata_with_search(self):
        """검색 메타데이터 통합 테스트"""
        state = {
            "search_metadata": {"query": "test"}
        }
        result = consolidate_metadata(state)
        
        assert "search" in result
        assert result["search"]["query"] == "test"
    
    def test_consolidate_metadata_with_format(self):
        """포맷 메타데이터 통합 테스트"""
        state = {
            "format_metadata": {"format": "json"}
        }
        result = consolidate_metadata(state)
        
        assert "format" in result
        assert result["format"]["format"] == "json"
    
    def test_consolidate_metadata_with_quality(self):
        """품질 메트릭 통합 테스트"""
        state = {
            "quality_metrics": {"score": 0.9}
        }
        result = consolidate_metadata(state)
        
        assert "quality" in result
        assert result["quality"]["score"] == 0.9
    
    def test_consolidate_metadata_all(self):
        """모든 메타데이터 통합 테스트"""
        state = {
            "metadata": {"key1": "value1"},
            "search_metadata": {"query": "test"},
            "format_metadata": {"format": "json"},
            "quality_metrics": {"score": 0.9}
        }
        result = consolidate_metadata(state)
        
        assert result["key1"] == "value1"
        assert result["search"]["query"] == "test"
        assert result["format"]["format"] == "json"
        assert result["quality"]["score"] == 0.9

