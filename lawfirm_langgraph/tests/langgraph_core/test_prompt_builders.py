# -*- coding: utf-8 -*-
"""
Prompt Builders 테스트
langgraph_core/nodes/prompt_builders.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from lawfirm_langgraph.langgraph_core.nodes.prompt_builders import (
    QueryBuilder,
    PromptBuilder
)


class TestQueryBuilder:
    """QueryBuilder 테스트"""
    
    def test_build_semantic_query_with_terms(self):
        """확장된 용어가 있는 의미적 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지", "계약관계", "해제"]
        
        result = QueryBuilder.build_semantic_query(query, expanded_terms)
        
        assert query in result
        assert isinstance(result, str)
    
    def test_build_semantic_query_without_terms(self):
        """확장된 용어가 없는 의미적 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = []
        
        result = QueryBuilder.build_semantic_query(query, expanded_terms)
        
        assert result == query
    
    def test_build_semantic_query_many_terms(self):
        """많은 용어가 있는 경우 테스트"""
        query = "계약 해지"
        expanded_terms = ["term1", "term2", "term3", "term4", "term5", "term6"]
        
        result = QueryBuilder.build_semantic_query(query, expanded_terms)
        
        assert query in result
    
    def test_build_keyword_queries_basic(self):
        """기본 키워드 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지"]
        query_type = "general_question"
        
        result = QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert query in result
    
    def test_build_keyword_queries_precedent_search(self):
        """판례 검색용 키워드 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지"]
        query_type = "precedent_search"
        
        result = QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)
        
        assert isinstance(result, list)
        assert any("판례" in q or "사건" in q for q in result)
    
    def test_build_keyword_queries_law_inquiry(self):
        """법령 조문 검색용 키워드 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지"]
        query_type = "law_inquiry"
        
        result = QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)
        
        assert isinstance(result, list)
        assert any("법률" in q or "조항" in q or "법령" in q for q in result)
    
    def test_build_keyword_queries_legal_advice(self):
        """법률 조언용 키워드 쿼리 생성 테스트"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지"]
        query_type = "legal_advice"
        
        result = QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)
        
        assert isinstance(result, list)
        assert any("조언" in q or "해석" in q for q in result)
    
    def test_build_keyword_queries_max_limit(self):
        """키워드 쿼리 최대 개수 제한 테스트"""
        query = "계약 해지"
        expanded_terms = ["term1", "term2", "term3", "term4", "term5", "term6", "term7"]
        query_type = "general_question"
        
        result = QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)
        
        assert len(result) <= 5
    
    def test_build_conversation_context_dict_valid(self):
        """유효한 ConversationContext 딕셔너리 변환 테스트"""
        mock_context = Mock()
        mock_context.session_id = "test_session"
        mock_context.turns = [Mock(), Mock()]
        mock_context.entities = {"person": {"John"}}
        mock_context.topic_stack = ["topic1", "topic2"]
        
        result = QueryBuilder.build_conversation_context_dict(mock_context)
        
        assert isinstance(result, dict)
        assert result["session_id"] == "test_session"
        assert result["turn_count"] == 2
    
    def test_build_conversation_context_dict_none(self):
        """None ConversationContext 테스트"""
        result = QueryBuilder.build_conversation_context_dict(None)
        
        assert result is None
    
    def test_build_conversation_context_dict_exception(self):
        """예외 발생 시 ConversationContext 변환 테스트"""
        mock_context = Mock()
        mock_context.session_id = property(lambda self: None)
        
        result = QueryBuilder.build_conversation_context_dict(mock_context)
        
        assert result is None or isinstance(result, dict)


class TestPromptBuilder:
    """PromptBuilder 테스트"""
    
    def test_build_query_enhancement_prompt_base(self):
        """쿼리 강화 프롬프트 생성 테스트"""
        query = "계약 해지 방법"
        query_type = "legal_advice"
        extracted_keywords = ["계약", "해지"]
        legal_field = "계약법"
        
        def format_field_info(field: str) -> str:
            return f"법률 분야: {field}"
        
        def format_query_guide(qtype: str) -> Dict[str, Any]:
            return {
                "description": "법률 조언",
                "search_focus": "관련 법령",
                "search_strategy": "핵심 키워드 중심",
                "database_fields": "전체",
                "keyword_suggestions": ["계약", "해지"]
            }
        
        result = PromptBuilder.build_query_enhancement_prompt_base(
            query=query,
            query_type=query_type,
            extracted_keywords=extracted_keywords,
            legal_field=legal_field,
            format_field_info=format_field_info,
            format_query_guide=format_query_guide
        )
        
        assert isinstance(result, str)
        assert query in result
        assert query_type in result
        assert legal_field in result
    
    def test_build_query_enhancement_prompt_base_empty_keywords(self):
        """빈 키워드로 쿼리 강화 프롬프트 생성 테스트"""
        query = "계약 해지 방법"
        query_type = "legal_advice"
        extracted_keywords = []
        legal_field = ""
        
        def format_field_info(field: str) -> str:
            return ""
        
        def format_query_guide(qtype: str) -> Dict[str, Any]:
            return {}
        
        result = PromptBuilder.build_query_enhancement_prompt_base(
            query=query,
            query_type=query_type,
            extracted_keywords=extracted_keywords,
            legal_field=legal_field,
            format_field_info=format_field_info,
            format_query_guide=format_query_guide
        )
        
        assert isinstance(result, str)
        assert query in result

