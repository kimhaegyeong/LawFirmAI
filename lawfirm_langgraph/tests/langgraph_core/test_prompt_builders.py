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
    
    def test_build_semantic_query(self):
        """의미적 쿼리 생성 테스트 (용어 유무, 개수별)"""
        query = "계약 해지"
        
        result_with_terms = QueryBuilder.build_semantic_query(query, ["계약서", "해지", "계약관계", "해제"])
        assert query in result_with_terms
        assert isinstance(result_with_terms, str)
        
        result_without_terms = QueryBuilder.build_semantic_query(query, [])
        assert result_without_terms == query
        
        result_many_terms = QueryBuilder.build_semantic_query(query, ["term1", "term2", "term3", "term4", "term5", "term6"])
        assert query in result_many_terms
    
    def test_build_keyword_queries(self):
        """키워드 쿼리 생성 테스트 (다양한 query_type 및 제한)"""
        query = "계약 해지"
        expanded_terms = ["계약서", "해지"]
        
        result_basic = QueryBuilder.build_keyword_queries(query, expanded_terms, "general_question")
        assert isinstance(result_basic, list)
        assert len(result_basic) > 0
        assert query in result_basic
        
        result_precedent = QueryBuilder.build_keyword_queries(query, expanded_terms, "precedent_search")
        assert any("판례" in q or "사건" in q for q in result_precedent)
        
        result_law = QueryBuilder.build_keyword_queries(query, expanded_terms, "law_inquiry")
        assert any("법률" in q or "조항" in q or "법령" in q for q in result_law)
        
        result_advice = QueryBuilder.build_keyword_queries(query, expanded_terms, "legal_advice")
        assert any("조언" in q or "해석" in q for q in result_advice)
        
        result_max = QueryBuilder.build_keyword_queries(query, ["term1", "term2", "term3", "term4", "term5", "term6", "term7"], "general_question")
        assert len(result_max) <= 5
    
    def test_build_conversation_context_dict(self):
        """ConversationContext 딕셔너리 변환 테스트 (유효한, None, 예외)"""
        mock_context = Mock()
        mock_context.session_id = "test_session"
        mock_context.turns = [Mock(), Mock()]
        mock_context.entities = {"person": {"John"}}
        mock_context.topic_stack = ["topic1", "topic2"]
        
        result_valid = QueryBuilder.build_conversation_context_dict(mock_context)
        assert isinstance(result_valid, dict)
        assert result_valid["session_id"] == "test_session"
        assert result_valid["turn_count"] == 2
        
        result_none = QueryBuilder.build_conversation_context_dict(None)
        assert result_none is None
        
        mock_context_error = Mock()
        mock_context_error.session_id = property(lambda self: None)
        result_error = QueryBuilder.build_conversation_context_dict(mock_context_error)
        assert result_error is None or isinstance(result_error, dict)


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

