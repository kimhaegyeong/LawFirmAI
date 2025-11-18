# -*- coding: utf-8 -*-
"""
Chain Builders 테스트
langgraph_core/nodes/chain_builders.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch

from lawfirm_langgraph.langgraph_core.nodes.chain_builders import (
    DirectAnswerChainBuilder,
)


class TestDirectAnswerChainBuilder:
    """DirectAnswerChainBuilder 테스트"""
    
    def test_build_query_type_analysis_prompt(self):
        """질문 유형 분석 프롬프트 생성 테스트"""
        query = "안녕하세요"
        prompt = DirectAnswerChainBuilder.build_query_type_analysis_prompt(query)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert "query_type" in prompt
        assert "greeting" in prompt
        assert "term_definition" in prompt
        assert "simple_question" in prompt
    
    def test_build_prompt_generation_prompt(self):
        """프롬프트 생성 테스트 (다양한 query_type)"""
        prompt_greeting = DirectAnswerChainBuilder.build_prompt_generation_prompt("안녕하세요", "greeting")
        assert isinstance(prompt_greeting, str)
        assert "안녕하세요" in prompt_greeting
        assert "인사" in prompt_greeting or "응답" in prompt_greeting
        
        prompt_term = DirectAnswerChainBuilder.build_prompt_generation_prompt("계약이란 무엇인가요?", "term_definition")
        assert isinstance(prompt_term, str)
        assert "정의" in prompt_term
        
        prompt_simple = DirectAnswerChainBuilder.build_prompt_generation_prompt("법률 상담은 어디서 받나요?", "simple_question")
        assert isinstance(prompt_simple, str)
        assert "간단" in prompt_simple or "답변" in prompt_simple
        
        prompt_unknown = DirectAnswerChainBuilder.build_prompt_generation_prompt("테스트 질문", "unknown")
        assert isinstance(prompt_unknown, str)
        assert "테스트 질문" in prompt_unknown

