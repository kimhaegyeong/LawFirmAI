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
    
    def test_build_prompt_generation_prompt_greeting(self):
        """인사말 프롬프트 생성 테스트"""
        query = "안녕하세요"
        prompt = DirectAnswerChainBuilder.build_prompt_generation_prompt(query, "greeting")
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert "인사" in prompt or "응답" in prompt
    
    def test_build_prompt_generation_prompt_term_definition(self):
        """용어 정의 프롬프트 생성 테스트"""
        query = "계약이란 무엇인가요?"
        prompt = DirectAnswerChainBuilder.build_prompt_generation_prompt(query, "term_definition")
        
        assert isinstance(prompt, str)
        assert query in prompt or "용어" in prompt
        assert "정의" in prompt
    
    def test_build_prompt_generation_prompt_simple_question(self):
        """간단한 질문 프롬프트 생성 테스트"""
        query = "법률 상담은 어디서 받나요?"
        prompt = DirectAnswerChainBuilder.build_prompt_generation_prompt(query, "simple_question")
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert "간단" in prompt or "답변" in prompt
    
    def test_build_prompt_generation_prompt_unknown_type(self):
        """알 수 없는 유형 프롬프트 생성 테스트"""
        query = "테스트 질문"
        prompt = DirectAnswerChainBuilder.build_prompt_generation_prompt(query, "unknown")
        
        assert isinstance(prompt, str)
        assert query in prompt

