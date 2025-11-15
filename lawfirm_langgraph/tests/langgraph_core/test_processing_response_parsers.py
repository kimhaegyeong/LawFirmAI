# -*- coding: utf-8 -*-
"""
Processing Response Parsers 테스트
langgraph_core/processing/response_parsers.py 단위 테스트
"""

import pytest
import json
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.processing.response_parsers import (
    ResponseParser,
    ClassificationParser
)


class TestResponseParser:
    """ResponseParser 테스트"""
    
    def test_extract_json_valid(self):
        """유효한 JSON 추출 테스트"""
        response = 'Here is the result: {"key": "value", "number": 123}'
        json_str = ResponseParser.extract_json(response)
        
        assert json_str is not None
        assert "key" in json_str
    
    def test_extract_json_no_json(self):
        """JSON이 없는 응답 테스트"""
        response = "This is a plain text response without JSON."
        json_str = ResponseParser.extract_json(response)
        
        assert json_str is None
    
    def test_extract_json_nested(self):
        """중첩된 JSON 추출 테스트"""
        response = 'Result: {"outer": {"inner": "value"}}'
        json_str = ResponseParser.extract_json(response)
        
        assert json_str is not None
    
    def test_parse_json_safe_valid(self):
        """유효한 JSON 파싱 테스트"""
        json_str = '{"key": "value", "number": 123}'
        default = {}
        
        result = ResponseParser.parse_json_safe(json_str, default)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 123
    
    def test_parse_json_safe_invalid(self):
        """유효하지 않은 JSON 파싱 테스트"""
        json_str = '{"key": "value", invalid}'
        default = {"error": "parsing failed"}
        
        result = ResponseParser.parse_json_safe(json_str, default)
        
        assert result == default
    
    def test_parse_json_optional_valid(self):
        """유효한 JSON 선택적 파싱 테스트"""
        json_str = '{"key": "value"}'
        
        result = ResponseParser.parse_json_optional(json_str)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
    
    def test_parse_json_optional_invalid(self):
        """유효하지 않은 JSON 선택적 파싱 테스트"""
        json_str = '{"key": "value", invalid}'
        
        result = ResponseParser.parse_json_optional(json_str)
        
        assert result is None


class TestClassificationParser:
    """ClassificationParser 테스트"""
    
    def test_parse_question_type_response_valid(self):
        """유효한 질문 유형 응답 파싱 테스트"""
        response = '{"question_type": "legal_advice", "confidence": 0.9, "reasoning": "법률 조언 질문"}'
        
        result = ClassificationParser.parse_question_type_response(response)
        
        assert isinstance(result, dict)
        assert result["question_type"] == "legal_advice"
        assert result["confidence"] == 0.9
    
    def test_parse_question_type_response_invalid(self):
        """유효하지 않은 질문 유형 응답 파싱 테스트"""
        response = "This is not a valid JSON response."
        
        result = ClassificationParser.parse_question_type_response(response)
        
        assert isinstance(result, dict)
        assert "question_type" in result
        assert result["question_type"] == "general_question"
    
    def test_parse_question_type_response_missing_field(self):
        """필수 필드가 없는 질문 유형 응답 파싱 테스트"""
        response = '{"confidence": 0.9}'
        
        result = ClassificationParser.parse_question_type_response(response)
        
        assert isinstance(result, dict)
        assert result["question_type"] == "general_question"
    
    def test_parse_legal_field_response_valid(self):
        """유효한 법률 분야 응답 파싱 테스트"""
        response = '{"legal_field": "계약법", "confidence": 0.85}'
        
        result = ClassificationParser.parse_legal_field_response(response)
        
        assert isinstance(result, dict)
        assert result["legal_field"] == "계약법"
    
    def test_parse_legal_field_response_invalid(self):
        """유효하지 않은 법률 분야 응답 파싱 테스트"""
        response = "This is not a valid JSON response."
        
        result = ClassificationParser.parse_legal_field_response(response)
        
        assert result is None

