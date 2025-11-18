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
    
    def test_extract_json(self):
        """JSON 추출 테스트 (유효한, 중첩된, 없는 경우)"""
        valid_response = 'Here is the result: {"key": "value", "number": 123}'
        json_str_valid = ResponseParser.extract_json(valid_response)
        assert json_str_valid is not None
        assert "key" in json_str_valid
        
        nested_response = 'Result: {"outer": {"inner": "value"}}'
        json_str_nested = ResponseParser.extract_json(nested_response)
        assert json_str_nested is not None
        
        no_json_response = "This is a plain text response without JSON."
        json_str_no_json = ResponseParser.extract_json(no_json_response)
        assert json_str_no_json is None
    
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
    
    def test_parse_question_type_response_invalid_or_missing(self):
        """유효하지 않거나 필수 필드가 없는 질문 유형 응답 파싱 테스트"""
        invalid_response = "This is not a valid JSON response."
        result_invalid = ClassificationParser.parse_question_type_response(invalid_response)
        assert isinstance(result_invalid, dict)
        assert result_invalid["question_type"] == "general_question"
        
        missing_field_response = '{"confidence": 0.9}'
        result_missing = ClassificationParser.parse_question_type_response(missing_field_response)
        assert isinstance(result_missing, dict)
        assert result_missing["question_type"] == "general_question"
    
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

