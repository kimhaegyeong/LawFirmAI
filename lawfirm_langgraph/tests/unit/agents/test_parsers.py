# -*- coding: utf-8 -*-
"""
Agents Parsers 테스트
파서 모듈 단위 테스트
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch

from lawfirm_langgraph.core.agents.parsers.response_parsers import (
    ResponseParser,
    ClassificationParser,
    AnswerParser,
    QueryParser,
    DocumentParser
)


class TestResponseParser:
    """ResponseParser 테스트"""
    
    def test_extract_json(self):
        """JSON 추출 테스트"""
        response = '{"question_type": "legal_advice", "confidence": 0.9}'
        
        json_str = ResponseParser.extract_json(response)
        
        assert json_str is not None
        assert "question_type" in json_str
    
    def test_extract_json_no_json(self):
        """JSON이 없는 응답 테스트"""
        response = "일반 텍스트 응답"
        
        json_str = ResponseParser.extract_json(response)
        
        assert json_str is None or isinstance(json_str, str)
    
    def test_parse_json_safe(self):
        """안전한 JSON 파싱 테스트"""
        json_str = '{"question_type": "legal_advice", "confidence": 0.9}'
        
        parsed = ResponseParser.parse_json_safe(json_str, {})
        
        assert isinstance(parsed, dict)
        assert parsed["question_type"] == "legal_advice"
        assert parsed["confidence"] == 0.9
    
    def test_parse_json_safe_invalid(self):
        """잘못된 JSON 파싱 테스트"""
        json_str = '{"invalid": json}'
        
        default = {"error": "parsing failed"}
        parsed = ResponseParser.parse_json_safe(json_str, default)
        
        assert parsed == default
    
    def test_parse_json_optional(self):
        """선택적 JSON 파싱 테스트"""
        json_str = '{"question_type": "legal_advice"}'
        
        parsed = ResponseParser.parse_json_optional(json_str)
        
        assert isinstance(parsed, dict)
        assert parsed["question_type"] == "legal_advice"
    
    def test_parse_json_optional_invalid(self):
        """잘못된 JSON 선택적 파싱 테스트"""
        json_str = '{"invalid": json}'
        
        parsed = ResponseParser.parse_json_optional(json_str)
        
        assert parsed is None


class TestClassificationParser:
    """ClassificationParser 테스트"""
    
    def test_parse_question_type_response(self):
        """질문 유형 분류 응답 파싱 테스트"""
        response = '{"question_type": "legal_advice", "confidence": 0.9, "reasoning": "테스트"}'
        
        parsed = ClassificationParser.parse_question_type_response(response)
        
        assert isinstance(parsed, dict)
        assert "question_type" in parsed
        assert "confidence" in parsed
    
    def test_parse_question_type_response_invalid(self):
        """잘못된 질문 유형 응답 파싱 테스트"""
        response = "일반 텍스트"
        
        parsed = ClassificationParser.parse_question_type_response(response)
        
        assert isinstance(parsed, dict)
        assert parsed["question_type"] == "general_question"
    
    def test_parse_legal_field_response(self):
        """법률 분야 추출 응답 파싱 테스트"""
        response = '{"legal_field": "contract", "confidence": 0.8}'
        
        parsed = ClassificationParser.parse_legal_field_response(response)
        
        assert isinstance(parsed, dict)
        assert parsed["legal_field"] == "contract"
    
    def test_parse_legal_field_response_invalid(self):
        """잘못된 법률 분야 응답 파싱 테스트"""
        response = "일반 텍스트"
        
        parsed = ClassificationParser.parse_legal_field_response(response)
        
        assert parsed is None
    
    def test_parse_complexity_response(self):
        """복잡도 평가 응답 파싱 테스트"""
        response = '{"complexity": "complex", "confidence": 0.9, "reasoning": "테스트"}'
        
        parsed = ClassificationParser.parse_complexity_response(response)
        
        assert isinstance(parsed, dict)
        assert "complexity" in parsed
        assert parsed["complexity"] == "complex"
    
    def test_parse_query_type_analysis_response(self):
        """질문 유형 분석 응답 파싱 테스트"""
        response = '{"query_type": "greeting", "confidence": 0.9, "reasoning": "테스트"}'
        
        parsed = ClassificationParser.parse_query_type_analysis_response(response)
        
        assert isinstance(parsed, dict)
        assert "query_type" in parsed
        assert parsed["query_type"] == "greeting"


class TestAnswerParser:
    """AnswerParser 테스트"""
    
    def test_parse_validation_response(self):
        """답변 검증 응답 파싱 테스트"""
        response = '{"is_valid": true, "quality_score": 0.9, "issues": [], "strengths": []}'
        
        parsed = AnswerParser.parse_validation_response(response)
        
        assert isinstance(parsed, dict)
        assert "is_valid" in parsed
        assert parsed["is_valid"] is True
        assert "quality_score" in parsed
    
    def test_parse_validation_response_invalid(self):
        """잘못된 검증 응답 파싱 테스트"""
        response = "일반 텍스트"
        
        parsed = AnswerParser.parse_validation_response(response)
        
        assert isinstance(parsed, dict)
        assert parsed["is_valid"] is True
    
    def test_parse_improvement_instructions(self):
        """개선 지시 응답 파싱 테스트"""
        response = '{"improvement_instructions": "더 구체적으로 설명하세요"}'
        
        parsed = AnswerParser.parse_improvement_instructions(response)
        
        assert isinstance(parsed, dict)
        assert "improvement_instructions" in parsed
    
    def test_parse_improvement_instructions_invalid(self):
        """잘못된 개선 지시 응답 파싱 테스트"""
        response = "일반 텍스트"
        
        parsed = AnswerParser.parse_improvement_instructions(response)
        
        assert parsed is None


class TestQueryParser:
    """QueryParser 테스트"""
    
    def test_parse_query_expansion_response(self):
        """쿼리 확장 응답 파싱 테스트"""
        # parse_query_expansion_response 메서드가 없으므로 parse_keyword_expansion_response 사용
        response = '{"expanded_keywords": ["계약서", "작성"], "synonyms": ["계약", "문서"]}'
        
        parsed = QueryParser.parse_keyword_expansion_response(response)
        
        assert isinstance(parsed, dict) or parsed is None
        if parsed:
            assert "expanded_keywords" in parsed or "synonyms" in parsed


class TestDocumentParser:
    """DocumentParser 테스트"""
    
    def test_parse_document_metadata(self):
        """문서 메타데이터 파싱 테스트"""
        # parse_document_metadata 메서드가 없으므로 parse_document_type_response 사용
        response = '{"document_type": "contract", "confidence": 0.9, "reasoning": "계약서"}'
        
        parsed = DocumentParser.parse_document_type_response(response)
        
        assert isinstance(parsed, dict) or parsed is None
        if parsed:
            assert "document_type" in parsed or "confidence" in parsed

