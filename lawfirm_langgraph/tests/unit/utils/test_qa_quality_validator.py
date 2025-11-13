# -*- coding: utf-8 -*-
"""
QA Quality Validator 테스트
유틸리티 qa_quality_validator 모듈 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.utils.qa_quality_validator import QAQualityValidator


class TestQAQualityValidator:
    """QA Quality Validator 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        validator = QAQualityValidator()
        
        assert validator is not None
        assert hasattr(validator, 'quality_criteria')
        assert hasattr(validator, 'legal_patterns')
        assert hasattr(validator, 'inappropriate_patterns')
        assert isinstance(validator.quality_criteria, dict)
        assert isinstance(validator.legal_patterns, list)
        assert isinstance(validator.inappropriate_patterns, list)
    
    def test_validate_qa_pair_valid(self):
        """유효한 Q&A 쌍 검증 테스트"""
        validator = QAQualityValidator()
        
        qa_pair = {
            "question": "민법 제750조에 대해 알려주세요",
            "answer": "민법 제750조는 불법행위로 인한 손해배상에 관한 조문입니다. 이 조문은 고의 또는 과실로 인한 불법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다고 규정하고 있습니다."
        }
        
        result = validator.validate_qa_pair(qa_pair)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "quality_score" in result
        assert "issues" in result
        assert "suggestions" in result
        assert "confidence" in result
    
    def test_validate_qa_pair_short_answer(self):
        """짧은 답변 검증 테스트"""
        validator = QAQualityValidator()
        
        qa_pair = {
            "question": "민법 제750조에 대해 알려주세요",
            "answer": "짧은 답변"
        }
        
        result = validator.validate_qa_pair(qa_pair)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
    
    def test_validate_qa_pair_empty_question(self):
        """빈 질문 검증 테스트"""
        validator = QAQualityValidator()
        
        qa_pair = {
            "question": "",
            "answer": "답변입니다"
        }
        
        result = validator.validate_qa_pair(qa_pair)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
    
    def test_validate_qa_pair_empty_answer(self):
        """빈 답변 검증 테스트"""
        validator = QAQualityValidator()
        
        qa_pair = {
            "question": "질문입니다",
            "answer": ""
        }
        
        result = validator.validate_qa_pair(qa_pair)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
    
    def test_validate_qa_pair_missing_keys(self):
        """키 누락 검증 테스트"""
        validator = QAQualityValidator()
        
        qa_pair = {}
        
        result = validator.validate_qa_pair(qa_pair)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
    
    def test_quality_criteria_structure(self):
        """품질 기준 구조 테스트"""
        validator = QAQualityValidator()
        
        criteria = validator.quality_criteria
        
        assert "min_question_length" in criteria
        assert "max_question_length" in criteria
        assert "min_answer_length" in criteria
        assert "max_answer_length" in criteria
        assert "min_quality_score" in criteria
        assert "high_quality_threshold" in criteria
    
    def test_legal_patterns_exist(self):
        """법률 패턴 존재 테스트"""
        validator = QAQualityValidator()
        
        assert len(validator.legal_patterns) > 0
        for pattern in validator.legal_patterns:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
    
    def test_inappropriate_patterns_exist(self):
        """부적절한 패턴 존재 테스트"""
        validator = QAQualityValidator()
        
        assert len(validator.inappropriate_patterns) > 0
        for pattern in validator.inappropriate_patterns:
            assert isinstance(pattern, str)
            assert len(pattern) > 0

