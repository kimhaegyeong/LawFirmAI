# -*- coding: utf-8 -*-
"""
Workflow Constants 테스트
에이전트 workflow_constants 모듈 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.agents.workflow_constants import (
    WorkflowConstants,
    RetryConfig,
    QualityThresholds,
    AnswerExtractionPatterns
)


class TestWorkflowConstants:
    """Workflow Constants 테스트"""
    
    def test_workflow_constants_llm_settings(self):
        """LLM 설정 상수 테스트"""
        assert hasattr(WorkflowConstants, 'MAX_OUTPUT_TOKENS')
        assert isinstance(WorkflowConstants.MAX_OUTPUT_TOKENS, int)
        assert WorkflowConstants.MAX_OUTPUT_TOKENS > 0
        
        assert hasattr(WorkflowConstants, 'TEMPERATURE')
        assert isinstance(WorkflowConstants.TEMPERATURE, float)
        assert 0.0 <= WorkflowConstants.TEMPERATURE <= 1.0
        
        assert hasattr(WorkflowConstants, 'TIMEOUT')
        assert isinstance(WorkflowConstants.TIMEOUT, int)
        assert WorkflowConstants.TIMEOUT > 0
    
    def test_workflow_constants_search_settings(self):
        """검색 설정 상수 테스트"""
        assert hasattr(WorkflowConstants, 'SEMANTIC_SEARCH_K')
        assert isinstance(WorkflowConstants.SEMANTIC_SEARCH_K, int)
        assert WorkflowConstants.SEMANTIC_SEARCH_K > 0
        
        assert hasattr(WorkflowConstants, 'MAX_DOCUMENTS')
        assert isinstance(WorkflowConstants.MAX_DOCUMENTS, int)
        assert WorkflowConstants.MAX_DOCUMENTS > 0
        
        assert hasattr(WorkflowConstants, 'CATEGORY_SEARCH_LIMIT')
        assert isinstance(WorkflowConstants.CATEGORY_SEARCH_LIMIT, int)
        assert WorkflowConstants.CATEGORY_SEARCH_LIMIT > 0
    
    def test_workflow_constants_retry_settings(self):
        """재시도 설정 상수 테스트"""
        assert hasattr(WorkflowConstants, 'MAX_RETRIES')
        assert isinstance(WorkflowConstants.MAX_RETRIES, int)
        assert WorkflowConstants.MAX_RETRIES > 0
        
        assert hasattr(WorkflowConstants, 'RETRY_DELAY')
        assert isinstance(WorkflowConstants.RETRY_DELAY, int)
        assert WorkflowConstants.RETRY_DELAY > 0
    
    def test_workflow_constants_confidence_settings(self):
        """신뢰도 설정 상수 테스트"""
        assert hasattr(WorkflowConstants, 'LLM_CLASSIFICATION_CONFIDENCE')
        assert isinstance(WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE, float)
        assert 0.0 <= WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE <= 1.0
        
        assert hasattr(WorkflowConstants, 'FALLBACK_CONFIDENCE')
        assert isinstance(WorkflowConstants.FALLBACK_CONFIDENCE, float)
        assert 0.0 <= WorkflowConstants.FALLBACK_CONFIDENCE <= 1.0
        
        assert hasattr(WorkflowConstants, 'DEFAULT_CONFIDENCE')
        assert isinstance(WorkflowConstants.DEFAULT_CONFIDENCE, float)
        assert 0.0 <= WorkflowConstants.DEFAULT_CONFIDENCE <= 1.0
    
    def test_workflow_constants_answer_length(self):
        """답변 길이 임계값 테스트"""
        assert hasattr(WorkflowConstants, 'MIN_ANSWER_LENGTH_GENERATION')
        assert isinstance(WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION, int)
        assert WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION > 0
        
        assert hasattr(WorkflowConstants, 'MIN_ANSWER_LENGTH_VALIDATION')
        assert isinstance(WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION, int)
        assert WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION > 0


class TestRetryConfig:
    """Retry Config 테스트"""
    
    def test_retry_config_max_generation_retries(self):
        """최대 생성 재시도 테스트"""
        assert hasattr(RetryConfig, 'MAX_GENERATION_RETRIES')
        assert isinstance(RetryConfig.MAX_GENERATION_RETRIES, int)
        assert RetryConfig.MAX_GENERATION_RETRIES > 0
    
    def test_retry_config_max_validation_retries(self):
        """최대 검증 재시도 테스트"""
        assert hasattr(RetryConfig, 'MAX_VALIDATION_RETRIES')
        assert isinstance(RetryConfig.MAX_VALIDATION_RETRIES, int)
        assert RetryConfig.MAX_VALIDATION_RETRIES > 0
    
    def test_retry_config_max_total_retries(self):
        """최대 전체 재시도 테스트"""
        assert hasattr(RetryConfig, 'MAX_TOTAL_RETRIES')
        assert isinstance(RetryConfig.MAX_TOTAL_RETRIES, int)
        assert RetryConfig.MAX_TOTAL_RETRIES > 0


class TestQualityThresholds:
    """Quality Thresholds 테스트"""
    
    def test_quality_thresholds_pass(self):
        """품질 통과 임계값 테스트"""
        assert hasattr(QualityThresholds, 'QUALITY_PASS_THRESHOLD')
        assert isinstance(QualityThresholds.QUALITY_PASS_THRESHOLD, float)
        assert 0.0 <= QualityThresholds.QUALITY_PASS_THRESHOLD <= 1.0
    
    def test_quality_thresholds_levels(self):
        """품질 레벨 임계값 테스트"""
        assert hasattr(QualityThresholds, 'HIGH_QUALITY_THRESHOLD')
        assert isinstance(QualityThresholds.HIGH_QUALITY_THRESHOLD, float)
        assert 0.0 <= QualityThresholds.HIGH_QUALITY_THRESHOLD <= 1.0
        
        assert hasattr(QualityThresholds, 'MEDIUM_QUALITY_THRESHOLD')
        assert isinstance(QualityThresholds.MEDIUM_QUALITY_THRESHOLD, float)
        assert 0.0 <= QualityThresholds.MEDIUM_QUALITY_THRESHOLD <= 1.0
    
    def test_quality_thresholds_min_lengths(self):
        """품질별 최소 길이 테스트"""
        assert hasattr(QualityThresholds, 'HIGH_QUALITY_MIN_LENGTH')
        assert isinstance(QualityThresholds.HIGH_QUALITY_MIN_LENGTH, int)
        assert QualityThresholds.HIGH_QUALITY_MIN_LENGTH > 0
        
        assert hasattr(QualityThresholds, 'MEDIUM_QUALITY_MIN_LENGTH')
        assert isinstance(QualityThresholds.MEDIUM_QUALITY_MIN_LENGTH, int)
        assert QualityThresholds.MEDIUM_QUALITY_MIN_LENGTH > 0
        
        assert hasattr(QualityThresholds, 'LOW_QUALITY_MIN_LENGTH')
        assert isinstance(QualityThresholds.LOW_QUALITY_MIN_LENGTH, int)
        assert QualityThresholds.LOW_QUALITY_MIN_LENGTH > 0


class TestAnswerExtractionPatterns:
    """Answer Extraction Patterns 테스트"""
    
    def test_reasoning_section_patterns(self):
        """추론 섹션 패턴 테스트"""
        assert hasattr(AnswerExtractionPatterns, 'REASONING_SECTION_PATTERNS')
        assert isinstance(AnswerExtractionPatterns.REASONING_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.REASONING_SECTION_PATTERNS) > 0
        
        for pattern in AnswerExtractionPatterns.REASONING_SECTION_PATTERNS:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
    
    def test_output_section_patterns(self):
        """출력 섹션 패턴 테스트"""
        assert hasattr(AnswerExtractionPatterns, 'OUTPUT_SECTION_PATTERNS')
        assert isinstance(AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS) > 0
        
        for pattern in AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
    
    def test_step_patterns(self):
        """단계 패턴 테스트"""
        assert hasattr(AnswerExtractionPatterns, 'STEP_PATTERNS')
        assert isinstance(AnswerExtractionPatterns.STEP_PATTERNS, list)
        assert len(AnswerExtractionPatterns.STEP_PATTERNS) > 0
        
        for pattern in AnswerExtractionPatterns.STEP_PATTERNS:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
    
    def test_answer_section_patterns(self):
        """답변 섹션 패턴 테스트"""
        assert hasattr(AnswerExtractionPatterns, 'ANSWER_SECTION_PATTERNS')
        assert isinstance(AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS) > 0
        
        for pattern in AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS:
            assert isinstance(pattern, str)
            assert len(pattern) > 0

