# -*- coding: utf-8 -*-
"""
Workflow Constants 테스트
langgraph_core/utils/workflow_constants.py 단위 테스트
"""

import pytest

from lawfirm_langgraph.langgraph_core.utils.workflow_constants import (
    WorkflowConstants,
    RetryConfig,
    QualityThresholds,
    AnswerExtractionPatterns
)


class TestWorkflowConstants:
    """WorkflowConstants 테스트"""
    
    def test_max_output_tokens(self):
        """최대 출력 토큰 테스트"""
        assert isinstance(WorkflowConstants.MAX_OUTPUT_TOKENS, int)
        assert WorkflowConstants.MAX_OUTPUT_TOKENS > 0
    
    def test_temperature(self):
        """온도 설정 테스트"""
        assert isinstance(WorkflowConstants.TEMPERATURE, float)
        assert 0.0 <= WorkflowConstants.TEMPERATURE <= 1.0
    
    def test_timeout(self):
        """타임아웃 설정 테스트"""
        assert isinstance(WorkflowConstants.TIMEOUT, int)
        assert WorkflowConstants.TIMEOUT > 0
    
    def test_semantic_search_k(self):
        """의미적 검색 K 값 테스트"""
        assert isinstance(WorkflowConstants.SEMANTIC_SEARCH_K, int)
        assert WorkflowConstants.SEMANTIC_SEARCH_K > 0
    
    def test_max_documents(self):
        """최대 문서 수 테스트"""
        assert isinstance(WorkflowConstants.MAX_DOCUMENTS, int)
        assert WorkflowConstants.MAX_DOCUMENTS > 0
    
    def test_max_retries(self):
        """최대 재시도 횟수 테스트"""
        assert isinstance(WorkflowConstants.MAX_RETRIES, int)
        assert WorkflowConstants.MAX_RETRIES > 0
    
    def test_confidence_values(self):
        """신뢰도 값 테스트"""
        assert isinstance(WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE, float)
        assert isinstance(WorkflowConstants.FALLBACK_CONFIDENCE, float)
        assert isinstance(WorkflowConstants.DEFAULT_CONFIDENCE, float)
        assert 0.0 <= WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE <= 1.0
        assert 0.0 <= WorkflowConstants.FALLBACK_CONFIDENCE <= 1.0
        assert 0.0 <= WorkflowConstants.DEFAULT_CONFIDENCE <= 1.0
    
    def test_min_answer_length(self):
        """최소 답변 길이 테스트"""
        assert isinstance(WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION, int)
        assert isinstance(WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION, int)
        assert WorkflowConstants.MIN_ANSWER_LENGTH_GENERATION > 0
        assert WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION > 0


class TestRetryConfig:
    """RetryConfig 테스트"""
    
    def test_max_generation_retries(self):
        """최대 생성 재시도 횟수 테스트"""
        assert isinstance(RetryConfig.MAX_GENERATION_RETRIES, int)
        assert RetryConfig.MAX_GENERATION_RETRIES > 0
    
    def test_max_validation_retries(self):
        """최대 검증 재시도 횟수 테스트"""
        assert isinstance(RetryConfig.MAX_VALIDATION_RETRIES, int)
        assert RetryConfig.MAX_VALIDATION_RETRIES >= 0
    
    def test_max_total_retries(self):
        """최대 전체 재시도 횟수 테스트"""
        assert isinstance(RetryConfig.MAX_TOTAL_RETRIES, int)
        assert RetryConfig.MAX_TOTAL_RETRIES > 0


class TestQualityThresholds:
    """QualityThresholds 테스트"""
    
    def test_quality_pass_threshold(self):
        """품질 통과 임계값 테스트"""
        assert isinstance(QualityThresholds.QUALITY_PASS_THRESHOLD, float)
        assert 0.0 <= QualityThresholds.QUALITY_PASS_THRESHOLD <= 1.0
    
    def test_quality_thresholds(self):
        """품질 임계값 테스트"""
        assert isinstance(QualityThresholds.HIGH_QUALITY_THRESHOLD, float)
        assert isinstance(QualityThresholds.MEDIUM_QUALITY_THRESHOLD, float)
        assert 0.0 <= QualityThresholds.HIGH_QUALITY_THRESHOLD <= 1.0
        assert 0.0 <= QualityThresholds.MEDIUM_QUALITY_THRESHOLD <= 1.0
    
    def test_quality_min_lengths(self):
        """품질별 최소 길이 테스트"""
        assert isinstance(QualityThresholds.HIGH_QUALITY_MIN_LENGTH, int)
        assert isinstance(QualityThresholds.MEDIUM_QUALITY_MIN_LENGTH, int)
        assert isinstance(QualityThresholds.LOW_QUALITY_MIN_LENGTH, int)
        assert QualityThresholds.HIGH_QUALITY_MIN_LENGTH > 0
        assert QualityThresholds.MEDIUM_QUALITY_MIN_LENGTH > 0
        assert QualityThresholds.LOW_QUALITY_MIN_LENGTH > 0


class TestAnswerExtractionPatterns:
    """AnswerExtractionPatterns 테스트"""
    
    def test_reasoning_section_patterns(self):
        """추론 과정 섹션 패턴 테스트"""
        assert isinstance(AnswerExtractionPatterns.REASONING_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.REASONING_SECTION_PATTERNS) > 0
    
    def test_output_section_patterns(self):
        """출력 섹션 패턴 테스트"""
        assert isinstance(AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS) > 0
    
    def test_step_patterns(self):
        """단계 패턴 테스트"""
        assert isinstance(AnswerExtractionPatterns.STEP_PATTERNS, list)
        assert len(AnswerExtractionPatterns.STEP_PATTERNS) > 0
    
    def test_answer_section_patterns(self):
        """답변 섹션 패턴 테스트"""
        assert isinstance(AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS, list)
        assert len(AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS) > 0

