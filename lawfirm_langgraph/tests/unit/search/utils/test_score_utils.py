# -*- coding: utf-8 -*-
"""검색 점수 정규화 유틸리티 테스트"""

import pytest
from lawfirm_langgraph.core.search.utils.score_utils import (
    normalize_score,
    clamp_score,
    normalize_scores_batch
)


class TestNormalizeScore:
    """normalize_score 함수 테스트"""
    
    def test_normalize_score_within_range(self):
        """정상 범위 내 점수는 그대로 반환"""
        assert normalize_score(0.5) == 0.5
        assert normalize_score(0.0) == 0.0
        assert normalize_score(1.0) == 1.0
        assert normalize_score(0.75) == 0.75
    
    def test_normalize_score_below_min(self):
        """최소값 미만 점수는 최소값으로 클리핑"""
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(-10.0) == 0.0
    
    def test_normalize_score_above_max(self):
        """최대값 초과 점수는 로그 스케일로 정규화"""
        # 1.0 초과 점수는 로그 스케일로 완화되어 1.0에 수렴
        result = normalize_score(1.1)
        assert 0.0 <= result <= 1.0
        assert result <= 1.0
        
        result = normalize_score(1.2)
        assert 0.0 <= result <= 1.0
        assert result <= 1.0
        
        result = normalize_score(2.0)
        assert 0.0 <= result <= 1.0
        assert result <= 1.0
        
        # 매우 큰 값도 1.0 이하로 제한
        result = normalize_score(10.0)
        assert 0.0 <= result <= 1.0
        assert result <= 1.0
    
    def test_normalize_score_custom_range(self):
        """사용자 정의 범위 테스트"""
        assert normalize_score(5.0, min_val=0.0, max_val=10.0) == 0.5
        assert normalize_score(15.0, min_val=0.0, max_val=10.0) <= 1.0
        assert normalize_score(-5.0, min_val=0.0, max_val=10.0) == 0.0


class TestClampScore:
    """clamp_score 함수 테스트"""
    
    def test_clamp_score_within_range(self):
        """정상 범위 내 점수는 그대로 반환"""
        assert clamp_score(0.5) == 0.5
        assert clamp_score(0.0) == 0.0
        assert clamp_score(1.0) == 1.0
    
    def test_clamp_score_below_min(self):
        """최소값 미만 점수는 최소값으로 클리핑"""
        assert clamp_score(-0.5) == 0.0
        assert clamp_score(-10.0) == 0.0
    
    def test_clamp_score_above_max(self):
        """최대값 초과 점수는 최대값으로 클리핑"""
        assert clamp_score(1.1) == 1.0
        assert clamp_score(1.2) == 1.0
        assert clamp_score(2.0) == 1.0
        assert clamp_score(10.0) == 1.0
    
    def test_clamp_score_custom_range(self):
        """사용자 정의 범위 테스트"""
        assert clamp_score(5.0, min_val=0.0, max_val=10.0) == 5.0
        assert clamp_score(15.0, min_val=0.0, max_val=10.0) == 10.0
        assert clamp_score(-5.0, min_val=0.0, max_val=10.0) == 0.0


class TestNormalizeScoresBatch:
    """normalize_scores_batch 함수 테스트"""
    
    def test_normalize_scores_batch_empty(self):
        """빈 리스트는 그대로 반환"""
        assert normalize_scores_batch([]) == []
    
    def test_normalize_scores_batch_clamp_method(self):
        """clamp 방법으로 정규화"""
        scores = [0.5, 1.1, 1.2, -0.1, 0.8]
        normalized = normalize_scores_batch(scores, method="clamp")
        
        assert len(normalized) == len(scores)
        assert all(0.0 <= score <= 1.0 for score in normalized)
        assert normalized[0] == 0.5  # 정상 범위
        assert normalized[1] == 1.0  # 클리핑
        assert normalized[2] == 1.0  # 클리핑
        assert normalized[3] == 0.0  # 클리핑
        assert normalized[4] == 0.8  # 정상 범위
    
    def test_normalize_scores_batch_min_max_method(self):
        """min_max 방법으로 정규화"""
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        normalized = normalize_scores_batch(scores, method="min_max")
        
        assert len(normalized) == len(scores)
        assert all(0.0 <= score <= 1.0 for score in normalized)
        assert normalized[0] == 0.0  # 최소값
        assert normalized[-1] == 1.0  # 최대값
    
    def test_normalize_scores_batch_with_exceeding_scores(self):
        """1.0 초과 점수가 포함된 경우"""
        scores = [0.5, 1.1, 1.2, 0.8]
        normalized = normalize_scores_batch(scores, method="min_max")
        
        assert len(normalized) == len(scores)
        assert all(0.0 <= score <= 1.0 for score in normalized)
    
    def test_normalize_scores_batch_same_scores(self):
        """모든 점수가 같은 경우"""
        scores = [0.5, 0.5, 0.5]
        normalized = normalize_scores_batch(scores, method="min_max")
        
        assert len(normalized) == len(scores)
        assert all(score == 0.5 for score in normalized)
    
    def test_normalize_scores_batch_single_score(self):
        """단일 점수"""
        scores = [1.2]
        normalized = normalize_scores_batch(scores, method="min_max")
        
        assert len(normalized) == 1
        assert normalized[0] == 0.5  # 모든 점수가 같으면 0.5


class TestScoreNormalizationIntegration:
    """점수 정규화 통합 테스트"""
    
    def test_score_normalization_prevents_exceeding_one(self):
        """1.0 초과 점수가 발생하지 않도록 보장"""
        test_scores = [
            0.0, 0.1, 0.5, 0.9, 1.0,  # 정상 범위
            1.1, 1.2, 1.5, 2.0, 10.0,  # 초과 점수
            -0.1, -1.0, -10.0  # 음수 점수
        ]
        
        for score in test_scores:
            normalized = normalize_score(score)
            assert 0.0 <= normalized <= 1.0, f"Score {score} normalized to {normalized} is out of range"
    
    def test_score_normalization_preserves_order(self):
        """정규화 후에도 점수 순서가 유지되는지 확인"""
        scores = [0.1, 0.5, 0.8, 1.0, 1.1, 1.2]
        normalized = [normalize_score(score) for score in scores]
        
        # 정규화 후에도 순서는 유지되어야 함 (단, 1.0 초과는 모두 1.0에 수렴)
        for i in range(len(normalized) - 1):
            # 1.0 이하 점수들은 순서 유지
            if scores[i] <= 1.0 and scores[i+1] <= 1.0:
                assert normalized[i] <= normalized[i+1], \
                    f"Order not preserved: {normalized[i]} > {normalized[i+1]}"

