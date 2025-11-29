# -*- coding: utf-8 -*-
"""검색 점수 정규화 통합 테스트"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
    from lawfirm_langgraph.core.search.utils.score_utils import normalize_score
except ImportError:
    SemanticSearchEngineV2 = None
    normalize_score = None


class TestSimilarityFromDistance:
    """거리-유사도 변환 함수 테스트"""
    
    def test_calculate_similarity_from_distance_range(self):
        """거리-유사도 변환이 0.0~1.0 범위를 보장하는지 테스트"""
        if SemanticSearchEngineV2 is None:
            pytest.skip("SemanticSearchEngineV2를 import할 수 없습니다")
        
        # Mock 엔진 생성
        engine = Mock(spec=SemanticSearchEngineV2)
        engine.logger = Mock()
        engine.index = Mock()
        engine.index.metric_type = None
        
        # 실제 메서드 바인딩
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2 as RealEngine
        engine._calculate_similarity_from_distance = RealEngine._calculate_similarity_from_distance.__get__(engine, RealEngine)
        
        # 다양한 거리 값 테스트
        test_distances = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        
        for distance in test_distances:
            similarity = engine._calculate_similarity_from_distance(float(distance))
            assert 0.0 <= similarity <= 1.0, \
                f"Distance {distance} produced similarity {similarity} out of range [0.0, 1.0]"
    
    def test_calculate_similarity_with_bonus(self):
        """보너스 곱셈 후에도 1.0을 초과하지 않는지 테스트"""
        if SemanticSearchEngineV2 is None:
            pytest.skip("SemanticSearchEngineV2를 import할 수 없습니다")
        
        # Mock 엔진 생성
        engine = Mock(spec=SemanticSearchEngineV2)
        engine.logger = Mock()
        engine.index = Mock()
        
        # IndexIVFPQ 시뮬레이션 (보너스 곱셈이 발생하는 경우)
        import faiss
        engine.index.metric_type = faiss.METRIC_L2
        engine.index.__class__.__name__ = 'IndexIVFPQ'
        
        # 실제 메서드 바인딩
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2 as RealEngine
        engine._calculate_similarity_from_distance = RealEngine._calculate_similarity_from_distance.__get__(engine, RealEngine)
        
        # 작은 거리 값 테스트 (보너스가 적용되는 경우)
        test_distances = [0.0, 0.1, 0.5, 0.9, 0.99]
        
        for distance in test_distances:
            similarity = engine._calculate_similarity_from_distance(float(distance))
            assert 0.0 <= similarity <= 1.0, \
                f"Distance {distance} with bonus produced similarity {similarity} out of range [0.0, 1.0]"


class TestHybridScore:
    """하이브리드 점수 계산 함수 테스트"""
    
    def test_calculate_hybrid_score_range(self):
        """하이브리드 점수가 0.0~1.0 범위를 보장하는지 테스트"""
        if SemanticSearchEngineV2 is None:
            pytest.skip("SemanticSearchEngineV2를 import할 수 없습니다")
        
        # Mock 엔진 생성
        engine = Mock(spec=SemanticSearchEngineV2)
        engine.logger = Mock()
        
        # 실제 메서드 바인딩
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2 as RealEngine
        engine._calculate_hybrid_score = RealEngine._calculate_hybrid_score.__get__(engine, RealEngine)
        
        # 다양한 입력 조합 테스트
        test_cases = [
            (0.5, 0.5, 0.5),  # 정상 범위
            (0.9, 0.8, 0.7),  # 높은 점수
            (0.99, 0.99, 0.99),  # 매우 높은 점수
            (0.75, 0.5, 0.5),  # 보너스가 적용되는 경우
            (0.8, 0.5, 0.5),  # 보너스가 적용되는 경우
        ]
        
        for similarity, ml_confidence, quality_score in test_cases:
            hybrid_score = engine._calculate_hybrid_score(
                similarity=similarity,
                ml_confidence=ml_confidence,
                quality_score=quality_score
            )
            assert 0.0 <= hybrid_score <= 1.0, \
                f"Hybrid score {hybrid_score} out of range [0.0, 1.0] " \
                f"for inputs: similarity={similarity}, ml_confidence={ml_confidence}, quality_score={quality_score}"
    
    def test_calculate_hybrid_score_with_bonus(self):
        """보너스 적용 후에도 1.0을 초과하지 않는지 테스트"""
        if SemanticSearchEngineV2 is None:
            pytest.skip("SemanticSearchEngineV2를 import할 수 없습니다")
        
        # Mock 엔진 생성
        engine = Mock(spec=SemanticSearchEngineV2)
        engine.logger = Mock()
        
        # 실제 메서드 바인딩
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2 as RealEngine
        engine._calculate_hybrid_score = RealEngine._calculate_hybrid_score.__get__(engine, RealEngine)
        
        # 보너스가 적용되는 경우 (similarity >= 0.75)
        high_similarity_cases = [
            (0.75, 0.9, 0.9),  # 보너스 적용
            (0.8, 0.95, 0.95),  # 보너스 적용
            (0.9, 0.99, 0.99),  # 보너스 적용
            (0.99, 0.99, 0.99),  # 보너스 적용
        ]
        
        for similarity, ml_confidence, quality_score in high_similarity_cases:
            hybrid_score = engine._calculate_hybrid_score(
                similarity=similarity,
                ml_confidence=ml_confidence,
                quality_score=quality_score
            )
            assert 0.0 <= hybrid_score <= 1.0, \
                f"Hybrid score {hybrid_score} with bonus out of range [0.0, 1.0] " \
                f"for inputs: similarity={similarity}, ml_confidence={ml_confidence}, quality_score={quality_score}"


class TestScoreNormalizationIntegration:
    """점수 정규화 통합 테스트"""
    
    def test_all_scores_in_range(self):
        """모든 점수 계산 경로에서 0.0~1.0 범위 보장"""
        if normalize_score is None:
            pytest.skip("normalize_score를 import할 수 없습니다")
        
        # 다양한 점수 값 테스트
        test_scores = [
            # 정상 범위
            0.0, 0.1, 0.5, 0.9, 1.0,
            # 초과 점수
            1.1, 1.2, 1.5, 2.0, 5.0, 10.0,
            # 음수 점수
            -0.1, -1.0, -10.0,
        ]
        
        for score in test_scores:
            normalized = normalize_score(float(score))
            assert 0.0 <= normalized <= 1.0, \
                f"Score {score} normalized to {normalized} is out of range [0.0, 1.0]"
    
    def test_score_normalization_preserves_relative_order(self):
        """정규화 후에도 상대적 순서가 유지되는지 확인"""
        if normalize_score is None:
            pytest.skip("normalize_score를 import할 수 없습니다")
        
        # 정상 범위 내 점수들은 순서 유지
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        normalized = [normalize_score(score) for score in scores]
        
        for i in range(len(normalized) - 1):
            assert normalized[i] <= normalized[i+1], \
                f"Order not preserved: {normalized[i]} > {normalized[i+1]}"


class TestSearchResultScoreNormalization:
    """검색 결과 점수 정규화 테스트"""
    
    def test_search_result_scores_normalized(self):
        """검색 결과의 모든 점수 필드가 정규화되는지 테스트"""
        if normalize_score is None:
            pytest.skip("normalize_score를 import할 수 없습니다")
        
        # 시뮬레이션된 검색 결과
        test_results = [
            {
                "relevance_score": 0.8,
                "similarity": 0.75,
                "score": 0.85,
                "final_weighted_score": 0.9,
                "combined_score": 0.88
            },
            {
                "relevance_score": 1.1,  # 초과 점수
                "similarity": 1.2,  # 초과 점수
                "score": 0.9,
                "final_weighted_score": 1.15,  # 초과 점수
                "combined_score": 1.05  # 초과 점수
            },
            {
                "relevance_score": -0.1,  # 음수
                "similarity": 0.5,
                "score": 0.6,
                "final_weighted_score": 0.7
            }
        ]
        
        # 정규화 적용
        for result in test_results:
            if "relevance_score" in result:
                result["relevance_score"] = normalize_score(result["relevance_score"])
            if "similarity" in result:
                result["similarity"] = normalize_score(result["similarity"])
            if "score" in result:
                result["score"] = normalize_score(result["score"])
            if "final_weighted_score" in result:
                result["final_weighted_score"] = normalize_score(result["final_weighted_score"])
            if "combined_score" in result:
                result["combined_score"] = normalize_score(result["combined_score"])
        
        # 모든 점수가 0.0~1.0 범위인지 확인
        for result in test_results:
            for key, value in result.items():
                if "score" in key.lower() or "similarity" in key.lower():
                    assert 0.0 <= value <= 1.0, \
                        f"Score field '{key}' has value {value} out of range [0.0, 1.0]"

