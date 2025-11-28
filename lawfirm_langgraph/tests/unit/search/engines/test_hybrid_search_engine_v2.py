# -*- coding: utf-8 -*-
"""
HybridSearchEngineV2 단위 테스트
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

import pytest
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionType


class TestHybridSearchEngineV2:
    """HybridSearchEngineV2 테스트 클래스"""

    @pytest.fixture
    def mock_exact_search(self):
        """ExactSearchEngineV2 Mock"""
        mock = MagicMock()
        mock.search.return_value = {
            "law": [
                {
                    "text": "민법 제543조 계약 해지",
                    "metadata": {"type": "law", "article_id": "543"},
                    "relevance_score": 0.8
                }
            ],
            "precedent": [],
            "decision": [],
            "interpretation": []
        }
        return mock

    @pytest.fixture
    def mock_semantic_search(self):
        """SemanticSearchEngineV2 Mock"""
        mock = MagicMock()
        mock.search.return_value = [
            {
                "text": "계약 해지에 대한 법률 규정",
                "metadata": {"type": "law", "article_id": "543"},
                "relevance_score": 0.7,
                "similarity": 0.7
            }
        ]
        return mock

    @pytest.fixture
    def mock_question_classifier(self):
        """QuestionClassifier Mock"""
        mock = MagicMock()
        mock.classify.return_value = "law_inquiry"
        return mock

    @pytest.fixture
    def mock_result_merger(self):
        """ResultMerger Mock"""
        mock = MagicMock()
        from lawfirm_langgraph.core.search.processors.result_merger import MergedResult
        mock.merge_results.return_value = [
            MergedResult(
                text="민법 제543조 계약 해지",
                score=0.75,
                source="hybrid",
                metadata={"type": "law", "article_id": "543"}
            )
        ]
        return mock

    @pytest.fixture
    def mock_result_ranker(self):
        """ResultRanker Mock"""
        mock = MagicMock()
        from lawfirm_langgraph.core.search.processors.result_merger import MergedResult
        mock.rank_results.return_value = [
            MergedResult(
                text="민법 제543조 계약 해지",
                score=0.75,
                source="hybrid",
                metadata={"type": "law", "article_id": "543"}
            )
        ]
        mock.apply_diversity_filter.return_value = [
            MergedResult(
                text="민법 제543조 계약 해지",
                score=0.75,
                source="hybrid",
                metadata={"type": "law", "article_id": "543"}
            )
        ]
        return mock

    @pytest.fixture
    def mock_config(self):
        """Config Mock"""
        mock = MagicMock()
        mock.database_path = "test_db.db"
        mock.embedding_model = "test_model"
        mock.use_mlflow_index = False
        mock.mlflow_run_id = None
        return mock

    @pytest.fixture
    def hybrid_engine(self, mock_exact_search, mock_semantic_search, 
                     mock_question_classifier, mock_result_merger, 
                     mock_result_ranker, mock_config):
        """HybridSearchEngineV2 인스턴스 생성"""
        with patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.Config', return_value=mock_config), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ExactSearchEngineV2', return_value=mock_exact_search), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.SemanticSearchEngineV2', return_value=mock_semantic_search), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.QuestionClassifier', return_value=mock_question_classifier), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultMerger', return_value=mock_result_merger), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultRanker', return_value=mock_result_ranker):
            
            engine = HybridSearchEngineV2()
            return engine

    def test_initialization(self, mock_config):
        """초기화 테스트"""
        with patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.Config', return_value=mock_config), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ExactSearchEngineV2'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.SemanticSearchEngineV2'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.QuestionClassifier'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultMerger'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultRanker'):
            
            engine = HybridSearchEngineV2()
            
            assert engine.db_path == "test_db.db"
            assert engine.model_name == "test_model"
            assert engine.search_config["exact_search_weight"] == 0.6
            assert engine.search_config["semantic_search_weight"] == 0.4
            assert engine.search_config["max_results"] == 50

    def test_initialization_with_custom_params(self, mock_config):
        """커스텀 파라미터로 초기화 테스트"""
        with patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.Config', return_value=mock_config), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ExactSearchEngineV2'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.SemanticSearchEngineV2'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.QuestionClassifier'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultMerger'), \
             patch('lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2.ResultRanker'):
            
            engine = HybridSearchEngineV2(db_path="custom_db.db", model_name="custom_model")
            
            assert engine.db_path == "custom_db.db"
            assert engine.model_name == "custom_model"

    def test_search_basic(self, hybrid_engine, mock_exact_search, mock_semantic_search,
                         mock_question_classifier, mock_result_merger, mock_result_ranker):
        """기본 검색 테스트"""
        query = "계약 해지"
        
        result = hybrid_engine.search(query)
        
        assert "results" in result
        assert "total" in result
        assert "exact_count" in result
        assert "semantic_count" in result
        assert "query" in result
        assert result["query"] == query
        assert isinstance(result["results"], list)
        
        # Mock 호출 확인
        mock_exact_search.search.assert_called_once()
        mock_semantic_search.search.assert_called_once()
        mock_question_classifier.classify.assert_called_once_with(query)
        mock_result_merger.merge_results.assert_called_once()
        mock_result_ranker.rank_results.assert_called_once()
        mock_result_ranker.apply_diversity_filter.assert_called_once()

    def test_search_with_search_types(self, hybrid_engine, mock_exact_search, mock_semantic_search):
        """검색 타입 지정 테스트"""
        query = "계약 해지"
        search_types = ["law", "precedent"]
        
        result = hybrid_engine.search(query, search_types=search_types)
        
        assert result["query"] == query
        # exact_search는 search_types를 받아야 함
        call_args = mock_exact_search.search.call_args
        assert call_args[1]["search_types"] == search_types

    def test_search_with_max_results(self, hybrid_engine, mock_result_ranker):
        """최대 결과 수 지정 테스트"""
        query = "계약 해지"
        max_results = 10
        
        result = hybrid_engine.search(query, max_results=max_results)
        
        assert result["total"] <= max_results
        # rank_results 호출 시 top_k 확인
        call_args = mock_result_ranker.rank_results.call_args
        assert call_args[1]["top_k"] == max_results

    def test_search_exact_only(self, hybrid_engine, mock_exact_search, mock_semantic_search):
        """정확 검색만 수행 테스트"""
        query = "계약 해지"
        
        result = hybrid_engine.search(query, include_exact=True, include_semantic=False)
        
        mock_exact_search.search.assert_called_once()
        mock_semantic_search.search.assert_not_called()

    def test_search_semantic_only(self, hybrid_engine, mock_exact_search, mock_semantic_search):
        """의미 검색만 수행 테스트"""
        query = "계약 해지"
        
        result = hybrid_engine.search(query, include_exact=False, include_semantic=True)
        
        mock_exact_search.search.assert_not_called()
        mock_semantic_search.search.assert_called_once()

    def test_search_question_type_weighting(self, hybrid_engine, mock_question_classifier, mock_result_merger):
        """질문 유형별 가중치 조정 테스트"""
        query = "계약 해지 판례"
        
        # precedent_search 타입으로 분류
        mock_question_classifier.classify.return_value = "precedent_search"
        
        result = hybrid_engine.search(query)
        
        # merge_results 호출 시 가중치 확인
        call_args = mock_result_merger.merge_results.call_args
        weights = call_args[1]["weights"]
        assert weights["exact"] == 0.4
        assert weights["semantic"] == 0.6

    def test_search_law_inquiry_weighting(self, hybrid_engine, mock_question_classifier, mock_result_merger):
        """법령 조회 질문 가중치 테스트"""
        query = "민법 제543조"
        
        mock_question_classifier.classify.return_value = "law_inquiry"
        
        result = hybrid_engine.search(query)
        
        call_args = mock_result_merger.merge_results.call_args
        weights = call_args[1]["weights"]
        assert weights["exact"] == 0.6
        assert weights["semantic"] == 0.4

    def test_search_complex_question_weighting(self, hybrid_engine, mock_question_classifier, mock_result_merger):
        """복합 질문 가중치 테스트"""
        query = "계약 해지와 해제의 차이"
        
        mock_question_classifier.classify.return_value = "complex_question"
        
        result = hybrid_engine.search(query)
        
        call_args = mock_result_merger.merge_results.call_args
        weights = call_args[1]["weights"]
        assert weights["exact"] == 0.5
        assert weights["semantic"] == 0.5

    def test_search_default_weighting(self, hybrid_engine, mock_question_classifier, mock_result_merger):
        """기본 가중치 테스트"""
        query = "일반 질문"
        
        mock_question_classifier.classify.return_value = "unknown_type"
        
        result = hybrid_engine.search(query)
        
        call_args = mock_result_merger.merge_results.call_args
        weights = call_args[1]["weights"]
        assert weights["exact"] == 0.6
        assert weights["semantic"] == 0.4

    def test_execute_exact_search(self, hybrid_engine, mock_exact_search):
        """정확 검색 실행 테스트"""
        query = "계약 해지"
        search_types = ["law", "precedent"]
        
        result = hybrid_engine._execute_exact_search(query, search_types)
        
        assert isinstance(result, dict)
        mock_exact_search.search.assert_called_once_with(query, search_types=search_types)

    def test_execute_exact_search_error(self, hybrid_engine, mock_exact_search):
        """정확 검색 에러 처리 테스트"""
        query = "계약 해지"
        search_types = ["law"]
        
        mock_exact_search.search.side_effect = Exception("Search error")
        
        result = hybrid_engine._execute_exact_search(query, search_types)
        
        assert result == {}

    def test_execute_semantic_search(self, hybrid_engine, mock_semantic_search):
        """의미 검색 실행 테스트"""
        query = "계약 해지"
        search_types = ["law", "precedent"]
        
        result = hybrid_engine._execute_semantic_search(query, search_types)
        
        assert isinstance(result, list)
        mock_semantic_search.search.assert_called_once()
        call_args = mock_semantic_search.search.call_args
        assert call_args[0][0] == query

    def test_execute_semantic_search_source_type_mapping(self, hybrid_engine, mock_semantic_search):
        """의미 검색 source_type 매핑 테스트"""
        query = "계약 해지"
        search_types = ["law", "precedent"]
        
        hybrid_engine._execute_semantic_search(query, search_types)
        
        call_args = mock_semantic_search.search.call_args
        source_types = call_args[1]["source_types"]
        assert "statute_article" in source_types
        assert "case_paragraph" in source_types

    def test_execute_semantic_search_error(self, hybrid_engine, mock_semantic_search):
        """의미 검색 에러 처리 테스트"""
        query = "계약 해지"
        search_types = ["law"]
        
        mock_semantic_search.search.side_effect = Exception("Search error")
        
        result = hybrid_engine._execute_semantic_search(query, search_types)
        
        assert result == []

    def test_search_error_handling(self, hybrid_engine, mock_exact_search):
        """검색 에러 처리 테스트"""
        query = "계약 해지"
        
        mock_exact_search.search.side_effect = Exception("Search error")
        
        result = hybrid_engine.search(query)
        
        assert "error" in result
        assert result["total"] == 0
        assert result["results"] == []

    def test_search_result_format(self, hybrid_engine):
        """검색 결과 형식 테스트"""
        query = "계약 해지"
        
        result = hybrid_engine.search(query)
        
        assert "results" in result
        assert "total" in result
        assert "exact_count" in result
        assert "semantic_count" in result
        assert "query" in result
        
        if result["results"]:
            first_result = result["results"][0]
            assert "text" in first_result
            assert "score" in first_result
            assert "metadata" in first_result

    def test_search_default_search_types(self, hybrid_engine, mock_exact_search):
        """기본 검색 타입 테스트"""
        query = "계약 해지"
        
        hybrid_engine.search(query)
        
        call_args = mock_exact_search.search.call_args
        search_types = call_args[1]["search_types"]
        assert "law" in search_types
        assert "precedent" in search_types
        assert "decision" in search_types
        assert "interpretation" in search_types

    def test_search_diversity_filter(self, hybrid_engine, mock_result_ranker):
        """다양성 필터 적용 테스트"""
        query = "계약 해지"
        
        hybrid_engine.search(query)
        
        call_args = mock_result_ranker.apply_diversity_filter.call_args
        assert call_args[1]["max_per_type"] == hybrid_engine.search_config["diversity_max_per_type"]

