# -*- coding: utf-8 -*-
"""
Search Handlers 테스트
검색 핸들러 모듈 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Tuple

from lawfirm_langgraph.core.search.handlers.search_handler import SearchHandler


class TestSearchHandler:
    """SearchHandler 테스트"""
    
    @pytest.fixture
    def search_handler(self):
        """SearchHandler 인스턴스"""
        mock_semantic_search = MagicMock()
        mock_keyword_mapper = MagicMock()
        mock_data_connector = MagicMock()
        mock_result_merger = MagicMock()
        mock_result_ranker = MagicMock()
        mock_performance_optimizer = MagicMock()
        mock_config = MagicMock()
        
        return SearchHandler(
            semantic_search=mock_semantic_search,
            keyword_mapper=mock_keyword_mapper,
            data_connector=mock_data_connector,
            result_merger=mock_result_merger,
            result_ranker=mock_result_ranker,
            performance_optimizer=mock_performance_optimizer,
            config=mock_config
        )
    
    def test_search_handler_initialization(self):
        """SearchHandler 초기화 테스트"""
        mock_semantic_search = MagicMock()
        mock_keyword_mapper = MagicMock()
        mock_data_connector = MagicMock()
        mock_result_merger = MagicMock()
        mock_result_ranker = MagicMock()
        mock_performance_optimizer = MagicMock()
        mock_config = MagicMock()
        
        handler = SearchHandler(
            semantic_search=mock_semantic_search,
            keyword_mapper=mock_keyword_mapper,
            data_connector=mock_data_connector,
            result_merger=mock_result_merger,
            result_ranker=mock_result_ranker,
            performance_optimizer=mock_performance_optimizer,
            config=mock_config
        )
        
        assert handler.semantic_search_engine == mock_semantic_search
        assert handler.data_connector == mock_data_connector
    
    def test_semantic_search(self, search_handler):
        """의미적 검색 테스트"""
        search_handler.semantic_search_engine.search.return_value = [
            {"content": "테스트 문서", "score": 0.9}
        ]
        
        results, count = search_handler.semantic_search("테스트 쿼리", k=5)
        
        assert isinstance(results, list)
        assert isinstance(count, int)
    
    def test_check_cache(self, search_handler):
        """캐시 확인 테스트"""
        mock_state = {
            "retrieved_docs": [],
            "processing_steps": []
        }
        
        with patch('lawfirm_langgraph.core.search.handlers.search_handler.WorkflowUtils') as mock_utils:
            mock_utils.set_state_value = Mock()
            mock_utils.add_step = Mock()
            mock_utils.update_processing_time = Mock()
            
            search_handler.performance_optimizer.cache.get_cached_documents = Mock(return_value=None)
            
            result = search_handler.check_cache(mock_state, "테스트 쿼리", "general", 0.0)
            
            assert isinstance(result, bool)
    
    def test_keyword_search(self, search_handler):
        """키워드 검색 테스트"""
        search_handler.data_connector.search_documents.return_value = [
            {"content": "테스트 문서", "score": 0.8}
        ]
        
        # 실제 코드는 query_type_str이 필수 파라미터
        results, count = search_handler.keyword_search(
            "테스트 쿼리",
            query_type_str="general",  # 필수 파라미터 추가
            limit=5
        )
        
        assert isinstance(results, list)
        assert isinstance(count, int)
    
    def test_hybrid_search(self, search_handler):
        """하이브리드 검색 테스트 - merge_and_rerank_search_results 사용"""
        semantic_results = [
            {"content": "테스트 문서 1", "relevance_score": 0.9, "search_type": "semantic"}
        ]
        keyword_results = [
            {"content": "테스트 문서 2", "relevance_score": 0.8, "search_type": "keyword"}
        ]
        
        optimized_queries = {"original": "테스트 쿼리"}
        rerank_params = {"top_k": 5, "diversity_weight": 0.3}
        
        # result_ranker를 None으로 설정하여 폴백 로직 사용
        search_handler.result_ranker = None
        
        # 실제 코드는 merge_and_rerank_search_results 메서드를 사용
        results = search_handler.merge_and_rerank_search_results(
            semantic_results,
            keyword_results,
            "테스트 쿼리",
            optimized_queries,
            rerank_params
        )
        
        assert isinstance(results, list)
        # 결과가 있으면 combined_score가 추가되어야 함
        if results:
            assert "combined_score" in results[0] or "relevance_score" in results[0]

