# -*- coding: utf-8 -*-
"""
SearchService 테스트
검색 서비스 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# LegalModelManager가 없으므로 mock으로 대체
import sys
from unittest.mock import MagicMock

# search_service 모듈을 import하기 전에 LegalModelManager를 mock
mock_model_manager = MagicMock()
sys.modules['lawfirm_langgraph.core.models.model_manager'] = MagicMock()
sys.modules['lawfirm_langgraph.core.models.model_manager'].LegalModelManager = mock_model_manager

from lawfirm_langgraph.core.services.search_service import MLEnhancedSearchService
from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
from lawfirm_langgraph.core.data.vector_store import LegalVectorStore
from lawfirm_langgraph.core.utils.config import Config


class TestMLEnhancedSearchService:
    """MLEnhancedSearchService 테스트"""
    
    @pytest.fixture
    def config(self):
        """테스트용 설정"""
        config = Config()
        config.database_path = ":memory:"
        return config
    
    @pytest.fixture
    def mock_database(self):
        """Mock LegalDataConnectorV2"""
        db = MagicMock()
        db.execute_query = Mock(return_value=[])
        return db
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        vs = MagicMock()
        vs.search = Mock(return_value=[])
        return vs
    
    @pytest.fixture
    def search_service(self, config, mock_database, mock_vector_store):
        """MLEnhancedSearchService 인스턴스"""
        with patch('lawfirm_langgraph.core.services.search_service.AIKeywordGenerator'):
            service = MLEnhancedSearchService(
                config=config,
                database=mock_database,
                vector_store=mock_vector_store
            )
            return service
    
    def test_search_service_initialization(self, config, mock_database, mock_vector_store):
        """SearchService 초기화 테스트"""
        with patch('lawfirm_langgraph.core.services.search_service.AIKeywordGenerator'):
            service = MLEnhancedSearchService(
                config=config,
                database=mock_database,
                vector_store=mock_vector_store
            )
            
            assert service.config == config
            assert service.database == mock_database
            assert service.vector_store == mock_vector_store
            assert service.use_ml_enhanced_search is True
            assert service.quality_threshold == 0.7
            assert service.confidence_threshold == 0.6
    
    def test_search_documents_semantic(self, search_service, mock_vector_store):
        """의미적 검색 테스트"""
        mock_vector_store.search.return_value = [
            {
                "text": "테스트 문서 내용",
                "score": 0.85,
                "metadata": {
                    "document_id": "doc1",
                    "law_name": "테스트 법률",
                    "article_number": "1조",
                    "article_title": "테스트 조항"
                }
            }
        ]
        
        with patch.object(search_service, '_filter_and_score_documents', return_value=mock_vector_store.search.return_value):
            results = search_service.search_documents("테스트 쿼리", search_type="semantic", limit=5)
            
            assert isinstance(results, list)
            if results:
                assert "document_id" in results[0]
                assert "title" in results[0]
                assert "content" in results[0]
                assert "similarity" in results[0]
                assert results[0]["search_type"] == "semantic"
    
    def test_search_documents_keyword(self, search_service):
        """키워드 검색 테스트"""
        with patch.object(search_service, '_ml_enhanced_keyword_search', return_value=[]):
            results = search_service.search_documents("테스트 쿼리", search_type="keyword", limit=5)
            
            assert isinstance(results, list)
    
    def test_search_documents_hybrid(self, search_service):
        """하이브리드 검색 테스트"""
        with patch.object(search_service, '_ml_enhanced_hybrid_search', return_value=[]):
            results = search_service.search_documents("테스트 쿼리", search_type="hybrid", limit=5)
            
            assert isinstance(results, list)
    
    def test_search_documents_supplementary(self, search_service):
        """보충 조항 검색 테스트"""
        with patch.object(search_service, '_search_supplementary_provisions', return_value=[]):
            results = search_service.search_documents("테스트 쿼리", search_type="supplementary", limit=5)
            
            assert isinstance(results, list)
    
    def test_search_documents_high_quality(self, search_service):
        """고품질 문서 검색 테스트"""
        with patch.object(search_service, '_search_high_quality_documents', return_value=[]):
            results = search_service.search_documents("테스트 쿼리", search_type="high_quality", limit=5)
            
            assert isinstance(results, list)
    
    def test_search_documents_invalid_type(self, search_service):
        """잘못된 검색 타입 테스트"""
        results = search_service.search_documents("테스트 쿼리", search_type="invalid_type", limit=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_documents_error_handling(self, search_service, mock_vector_store):
        """에러 핸들링 테스트"""
        mock_vector_store.search.side_effect = Exception("테스트 에러")
        
        results = search_service.search_documents("테스트 쿼리", search_type="semantic", limit=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_documents_with_filters(self, search_service, mock_vector_store):
        """필터를 사용한 검색 테스트"""
        mock_vector_store.search.return_value = []
        
        with patch.object(search_service, '_filter_and_score_documents', return_value=[]):
            filters = {"document_type": "statute"}
            results = search_service.search_documents("테스트 쿼리", search_type="semantic", limit=5, filters=filters)
            
            assert isinstance(results, list)
    
    def test_ml_enhanced_semantic_search(self, search_service, mock_vector_store):
        """ML 강화 의미적 검색 테스트"""
        mock_vector_store.search.return_value = [
            {
                "text": "테스트 문서",
                "score": 0.9,
                "metadata": {
                    "document_id": "doc1",
                    "law_name": "테스트 법률"
                }
            }
        ]
        
        with patch.object(search_service, '_filter_and_score_documents', return_value=mock_vector_store.search.return_value):
            results = search_service._ml_enhanced_semantic_search("테스트 쿼리", limit=5, filters=None)
            
            assert isinstance(results, list)
            if results:
                assert results[0]["search_type"] == "semantic"
    
    def test_ml_enhanced_semantic_search_error(self, search_service, mock_vector_store):
        """ML 강화 의미적 검색 에러 핸들링"""
        mock_vector_store.search.side_effect = Exception("테스트 에러")
        
        results = search_service._ml_enhanced_semantic_search("테스트 쿼리", limit=5, filters=None)
        
        assert isinstance(results, list)
        assert len(results) == 0

