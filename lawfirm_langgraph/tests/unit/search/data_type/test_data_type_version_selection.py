# -*- coding: utf-8 -*-
"""
data_type별 활성 버전 선택 단위 테스트 (PostgreSQL 전용)

이 테스트는 PostgreSQL 데이터베이스를 사용합니다.
SQLite는 지원하지 않습니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
except ImportError:
    from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2


class TestDataTypeVersionSelection:
    """data_type별 활성 버전 선택 테스트 클래스"""
    
    @pytest.fixture
    def engine(self):
        """SemanticSearchEngineV2 인스턴스 생성 (PostgreSQL 전용)"""
        with patch('lawfirm_langgraph.core.search.engines.semantic_search_engine_v2.DatabaseAdapter'):
            # PostgreSQL URL 사용 (실제 연결은 Mock으로 처리)
            engine = SemanticSearchEngineV2(
                db_path='postgresql://test:test@localhost:5432/test'
            )
            engine.logger = Mock()
            engine.logger.info = Mock()
            engine.logger.warning = Mock()
            engine.logger.debug = Mock()
            engine.logger.error = Mock()
            engine._db_adapter = Mock()
            engine._db_adapter.db_type = 'postgresql'
            engine._db_adapter.get_connection_context = Mock()
            return engine
    
    def test_determine_data_type_from_source_types_statutes(self, engine):
        """statute_article source_type이 statutes data_type으로 매핑되는지 테스트"""
        source_types = ['statute_article']
        data_type = engine._determine_data_type_from_source_types(source_types)
        assert data_type == 'statutes', f"Expected 'statutes', got {data_type}"
    
    def test_determine_data_type_from_source_types_precedents(self, engine):
        """case_paragraph source_type이 precedents data_type으로 매핑되는지 테스트"""
        source_types = ['case_paragraph']
        data_type = engine._determine_data_type_from_source_types(source_types)
        assert data_type == 'precedents', f"Expected 'precedents', got {data_type}"
    
    def test_determine_data_type_from_source_types_mixed(self, engine):
        """혼합된 source_types는 None을 반환하는지 테스트"""
        source_types = ['statute_article', 'case_paragraph']
        data_type = engine._determine_data_type_from_source_types(source_types)
        assert data_type is None, f"Expected None for mixed types, got {data_type}"
    
    def test_determine_data_type_from_source_types_empty(self, engine):
        """빈 source_types는 None을 반환하는지 테스트"""
        source_types = []
        data_type = engine._determine_data_type_from_source_types(source_types)
        assert data_type is None, f"Expected None for empty types, got {data_type}"
    
    def test_determine_data_type_from_source_types_none(self, engine):
        """None source_types는 None을 반환하는지 테스트"""
        source_types = None
        data_type = engine._determine_data_type_from_source_types(source_types)
        assert data_type is None, f"Expected None for None types, got {data_type}"
    
    def test_get_active_embedding_version_id_with_data_type(self, engine):
        """data_type 파라미터가 전달되면 해당 타입의 활성 버전을 조회하는지 테스트"""
        # Mock 설정
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # statutes 타입 활성 버전 반환
        mock_row = {
            'id': 3,
            'version': 2,
            'data_type': 'statutes',
            'is_active': True
        }
        mock_cursor.fetchone.return_value = mock_row
        
        engine._get_connection_context = Mock(return_value=MagicMock(__enter__=Mock(return_value=mock_conn), __exit__=Mock()))
        
        result = engine._get_active_embedding_version_id(data_type='statutes')
        
        # 검증: SQL 쿼리에 data_type 조건이 포함되었는지 확인
        assert mock_cursor.execute.called
        execute_call = str(mock_cursor.execute.call_args)
        assert 'data_type' in execute_call or '%s' in execute_call, "data_type 파라미터가 SQL 쿼리에 포함되지 않았습니다"
        assert result == 3, f"Expected version ID 3, got {result}"
    
    def test_get_active_embedding_version_id_without_data_type(self, engine):
        """data_type 파라미터가 없으면 첫 번째 활성 버전을 조회하는지 테스트"""
        # Mock 설정
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # 첫 번째 활성 버전 반환
        mock_row = {
            'id': 1,
            'version': 1,
            'data_type': 'precedents',
            'is_active': True
        }
        mock_cursor.fetchone.return_value = mock_row
        
        engine._get_connection_context = Mock(return_value=MagicMock(__enter__=Mock(return_value=mock_conn), __exit__=Mock()))
        
        result = engine._get_active_embedding_version_id()
        
        # 검증: 하위 호환성 유지
        assert result == 1, f"Expected version ID 1, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

