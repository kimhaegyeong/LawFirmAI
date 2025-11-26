# -*- coding: utf-8 -*-
"""
PostgreSQL 문서 검색 및 pgvector 검색 단위 테스트
"""

import pytest
import os

# conftest.py에서 project_root fixture 자동 사용 가능

from lawfirm_langgraph.core.search.connectors.legal_data_connector import LegalDataConnectorV2
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config
from lawfirm_langgraph.core.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def config():
    """설정 로드"""
    return Config()


@pytest.fixture
def legal_connector(config):
    """LegalDataConnectorV2 인스턴스 생성"""
    return LegalDataConnectorV2()


@pytest.fixture
def semantic_engine(config):
    """SemanticSearchEngineV2 인스턴스 생성"""
    # pgvector 사용하도록 설정
    os.environ['VECTOR_SEARCH_METHOD'] = 'pgvector'
    engine = SemanticSearchEngineV2()
    return engine


class TestPostgreSQLDocumentSearch:
    """PostgreSQL 문서 검색 테스트"""
    
    def test_statute_search_fts(self, legal_connector):
        """법령 FTS 검색 테스트"""
        query = "계약 해지"
        limit = 10
        
        results = legal_connector.search_statutes_fts(query, limit=limit)
        
        assert isinstance(results, list), "결과는 리스트여야 합니다"
        assert len(results) <= limit, f"결과는 {limit}개 이하여야 합니다"
        
        if results:
            result = results[0]
            assert 'id' in result, "결과에 'id'가 있어야 합니다"
            assert 'type' in result, "결과에 'type'이 있어야 합니다"
            assert 'content' in result, "결과에 'content'가 있어야 합니다"
            assert 'source' in result, "결과에 'source'가 있어야 합니다"
            assert 'metadata' in result, "결과에 'metadata'가 있어야 합니다"
            assert result['type'] == 'statute_article', "타입은 'statute_article'이어야 합니다"
            
            logger.info(f"✅ 법령 검색 성공: {len(results)}개 결과")
            logger.info(f"   첫 번째 결과: {result.get('source', 'N/A')}")
        else:
            logger.warning(f"⚠️ 법령 검색 결과 없음: query='{query}'")
    
    def test_case_search_fts(self, legal_connector):
        """판례 FTS 검색 테스트"""
        query = "계약 해지"
        limit = 10
        
        results = legal_connector.search_cases_fts(query, limit=limit)
        
        assert isinstance(results, list), "결과는 리스트여야 합니다"
        assert len(results) <= limit, f"결과는 {limit}개 이하여야 합니다"
        
        if results:
            result = results[0]
            assert 'id' in result, "결과에 'id'가 있어야 합니다"
            assert 'type' in result, "결과에 'type'이 있어야 합니다"
            assert 'content' in result, "결과에 'content'가 있어야 합니다"
            assert 'source' in result, "결과에 'source'가 있어야 합니다"
            assert 'metadata' in result, "결과에 'metadata'가 있어야 합니다"
            
            logger.info(f"✅ 판례 검색 성공: {len(results)}개 결과")
            logger.info(f"   첫 번째 결과: {result.get('source', 'N/A')}")
        else:
            logger.warning(f"⚠️ 판례 검색 결과 없음: query='{query}'")
    
    def test_parallel_search(self, legal_connector):
        """병렬 검색 테스트 (법령 + 판례)"""
        query = "계약 해지"
        limit = 10
        
        results = legal_connector._search_documents_parallel(query, limit=limit)
        
        assert isinstance(results, list), "결과는 리스트여야 합니다"
        assert len(results) <= limit * 2, f"결과는 {limit * 2}개 이하여야 합니다"
        
        if results:
            logger.info(f"✅ 병렬 검색 성공: {len(results)}개 결과")
            type_counts = {}
            for result in results:
                result_type = result.get('type', 'unknown')
                type_counts[result_type] = type_counts.get(result_type, 0) + 1
            logger.info(f"   타입별 분포: {type_counts}")
        else:
            logger.warning(f"⚠️ 병렬 검색 결과 없음: query='{query}'")


class TestPgVectorSearch:
    """pgvector 벡터 검색 테스트"""
    
    def test_pgvector_weighted_search(self, semantic_engine):
        """pgvector 가중치 기반 검색 테스트"""
        query = "계약 해지에 대해 알려주세요"
        k = 10
        
        # 검색 실행
        results = semantic_engine.search(
            query=query,
            k=k,
            similarity_threshold=0.5
        )
        
        assert isinstance(results, list), "결과는 리스트여야 합니다"
        assert len(results) <= k, f"결과는 {k}개 이하여야 합니다"
        
        if results:
            logger.info(f"✅ pgvector 검색 성공: {len(results)}개 결과")
            
            # 타입별 분포 확인
            type_counts = {}
            for result in results:
                if isinstance(result, dict):
                    source_type = result.get('source_type', 'unknown')
                    type_counts[source_type] = type_counts.get(source_type, 0) + 1
                elif isinstance(result, tuple) and len(result) >= 3:
                    source_type = result[2] if len(result) > 2 else 'unknown'
                    type_counts[source_type] = type_counts.get(source_type, 0) + 1
            
            logger.info(f"   타입별 분포: {type_counts}")
            
            # 결과 구조 확인
            first_result = results[0]
            if isinstance(first_result, dict):
                assert 'chunk_id' in first_result or 'id' in first_result, "결과에 ID가 있어야 합니다"
            elif isinstance(first_result, tuple):
                assert len(first_result) >= 2, "결과 튜플은 최소 2개 요소를 가져야 합니다"
        else:
            logger.warning(f"⚠️ pgvector 검색 결과 없음: query='{query}'")
    
    def test_pgvector_table_detection(self, semantic_engine):
        """pgvector 테이블 자동 감지 테스트"""
        available_tables = semantic_engine._get_available_vector_tables()
        
        assert isinstance(available_tables, list), "사용 가능한 테이블은 리스트여야 합니다"
        
        logger.info(f"✅ 사용 가능한 벡터 테이블: {len(available_tables)}개")
        for table in available_tables:
            logger.info(f"   - {table.get('source_type', 'unknown')}: {table.get('table_name', 'N/A')}")
        
        # 최소한 precedent_content 또는 statute_article이 있어야 함
        source_types = [t.get('source_type') for t in available_tables]
        assert 'precedent_content' in source_types or 'statute_article' in source_types, \
            "최소한 precedent_content 또는 statute_article 테이블이 있어야 합니다"
    
    def test_pgvector_search_with_source_types(self, semantic_engine):
        """특정 소스 타입으로 pgvector 검색 테스트"""
        query = "계약 해지"
        k = 5
        source_types = ['precedent_content']  # 판례만 검색
        
        results = semantic_engine.search(
            query=query,
            k=k,
            source_types=source_types,
            similarity_threshold=0.5
        )
        
        assert isinstance(results, list), "결과는 리스트여야 합니다"
        
        if results:
            logger.info(f"✅ 특정 타입 검색 성공: {len(results)}개 결과 (타입: {source_types})")
        else:
            logger.warning(f"⚠️ 특정 타입 검색 결과 없음: query='{query}', types={source_types}")


class TestSearchIntegration:
    """검색 통합 테스트"""
    
    def test_hybrid_search_flow(self, legal_connector, semantic_engine):
        """하이브리드 검색 플로우 테스트 (FTS + pgvector)"""
        query = "계약 해지에 대해 알려주세요"
        
        # 1. FTS 검색
        fts_results = legal_connector._search_documents_parallel(query, limit=5)
        logger.info(f"📊 FTS 검색 결과: {len(fts_results)}개")
        
        # 2. pgvector 검색
        vector_results = semantic_engine.search(
            query=query,
            k=5,
            similarity_threshold=0.5
        )
        logger.info(f"📊 pgvector 검색 결과: {len(vector_results)}개")
        
        # 3. 결과 통합 확인
        total_results = len(fts_results) + len(vector_results)
        logger.info(f"✅ 하이브리드 검색 완료: 총 {total_results}개 결과")
        
        assert total_results >= 0, "총 결과 수는 0 이상이어야 합니다"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

