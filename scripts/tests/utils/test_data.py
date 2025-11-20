#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 데이터 생성 유틸리티
"""

from typing import Dict, Any, Optional, List


# 표준 테스트 쿼리 데이터셋
TEST_QUERIES = {
    "민사법": [
        {
            "query": "계약 해지 사유에 대해 알려주세요",
            "query_type": "law_inquiry",
            "expected_keywords": ["계약", "해지", "사유"],
            "domain": "민사법"
        },
        {
            "query": "손해배상 청구권의 소멸시효는?",
            "query_type": "legal_advice",
            "expected_keywords": ["손해배상", "청구권", "소멸시효"],
            "domain": "민사법"
        },
        {
            "query": "전세금 반환 보증에 대해 알려주세요",
            "query_type": "law_inquiry",
            "expected_keywords": ["전세금", "반환", "보증"],
            "domain": "민사법"
        }
    ],
    "노동법": [
        {
            "query": "고용계약기간을 단축하는 것이 가능한지",
            "query_type": "legal_advice",
            "expected_keywords": ["고용계약", "기간", "단축"],
            "domain": "노동법"
        },
        {
            "query": "임신 중인 여성근로자의 휴일대체 가능 여부",
            "query_type": "legal_advice",
            "expected_keywords": ["임신", "여성근로자", "휴일대체"],
            "domain": "노동법"
        }
    ],
    "판례": [
        {
            "query": "불법행위로 인한 손해배상 판례를 찾아주세요",
            "query_type": "precedent_search",
            "expected_keywords": ["불법행위", "손해배상", "판례"],
            "domain": "민사법"
        }
    ]
}


class TestDataFactory:
    """테스트 데이터 생성 팩토리 클래스"""
    
    @staticmethod
    def create_version_info(
        version_name: str = "v1.0.0-test",
        embedding_version_id: int = 1,
        chunking_strategy: str = "standard",
        chunking_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        document_count: int = 100,
        total_chunks: int = 1000,
        status: str = "active"
    ) -> Dict[str, Any]:
        """버전 정보 딕셔너리 생성"""
        if chunking_config is None:
            chunking_config = {"chunk_size": 1000, "chunk_overlap": 200}
        
        if embedding_config is None:
            embedding_config = {"model": "test-model", "dimension": 768}
        
        return {
            "version": version_name,
            "embedding_version_id": embedding_version_id,
            "chunking_strategy": chunking_strategy,
            "chunking_config": chunking_config,
            "embedding_config": embedding_config,
            "document_count": document_count,
            "total_chunks": total_chunks,
            "status": status
        }
    
    @staticmethod
    def create_chunk_data(
        chunk_id: int = 1,
        embedding_version_id: int = 1,
        source_type: str = "statute_article",
        source_id: int = 1,
        chunk_index: int = 0,
        content: str = "Test chunk content"
    ) -> Dict[str, Any]:
        """청크 데이터 딕셔너리 생성"""
        return {
            "id": chunk_id,
            "embedding_version_id": embedding_version_id,
            "source_type": source_type,
            "source_id": source_id,
            "chunk_index": chunk_index,
            "content": content,
            "metadata": {}
        }
    
    @staticmethod
    def create_test_query(
        query: Optional[str] = None,
        query_type: str = "law_inquiry",
        domain: str = "민사법"
    ) -> Dict[str, Any]:
        """테스트 쿼리 생성"""
        if query is None:
            queries = TEST_QUERIES.get(domain, TEST_QUERIES["민사법"])
            query_data = queries[0] if queries else {
                "query": "테스트 질문",
                "query_type": query_type,
                "expected_keywords": [],
                "domain": domain
            }
            return query_data
        
        return {
            "query": query,
            "query_type": query_type,
            "expected_keywords": [],
            "domain": domain
        }
    
    @staticmethod
    def create_workflow_result(
        answer: str = "테스트 답변",
        sources: Optional[List[Dict[str, Any]]] = None,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.9
    ) -> Dict[str, Any]:
        """워크플로우 결과 생성"""
        if sources is None:
            sources = []
        if retrieved_docs is None:
            retrieved_docs = []
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "confidence": confidence,
            "sources_detail": []
        }
    
    @staticmethod
    def create_search_result(
        content: str = "테스트 검색 결과",
        similarity: float = 0.85,
        source_type: str = "statute_article",
        source_id: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """검색 결과 생성"""
        if metadata is None:
            metadata = {}
        
        return {
            "content": content,
            "similarity": similarity,
            "score": similarity,
            "source_type": source_type,
            "source_id": source_id,
            "metadata": metadata
        }


def create_version_info(
    version_name: str = "v1.0.0-test",
    **kwargs
) -> Dict[str, Any]:
    """버전 정보 생성 (편의 함수)"""
    return TestDataFactory.create_version_info(version_name, **kwargs)


def create_chunk_data(**kwargs) -> Dict[str, Any]:
    """청크 데이터 생성 (편의 함수)"""
    return TestDataFactory.create_chunk_data(**kwargs)


def create_test_query(**kwargs) -> Dict[str, Any]:
    """테스트 쿼리 생성 (편의 함수)"""
    return TestDataFactory.create_test_query(**kwargs)


def create_workflow_result(**kwargs) -> Dict[str, Any]:
    """워크플로우 결과 생성 (편의 함수)"""
    return TestDataFactory.create_workflow_result(**kwargs)


def create_search_result(**kwargs) -> Dict[str, Any]:
    """검색 결과 생성 (편의 함수)"""
    return TestDataFactory.create_search_result(**kwargs)


def get_test_queries(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """테스트 쿼리 목록 조회"""
    if domain:
        return TEST_QUERIES.get(domain, [])
    
    all_queries = []
    for queries in TEST_QUERIES.values():
        all_queries.extend(queries)
    return all_queries

