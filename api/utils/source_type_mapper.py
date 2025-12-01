# -*- coding: utf-8 -*-
"""
source_type과 실제 PostgreSQL 테이블명 간 매핑 유틸리티

실제 사용하는 source_type 값(예: "statute_article", "precedent_content")과 
PostgreSQL 테이블명 간 변환을 제공합니다.
"""
from typing import Dict, Optional, List, Any


# 실제 사용하는 source_type 값 → PostgreSQL 테이블명 매핑
SOURCE_TYPE_TO_TABLE: Dict[str, str] = {
    "statute_article": "statutes_articles",
    "precedent_content": "precedent_contents",
    "precedent_chunk": "precedent_chunks",
    # 하위 호환성을 위한 레거시 매핑
    "case_paragraph": "precedent_contents",  # 레거시: case_paragraph → precedent_content
    # decision_paragraph, interpretation_paragraph, regulation_paragraph는 
    # 해당 테이블이 존재하지 않으므로 매핑하지 않음
}

# PostgreSQL 테이블명 → 실제 사용하는 source_type 값 매핑
TABLE_TO_SOURCE_TYPE: Dict[str, Optional[str]] = {
    "statutes_articles": "statute_article",
    "precedent_contents": "precedent_content",
    "precedent_chunks": "precedent_chunk",
}

# sources_by_type에 사용할 실제 테이블명 목록
SOURCES_BY_TYPE_TABLES: List[str] = [
    "statutes_articles",
    "precedent_contents",
    "precedent_chunks",
]


def source_type_to_table(source_type: str) -> Optional[str]:
    """
    source_type 값을 PostgreSQL 테이블명으로 변환
    
    Args:
        source_type: source_type 값 (예: "statute_article", "precedent_content")
    
    Returns:
        PostgreSQL 테이블명 (예: "statutes_articles", "precedent_contents")
        매핑이 없으면 None 반환
    """
    return SOURCE_TYPE_TO_TABLE.get(source_type)


def table_to_source_type(table_name: str) -> Optional[str]:
    """
    PostgreSQL 테이블명을 source_type 값으로 변환
    
    Args:
        table_name: PostgreSQL 테이블명 (예: "statutes_articles", "precedent_contents")
    
    Returns:
        source_type 값 (예: "statute_article", "precedent_content")
        매핑이 없으면 None 반환
    """
    return TABLE_TO_SOURCE_TYPE.get(table_name)


def convert_sources_by_type_to_table_based(
    sources_by_type: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    source_type 값 기반 sources_by_type을 테이블명 기반으로 변환
    
    Args:
        sources_by_type: source_type 값 기반 구조
            예: {"statute_article": [...], "precedent_content": [...]}
    
    Returns:
        테이블명 기반 구조
            예: {"statutes_articles": [...], "precedent_contents": [...]}
    """
    table_based: Dict[str, List[Dict[str, Any]]] = {
        table: [] for table in SOURCES_BY_TYPE_TABLES
    }
    
    for source_type, items in sources_by_type.items():
        table_name = source_type_to_table(source_type)
        if table_name and table_name in table_based:
            table_based[table_name] = items
        elif source_type in SOURCES_BY_TYPE_TABLES:
            # 이미 테이블명인 경우 그대로 사용
            table_based[source_type] = items
    
    return table_based


def convert_sources_by_type_to_source_type_based(
    sources_by_type: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    테이블명 기반 sources_by_type을 source_type 값 기반으로 변환 (하위 호환성)
    
    Args:
        sources_by_type: 테이블명 기반 구조
            예: {"statutes_articles": [...], "precedent_contents": [...]}
    
    Returns:
        source_type 값 기반 구조
            예: {"statute_article": [...], "precedent_content": [...]}
    """
    source_type_based: Dict[str, List[Dict[str, Any]]] = {}
    
    for table_name, items in sources_by_type.items():
        source_type = table_to_source_type(table_name)
        if source_type:
            source_type_based[source_type] = items
        elif table_name in ["statute_article", "precedent_content", "precedent_chunk",
                           "decision_paragraph", "interpretation_paragraph", "regulation_paragraph",
                           "case_paragraph"]:  # 레거시 하위 호환성
            # 이미 source_type 값인 경우 그대로 사용
            source_type_based[table_name] = items
    
    return source_type_based


def get_default_sources_by_type() -> Dict[str, List]:
    """
    기본 sources_by_type 구조 반환 (테이블명 기반)
    
    Returns:
        테이블명을 키로 하는 빈 sources_by_type 구조
    """
    return {table: [] for table in SOURCES_BY_TYPE_TABLES}


def normalize_source_type(source_type: str) -> str:
    """
    source_type을 정규화 (레거시 값 변환 및 알려진 타입 확인)
    
    Args:
        source_type: 정규화할 source_type 값
    
    Returns:
        정규화된 source_type 값
    """
    if not source_type or source_type.lower() == "unknown":
        return "unknown"
    
    source_type = source_type.strip().lower()
    
    # 레거시 값 변환
    legacy_mapping = {
        "case_paragraph": "precedent_content",
        "precedent_contents": "precedent_content",  # 테이블명 → source_type
        "statutes_articles": "statute_article",  # 테이블명 → source_type
        "precedent_chunks": "precedent_chunk",  # 테이블명 → source_type
    }
    
    if source_type in legacy_mapping:
        return legacy_mapping[source_type]
    
    # 알려진 타입 목록
    known_types = [
        "statute_article",
        "precedent_content",
        "precedent_chunk",
        "decision_paragraph",
        "interpretation_paragraph",
        "regulation_paragraph",
    ]
    
    # 알려진 타입이면 그대로 반환, 아니면 원본 반환 (대소문자 복원)
    if source_type in known_types:
        return source_type
    
    # 알려진 타입이 아니면 원본 반환 (원본 대소문자 유지)
    return source_type

