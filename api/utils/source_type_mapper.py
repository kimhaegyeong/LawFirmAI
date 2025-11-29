# -*- coding: utf-8 -*-
"""
source_type과 실제 PostgreSQL 테이블명 간 매핑 유틸리티

레거시 source_type 값(예: "statute_article", "case_paragraph")과 실제 PostgreSQL 테이블명 간 변환을 제공합니다.
참고: text_chunks 테이블은 더 이상 사용되지 않지만, 하위 호환성을 위해 source_type 매핑은 유지됩니다.
"""
from typing import Dict, Optional, List, Any


# 레거시 source_type 값 → 실제 PostgreSQL 테이블명 매핑
SOURCE_TYPE_TO_TABLE: Dict[str, str] = {
    "statute_article": "statutes_articles",
    "case_paragraph": "precedent_contents",
    # decision_paragraph, interpretation_paragraph, regulation_paragraph는 
    # 해당 테이블이 존재하지 않으므로 매핑하지 않음
}

# 실제 PostgreSQL 테이블명 → 레거시 source_type 값 매핑
TABLE_TO_SOURCE_TYPE: Dict[str, Optional[str]] = {
    "statutes_articles": "statute_article",
    "precedent_contents": "case_paragraph",
    "precedent_chunks": None,  # 별도 벡터 저장소
}

# sources_by_type에 사용할 실제 테이블명 목록
SOURCES_BY_TYPE_TABLES: List[str] = [
    "statutes_articles",
    "precedent_contents",
    "precedent_chunks",
]


def source_type_to_table(source_type: str) -> Optional[str]:
    """
    레거시 source_type 값을 실제 PostgreSQL 테이블명으로 변환
    
    Args:
        source_type: 레거시 source_type 값 (예: "statute_article", "case_paragraph")
    
    Returns:
        실제 PostgreSQL 테이블명 (예: "statutes_articles", "precedent_contents")
        매핑이 없으면 None 반환
    """
    return SOURCE_TYPE_TO_TABLE.get(source_type)


def table_to_source_type(table_name: str) -> Optional[str]:
    """
    실제 PostgreSQL 테이블명을 레거시 source_type 값으로 변환
    
    Args:
        table_name: 실제 PostgreSQL 테이블명 (예: "statutes_articles", "precedent_contents")
    
    Returns:
        레거시 source_type 값 (예: "statute_article", "case_paragraph")
        매핑이 없으면 None 반환
    """
    return TABLE_TO_SOURCE_TYPE.get(table_name)


def convert_sources_by_type_to_table_based(
    sources_by_type: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    source_type 값 기반 sources_by_type을 실제 테이블명 기반으로 변환
    
    Args:
        sources_by_type: source_type 값 기반 구조
            예: {"statute_article": [...], "case_paragraph": [...]}
    
    Returns:
        실제 테이블명 기반 구조
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
    실제 테이블명 기반 sources_by_type을 source_type 값 기반으로 변환 (하위 호환성)
    
    Args:
        sources_by_type: 실제 테이블명 기반 구조
            예: {"statutes_articles": [...], "precedent_contents": [...]}
    
    Returns:
        source_type 값 기반 구조
            예: {"statute_article": [...], "case_paragraph": [...]}
    """
    source_type_based: Dict[str, List[Dict[str, Any]]] = {}
    
    for table_name, items in sources_by_type.items():
        source_type = table_to_source_type(table_name)
        if source_type:
            source_type_based[source_type] = items
        elif table_name in ["statute_article", "case_paragraph", "decision_paragraph", 
                           "interpretation_paragraph", "regulation_paragraph"]:
            # 이미 source_type 값인 경우 그대로 사용 (하위 호환성)
            source_type_based[table_name] = items
    
    return source_type_based


def get_default_sources_by_type() -> Dict[str, List]:
    """
    기본 sources_by_type 구조 반환 (실제 테이블명 기반)
    
    Returns:
        실제 테이블명을 키로 하는 빈 sources_by_type 구조
    """
    return {table: [] for table in SOURCES_BY_TYPE_TABLES}

