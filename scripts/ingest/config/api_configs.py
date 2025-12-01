# -*- coding: utf-8 -*-
"""
API 설정 정의
각 API별 설정을 중앙에서 관리
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class APIConfig:
    """API 설정"""
    target: str
    search_wrapper_key: str
    items_key: str
    table_name: str
    field_mappings: Dict[str, List[str]]  # 필드명 변형 매핑


DLYTRM_CONFIG = APIConfig(
    target='dlytrm',
    search_wrapper_key='dlytrmSearch',
    items_key='items',
    table_name='open_law_dlytrm_data',
    field_mappings={
        'term_id': ['일상용어 id', '일상용어id', '일상용어_id', 'id'],
        'term_name': ['일상용어명', '일상용어 명'],
        'source': ['출처'],
        'term_relation_link': ['용어간관계링크', '용어 간관계링크', '용어간관계 링크']
    }
)

LSTRM_AI_CONFIG = APIConfig(
    target='lstrmAI',
    search_wrapper_key='lstrmAISearch',
    items_key='items',
    table_name='open_law_lstrm_ai_data',
    field_mappings={
        'term_id': ['법령용어 id', '법령용어id', '법령용어_id', 'id'],
        'term_name': ['법령용어명', '법령용어 명'],
        'homonym_exists': ['동음이의어존재여부', '동음이의어 존재여부', '동음이의어존재 여부'],
        'homonym_note': ['비고'],
        'term_relation_link': ['용어간관계링크', '용어 간관계링크', '용어간관계 링크'],
        'article_relation_link': ['조문간관계링크', '조문 간관계링크', '조문간관계 링크']
    }
)

