# -*- coding: utf-8 -*-
"""
벡터 테이블 매핑 설정
새로운 문서 타입 추가 시 여기에만 추가하면 됨
"""

VECTOR_TABLE_MAPPING = {
    'precedent_content': {
        'table_name': 'precedent_chunks',
        'id_column': 'id',
        'vector_column': 'embedding_vector',
        'version_column': 'embedding_version',
        'source_type': 'precedent_content',
        'enabled': True,
        'priority': 1,  # 검색 우선순위
        'weight': 1.0,  # 기본 가중치
        'min_results': 2,  # 최소 보장 결과 수
        'max_results': None  # None이면 제한 없음
    },
    # 🔥 레거시 지원: case_paragraph는 precedent_content로 매핑
    'case_paragraph': {
        'table_name': 'precedent_chunks',
        'id_column': 'id',
        'vector_column': 'embedding_vector',
        'version_column': 'embedding_version',
        'source_type': 'precedent_content',  # 실제 source_type은 precedent_content
        'enabled': True,
        'priority': 1,  # 검색 우선순위
        'weight': 1.0,  # 기본 가중치
        'min_results': 2,  # 최소 보장 결과 수
        'max_results': None  # None이면 제한 없음
    },
    'statute_article': {
        'table_name': 'statute_embeddings',
        'id_column': 'article_id',
        'vector_column': 'embedding_vector',
        'version_column': 'embedding_version',
        'source_type': 'statute_article',
        'enabled': True,
        'priority': 2,
        'weight': 1.3,  # 법령은 더 높은 가중치
        'min_results': 1,
        'max_results': None
    }
    # 해석례와 결정례는 현재 데이터베이스에 없으므로 제외
    # 추후 데이터가 추가되면 아래 항목을 활성화할 수 있습니다:
    # 'interpretation': {
    #     'table_name': 'interpretation_embeddings',
    #     'id_column': 'interpretation_id',
    #     'vector_column': 'embedding_vector',
    #     'version_column': 'embedding_version',
    #     'source_type': 'interpretation',
    #     'enabled': True,
    #     'priority': 3,
    #     'weight': 1.2,
    #     'min_results': 1,
    #     'max_results': None
    # },
    # 'decision': {
    #     'table_name': 'decision_embeddings',
    #     'id_column': 'decision_id',
    #     'vector_column': 'embedding_vector',
    #     'version_column': 'embedding_version',
    #     'source_type': 'decision',
    #     'enabled': True,
    #     'priority': 4,
    #     'weight': 1.1,
    #     'min_results': 1,
    #     'max_results': None
    # }
}

