# -*- coding: utf-8 -*-
"""
Search 테스트 공통 Fixtures 및 유틸리티
"""

import sys
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가 (한 번만)
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 공통 imports (sys.path 설정 후)
import pytest  # noqa: E402
from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker  # noqa: E402


@pytest.fixture(scope="session")
def project_root():
    """프로젝트 루트 경로"""
    return _project_root


@pytest.fixture(scope="session")
def db_path(project_root):
    """데이터베이스 경로 찾기"""
    possible_paths = [
        "data/lawfirm_v2.db",
        "./data/lawfirm_v2.db",
        str(project_root / "data" / "lawfirm_v2.db")
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None


@pytest.fixture
def result_ranker():
    """ResultRanker 인스턴스 (Cross-Encoder 없이)"""
    return ResultRanker(use_cross_encoder=False)


@pytest.fixture
def sample_db_docs():
    """샘플 DB 검색 결과"""
    return [
        {
            "id": "db_1",
            "content": "민법 제543조 계약 해지",
            "relevance_score": 0.15,
            "search_type": "database",
            "source": "민법"
        },
        {
            "id": "db_2",
            "content": "민법 제544조 계약 해제",
            "relevance_score": 0.25,
            "search_type": "database",
            "source": "민법"
        },
        {
            "id": "db_3",
            "content": "민법 제545조 계약 해지 사유",
            "relevance_score": 0.35,
            "search_type": "database",
            "source": "민법"
        }
    ]


@pytest.fixture
def sample_vector_docs():
    """샘플 벡터 검색 결과"""
    return [
        {
            "id": "vec_1",
            "content": "계약 해지에 대한 법률 규정",
            "similarity": 0.45,
            "relevance_score": 0.45,
            "search_type": "semantic",
            "source": "법령"
        },
        {
            "id": "vec_2",
            "content": "계약 해지 사유와 절차",
            "similarity": 0.65,
            "relevance_score": 0.65,
            "search_type": "semantic",
            "source": "법령"
        },
        {
            "id": "vec_3",
            "content": "계약 해지의 법적 효과",
            "similarity": 0.85,
            "relevance_score": 0.85,
            "search_type": "semantic",
            "source": "법령"
        }
    ]


@pytest.fixture
def sample_keyword_docs():
    """샘플 키워드 검색 결과"""
    return [
        {
            "id": "kw_1",
            "content": "계약 해지",
            "relevance_score": 0.3,
            "search_type": "keyword",
            "source": "키워드"
        },
        {
            "id": "kw_2",
            "content": "계약 해지 사유",
            "relevance_score": 0.5,
            "search_type": "keyword",
            "source": "키워드"
        },
        {
            "id": "kw_3",
            "content": "계약 해지 절차",
            "relevance_score": 0.7,
            "search_type": "keyword",
            "source": "키워드"
        }
    ]


def create_sample_documents(count: int = 20) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    샘플 문서 생성 (공통 유틸리티 함수)
    
    Args:
        count: 생성할 총 문서 수
        
    Returns:
        (db_docs, vector_docs, keyword_docs) 튜플
    """
    db_docs = []
    vector_docs = []
    keyword_docs = []
    
    db_count = count // 3
    vector_count = count // 3
    keyword_count = count - db_count - vector_count
    
    for i in range(db_count):
        db_docs.append({
            "id": f"db_{i+1}",
            "content": f"민법 제{543+i}조 계약 해지 관련 조문",
            "text": f"민법 제{543+i}조 계약 해지 관련 조문",
            "relevance_score": 0.1 + (i * 0.02),
            "search_type": "database",
            "source": "민법"
        })
    
    for i in range(vector_count):
        vector_docs.append({
            "id": f"vec_{i+1}",
            "content": f"계약 해지에 대한 법률 규정 {i+1}",
            "text": f"계약 해지에 대한 법률 규정 {i+1}",
            "similarity": 0.3 + (i * 0.03),
            "relevance_score": 0.3 + (i * 0.03),
            "search_type": "semantic",
            "source": "법령"
        })
    
    for i in range(keyword_count):
        keyword_docs.append({
            "id": f"kw_{i+1}",
            "content": f"계약 해지 {i+1}",
            "text": f"계약 해지 {i+1}",
            "relevance_score": 0.2 + (i * 0.04),
            "search_type": "keyword",
            "source": "키워드"
        })
    
    return db_docs, vector_docs, keyword_docs

