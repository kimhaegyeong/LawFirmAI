# -*- coding: utf-8 -*-
"""
LawFirmAI 테스트 설정 및 공통 fixtures
pytest 설정 및 모든 테스트에서 사용할 수 있는 공통 fixtures
"""

import pytest
import tempfile
import os
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 테스트 데이터 경로
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def project_root_path():
    """프로젝트 루트 경로 fixture"""
    return project_root


@pytest.fixture
def temp_dir():
    """임시 디렉토리 fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 정리 작업은 pytest가 자동으로 처리


@pytest.fixture
def temp_db(temp_dir):
    """임시 데이터베이스 fixture"""
    db_path = os.path.join(temp_dir, "test.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_queries():
    """샘플 질의 데이터 fixture"""
    return {
        "precedent_search": "손해배상 관련 판례를 찾아주세요",
        "law_inquiry": "계약 해지 시 손해배상 범위는 어떻게 되나요?",
        "contract_review": "이 계약서를 검토해주세요",
        "legal_advice": "이혼 시 재산분할에 대해 조언해주세요",
        "document_analysis": "이 법률 문서를 분석해주세요",
        "case_study": "유사한 사례가 있는지 찾아주세요"
    }


@pytest.fixture
def sample_legal_documents():
    """샘플 법률 문서 fixture"""
    return {
        "contract": """
        계약서
        
        제1조 (목적)
        본 계약은 부동산 매매에 관한 사항을 규정함을 목적으로 한다.
        
        제2조 (매매대금)
        매매대금은 금 500,000,000원으로 한다.
        
        제3조 (계약해지)
        계약 당사자 중 일방이 계약을 해지하고자 할 때는 상대방에게 30일 전에 통지하여야 한다.
        """,
        "precedent": """
        대법원 2023다12345 판례
        
        【판시사항】
        계약 해지로 인한 손해배상의 범위에 관한 판례
        
        【판결요지】
        계약의 해지로 인한 손해배상은 계약 이행으로 얻을 수 있었던 이익의 상실을 의미하며,
        민법 제543조와 제544조에 따라 그 범위가 결정된다.
        """,
        "law_article": """
        민법 제543조 (계약의 해지)
        계약의 해지로 인한 손해배상은 계약 이행으로 얻을 수 있었던 이익의 상실을 의미한다.
        
        민법 제544조 (해지로 인한 손해배상의 범위)
        해지로 인한 손해배상의 범위는 계약 당사자가 예견하거나 예견할 수 있었던 손해에 한한다.
        """
    }


@pytest.fixture
def mock_chat_service():
    """Mock ChatService fixture"""
    mock_service = Mock()
    mock_service.process_message.return_value = "테스트 응답입니다."
    mock_service.get_conversation_history.return_value = []
    mock_service.clear_conversation.return_value = True
    return mock_service


@pytest.fixture
def mock_rag_service():
    """Mock RAG Service fixture"""
    mock_service = Mock()
    mock_service.search.return_value = {
        "results": [
            {
                "content": "테스트 검색 결과 1",
                "score": 0.95,
                "source": "test_source_1"
            },
            {
                "content": "테스트 검색 결과 2", 
                "score": 0.87,
                "source": "test_source_2"
            }
        ],
        "total_results": 2
    }
    mock_service.generate_answer.return_value = {
        "answer": "테스트 생성된 답변입니다.",
        "confidence": 0.92,
        "sources": ["test_source_1", "test_source_2"]
    }
    return mock_service


@pytest.fixture
def mock_database():
    """Mock Database fixture"""
    mock_db = Mock()
    mock_db.execute_query.return_value = [
        {"id": 1, "content": "테스트 데이터 1"},
        {"id": 2, "content": "테스트 데이터 2"}
    ]
    mock_db.insert_data.return_value = True
    mock_db.update_data.return_value = True
    mock_db.delete_data.return_value = True
    return mock_db


@pytest.fixture
def mock_vector_store():
    """Mock Vector Store fixture"""
    mock_store = Mock()
    mock_store.search_similar.return_value = [
        {"content": "유사한 문서 1", "similarity": 0.95},
        {"content": "유사한 문서 2", "similarity": 0.87}
    ]
    mock_store.add_documents.return_value = True
    mock_store.build_index.return_value = True
    return mock_store


@pytest.fixture
def test_config():
    """테스트 설정 fixture"""
    return {
        "database_url": "sqlite:///test.db",
        "model_path": "./models",
        "vector_store_path": "./data/embeddings",
        "max_tokens": 1000,
        "temperature": 0.7,
        "log_level": "INFO"
    }


@pytest.fixture
def sample_conversation_history():
    """샘플 대화 기록 fixture"""
    return [
        {
            "role": "user",
            "content": "계약 해지에 대해 알려주세요",
            "timestamp": "2024-01-01T10:00:00Z"
        },
        {
            "role": "assistant", 
            "content": "계약 해지는 민법 제543조에 따라...",
            "timestamp": "2024-01-01T10:00:05Z"
        },
        {
            "role": "user",
            "content": "손해배상 범위는 어떻게 되나요?",
            "timestamp": "2024-01-01T10:01:00Z"
        },
        {
            "role": "assistant",
            "content": "손해배상 범위는 민법 제544조에 따라...",
            "timestamp": "2024-01-01T10:01:05Z"
        }
    ]


@pytest.fixture
def performance_metrics():
    """성능 메트릭 fixture"""
    return {
        "response_time": 1.5,  # 초
        "memory_usage": 512,   # MB
        "cpu_usage": 25.0,     # %
        "throughput": 100,     # requests/min
        "accuracy": 0.92,      # 정확도
        "confidence": 0.88     # 신뢰도
    }


@pytest.fixture
def error_scenarios():
    """에러 시나리오 fixture"""
    return {
        "invalid_input": "",
        "too_long_input": "x" * 10000,
        "special_characters": "!@#$%^&*()",
        "sql_injection": "'; DROP TABLE users; --",
        "xss_attempt": "<script>alert('xss')</script>",
        "unicode_input": "한글 테스트 🚀"
    }


@pytest.fixture(scope="session")
def test_data_files():
    """테스트 데이터 파일들 fixture"""
    return {
        "legal_terms": TEST_DATA_DIR / "legal_terms.json",
        "precedents": TEST_DATA_DIR / "precedents.json", 
        "contracts": TEST_DATA_DIR / "contracts.json",
        "queries": TEST_DATA_DIR / "queries.json"
    }


# pytest 설정
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "unit: 단위 테스트"
    )
    config.addinivalue_line(
        "markers", "integration: 통합 테스트"
    )
    config.addinivalue_line(
        "markers", "performance: 성능 테스트"
    )
    config.addinivalue_line(
        "markers", "slow: 느린 테스트"
    )


def pytest_collection_modifyitems(config, items):
    """테스트 아이템 수정"""
    for item in items:
        # 파일 경로에 따라 자동으로 마커 추가
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # 느린 테스트 마커 추가 (임의 기준)
        if "benchmark" in str(item.fspath) or "stress" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
