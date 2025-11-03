# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures
테스트 설정 및 공통 픽스처
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))

# 환경 변수 설정 (테스트용)
os.environ.setdefault("LANGGRAPH_ENABLED", "true")
os.environ.setdefault("ENABLE_CHECKPOINT", "true")
os.environ.setdefault("CHECKPOINT_STORAGE", "memory")
os.environ.setdefault("LLM_PROVIDER", "google")
os.environ.setdefault("GOOGLE_MODEL", "gemini-2.5-flash-lite")
os.environ.setdefault("USE_AGENTIC_MODE", "false")


@pytest.fixture
def mock_config():
    """Mock LangGraphConfig 픽스처"""
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
    
    config = LangGraphConfig(
        enable_checkpoint=True,
        checkpoint_storage=CheckpointStorageType.MEMORY,
        checkpoint_db_path="./data/checkpoints/test_langgraph.db",
        checkpoint_ttl=3600,
        max_iterations=10,
        recursion_limit=25,
        enable_streaming=True,
        llm_provider="google",
        google_model="gemini-2.5-flash-lite",
        google_api_key="test_api_key",
        langgraph_enabled=True,
        langfuse_enabled=False,
        langsmith_enabled=False,
        use_agentic_mode=False,
    )
    return config


@pytest.fixture
def mock_workflow_state():
    """Mock 워크플로우 상태 픽스처"""
    return {
        "query": "테스트 질문",
        "answer": "",
        "context": [],
        "retrieved_docs": [],
        "processing_steps": [],
        "errors": [],
        "session_id": "test_session",
        "conversation_history": [],
        "classification": {
            "legal_field": "contract",
            "complexity": "medium",
            "urgency": "normal",
        },
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM 응답 픽스처"""
    return {
        "content": "테스트 답변입니다.",
        "metadata": {
            "model": "gemini-2.5-flash-lite",
            "tokens_used": 100,
        },
    }


@pytest.fixture
def mock_search_results():
    """Mock 검색 결과 픽스처"""
    return [
        {
            "content": "법률 문서 내용 1",
            "metadata": {
                "source": "test_source_1",
                "similarity": 0.85,
            },
        },
        {
            "content": "법률 문서 내용 2",
            "metadata": {
                "source": "test_source_2",
                "similarity": 0.75,
            },
        },
    ]


@pytest.fixture
def mock_workflow_service(mock_config):
    """Mock LangGraphWorkflowService 픽스처"""
    with patch('lawfirm_langgraph.langgraph_core.workflow.workflow_service.LangGraphWorkflowService') as MockService:
        service = MockService.return_value
        service.config = mock_config
        service.process_query_async = Mock(return_value={
            "answer": "테스트 답변",
            "context": [],
            "retrieved_docs": [],
            "processing_steps": ["step1", "step2"],
            "errors": [],
        })
        yield service


@pytest.fixture
def mock_legal_data_connector():
    """Mock LegalDataConnector 픽스처"""
    connector = MagicMock()
    connector.search = Mock(return_value=[
        {
            "content": "검색 결과 1",
            "metadata": {"source": "test", "similarity": 0.8},
        }
    ])
    connector.get_document = Mock(return_value={
        "content": "문서 내용",
        "metadata": {"source": "test"},
    })
    return connector


@pytest.fixture
def mock_answer_generator():
    """Mock AnswerGenerator 픽스처"""
    generator = MagicMock()
    generator.generate = Mock(return_value="생성된 답변")
    generator.generate_async = Mock(return_value="생성된 답변")
    return generator


@pytest.fixture
def cleanup_test_files():
    """테스트 파일 정리 픽스처"""
    test_files = []
    
    def add_file(file_path: str):
        test_files.append(file_path)
    
    yield add_file
    
    # 테스트 후 정리
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """테스트 환경 설정"""
    # 테스트용 환경 변수 설정
    monkeypatch.setenv("LANGGRAPH_ENABLED", "true")
    monkeypatch.setenv("ENABLE_CHECKPOINT", "true")
    monkeypatch.setenv("CHECKPOINT_STORAGE", "memory")
    monkeypatch.setenv("USE_AGENTIC_MODE", "false")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    
    # 로깅 레벨 설정
    import logging
    logging.getLogger().setLevel(logging.WARNING)

