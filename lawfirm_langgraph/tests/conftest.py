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
    with patch('lawfirm_langgraph.core.workflow.workflow_service.LangGraphWorkflowService') as MockService:
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
def mock_chat_service():
    """Mock ChatService 픽스처"""
    from lawfirm_langgraph.core.utils.config import Config
    
    with patch('lawfirm_langgraph.core.services.chat_service.HybridSearchEngineV2'):
        with patch('lawfirm_langgraph.core.services.chat_service.QuestionClassifier'):
            with patch('lawfirm_langgraph.core.services.chat_service.ImprovedAnswerGenerator'):
                with patch('lawfirm_langgraph.core.services.chat_service.IntegratedSessionManager'):
                    with patch('lawfirm_langgraph.core.services.chat_service.MultiTurnQuestionHandler'):
                        with patch('lawfirm_langgraph.core.services.chat_service.ContextCompressor'):
                            with patch('lawfirm_langgraph.core.services.chat_service.UserProfileManager'):
                                with patch('lawfirm_langgraph.core.services.chat_service.EmotionIntentAnalyzer'):
                                    with patch('lawfirm_langgraph.core.services.chat_service.ConversationFlowTracker'):
                                        with patch('lawfirm_langgraph.core.services.chat_service.ContextualMemoryManager'):
                                            with patch('lawfirm_langgraph.core.services.chat_service.ConversationQualityMonitor'):
                                                with patch('lawfirm_langgraph.core.services.chat_service.PerformanceMonitor'):
                                                    with patch('lawfirm_langgraph.core.services.chat_service.MemoryOptimizer'):
                                                        with patch('lawfirm_langgraph.core.services.chat_service.CacheManager'):
                                                            from lawfirm_langgraph.core.services.chat_service import ChatService
                                                            config = Config()
                                                            config.database_path = ":memory:"
                                                            service = ChatService(config)
                                                            return service


@pytest.fixture
def mock_database():
    """Mock DatabaseManager 픽스처"""
    db = MagicMock()
    db.execute_query = Mock(return_value=[])
    db.execute_update = Mock(return_value=1)
    db.get_connection = MagicMock()
    return db


@pytest.fixture
def temp_database():
    """임시 데이터베이스 픽스처"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


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
    
    # 로깅 레벨 설정 (환경 변수로 제어 가능)
    import logging
    import os
    
    # 환경 변수에서 로깅 레벨 읽기 (기본값: WARNING)
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str, logging.WARNING)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 핸들러가 없으면 추가
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(handler)
    
    # lawfirm_langgraph 로거 설정
    langgraph_logger = logging.getLogger("lawfirm_langgraph")
    langgraph_logger.setLevel(log_level)
    langgraph_logger.propagate = True


@pytest.fixture
def workflow_config():
    """EnhancedLegalQuestionWorkflow용 설정 픽스처"""
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
    
    return LangGraphConfig(
        enable_checkpoint=True,
        checkpoint_storage=CheckpointStorageType.MEMORY,
        langgraph_enabled=True,
        use_agentic_mode=False,
        llm_provider="google",
        google_model="gemini-2.5-flash-lite",
        google_api_key="test_api_key",
        max_iterations=10,
        recursion_limit=25,
    )


@pytest.fixture
def workflow_instance(workflow_config):
    """EnhancedLegalQuestionWorkflow 인스턴스 픽스처"""
    with patch('core.services.semantic_search_engine_v2.SemanticSearchEngineV2'):
        with patch('core.services.multi_turn_handler.MultiTurnQuestionHandler'):
            with patch('core.services.ai_keyword_generator.AIKeywordGenerator'):
                with patch('core.agents.handlers.direct_answer_handler.DirectAnswerHandler'):
                    with patch('core.agents.handlers.classification_handler.ClassificationHandler'):
                        with patch('core.agents.handlers.answer_generator.AnswerGenerator'):
                            with patch('core.agents.handlers.answer_formatter.AnswerFormatterHandler'):
                                from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
                                return EnhancedLegalQuestionWorkflow(workflow_config)


@pytest.fixture
def workflow_state():
    """표준 워크플로우 State 픽스처"""
    return {
        "input": {
            "query": "전세금 반환 보증에 대해 설명해주세요",
            "session_id": "test_session_123"
        },
        "query": "전세금 반환 보증에 대해 설명해주세요",
        "query_type": "legal_advice",
        "query_complexity": "moderate",
        "needs_search": True,
        "retrieved_docs": [
            {
                "content": "전세금 반환 보증 관련 내용입니다.",
                "type": "statute_article",
                "source": "주택임대차보호법",
                "relevance_score": 0.9,
                "source_type": "statute_article",
                "source_id": "test-1"
            }
        ],
        "semantic_results": [],
        "keyword_results": [],
        "semantic_count": 0,
        "keyword_count": 0,
        "search_params": {},
        "extracted_keywords": ["전세금", "반환", "보증"],
        "legal_field": "부동산법",
        "legal_domain": "real_estate",
        "confidence": 0.85,
        "metadata": {},
        "common": {
            "metadata": {}
        },
        "search": {},
        "processing_steps": [],
        "errors": []
    }


@pytest.fixture
def simple_query_state():
    """간단한 질문용 State 픽스처"""
    return {
        "input": {
            "query": "계약이란 무엇인가요?",
            "session_id": "test_session_456"
        },
        "query": "계약이란 무엇인가요?",
        "query_type": "definition",
        "query_complexity": "simple",
        "needs_search": False,
        "processing_steps": [],
        "errors": [],
        "common": {
            "metadata": {}
        }
    }


@pytest.fixture
def complex_query_state():
    """복잡한 질문용 State 픽스처"""
    return {
        "input": {
            "query": "전세 계약 시 발생할 수 있는 법적 분쟁과 해결 방법은?",
            "session_id": "test_session_789"
        },
        "query": "전세 계약 시 발생할 수 있는 법적 분쟁과 해결 방법은?",
        "query_type": "legal_advice",
        "query_complexity": "complex",
        "needs_search": True,
        "retrieved_docs": [],
        "semantic_results": [],
        "keyword_results": [],
        "extracted_keywords": ["전세", "계약", "법적", "분쟁"],
        "legal_field": "부동산법",
        "processing_steps": [],
        "errors": [],
        "common": {
            "metadata": {}
        }
    }


# 노드 클래스용 픽스처 (리팩토링된 구조)
@pytest.fixture
def mock_workflow_instance():
    """Mock 워크플로우 인스턴스 (노드 클래스용)"""
    workflow = MagicMock()
    workflow.logger = Mock()
    
    # ClassificationNodes용 메서드
    workflow.classify_query_and_complexity = Mock(return_value={"query": "test"})
    workflow.classification_parallel = Mock(return_value={"query": "test"})
    workflow.assess_urgency = Mock(return_value={"query": "test"})
    workflow.resolve_multi_turn = Mock(return_value={"query": "test"})
    workflow.route_expert = Mock(return_value={"query": "test"})
    workflow.direct_answer_node = Mock(return_value={"answer": "test"})
    
    # SearchNodes용 메서드
    workflow.expand_keywords = Mock(return_value={"extracted_keywords": ["test"]})
    workflow.prepare_search_query = Mock(return_value={"search_params": {}})
    workflow.execute_searches_parallel = Mock(return_value={"semantic_results": []})
    workflow.process_search_results_combined = Mock(return_value={"retrieved_docs": []})
    
    # DocumentNodes용 메서드
    workflow.analyze_document = Mock(return_value={"document_analysis": {}})
    workflow.prepare_documents_and_terms = Mock(return_value={"prepared_docs": []})
    
    # AnswerNodes용 메서드
    workflow.generate_and_validate_answer = Mock(return_value={"answer": "test answer"})
    workflow.generate_answer_stream = Mock(return_value={"answer": "streaming answer"})
    workflow.generate_answer_final = Mock(return_value={"answer": "final answer"})
    workflow.continue_answer_generation = Mock(return_value={"answer": "continued answer"})
    
    # AgenticNodes용 메서드
    workflow.agentic_decision_node = Mock(return_value={"agentic_decision": "test"})
    
    return workflow


@pytest.fixture
def classification_nodes(mock_workflow_instance):
    """ClassificationNodes 인스턴스"""
    from core.workflow.nodes.classification_nodes import ClassificationNodes
    return ClassificationNodes(
        workflow_instance=mock_workflow_instance,
        logger_instance=Mock()
    )


@pytest.fixture
def search_nodes(mock_workflow_instance):
    """SearchNodes 인스턴스"""
    from core.workflow.nodes.search_nodes import SearchNodes
    return SearchNodes(
        workflow_instance=mock_workflow_instance,
        logger_instance=Mock()
    )


@pytest.fixture
def document_nodes(mock_workflow_instance):
    """DocumentNodes 인스턴스"""
    from core.workflow.nodes.document_nodes import DocumentNodes
    return DocumentNodes(
        workflow_instance=mock_workflow_instance,
        logger_instance=Mock()
    )


@pytest.fixture
def answer_nodes(mock_workflow_instance):
    """AnswerNodes 인스턴스"""
    from core.workflow.nodes.answer_nodes import AnswerNodes
    return AnswerNodes(
        workflow_instance=mock_workflow_instance,
        logger_instance=Mock()
    )


@pytest.fixture
def agentic_nodes(mock_workflow_instance):
    """AgenticNodes 인스턴스"""
    from core.workflow.nodes.agentic_nodes import AgenticNodes
    return AgenticNodes(
        workflow_instance=mock_workflow_instance,
        logger_instance=Mock()
    )


@pytest.fixture
def ethical_rejection_node():
    """EthicalRejectionNode 인스턴스"""
    from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
    return EthicalRejectionNode(logger_instance=Mock())


@pytest.fixture
def node_registry():
    """NodeRegistry 인스턴스"""
    from core.workflow.registry.node_registry import NodeRegistry
    return NodeRegistry()


@pytest.fixture
def subgraph_registry():
    """SubgraphRegistry 인스턴스"""
    from core.workflow.registry.subgraph_registry import SubgraphRegistry
    return SubgraphRegistry()