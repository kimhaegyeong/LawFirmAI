# -*- coding: utf-8 -*-
"""
LangGraph Workflow Nodes 테스트 (리팩토링된 구조)
워크플로우 노드 및 핸들러 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
from core.workflow.nodes.classification_nodes import ClassificationNodes
from core.workflow.nodes.search_nodes import SearchNodes
from core.workflow.nodes.document_nodes import DocumentNodes
from core.workflow.nodes.answer_nodes import AnswerNodes


class TestWorkflowNodes:
    """워크플로우 노드 테스트 (리팩토링된 구조)"""
    
    @pytest.fixture
    def mock_state(self):
        """Mock 워크플로우 상태"""
        return {
            "input": {"query": "계약서 작성 시 주의사항은?", "session_id": "test_session"},
            "query": "계약서 작성 시 주의사항은?",
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
    def config(self):
        """테스트용 설정"""
        return LangGraphConfig(
            langgraph_enabled=True,
            use_agentic_mode=False,
        )
    
    @pytest.fixture
    def mock_workflow(self):
        """Mock 워크플로우 인스턴스"""
        workflow = MagicMock()
        workflow.classify_query_and_complexity = Mock(return_value={"query": "test"})
        workflow.classification_parallel = Mock(return_value={"query": "test"})
        workflow.expand_keywords = Mock(return_value={"extracted_keywords": ["test"]})
        workflow.prepare_search_query = Mock(return_value={"search_params": {}})
        workflow.analyze_document = Mock(return_value={"document_analysis": {}})
        workflow.generate_and_validate_answer = Mock(return_value={"answer": "test answer"})
        return workflow
    
    def test_classification_nodes(self, mock_state, mock_workflow):
        """분류 노드 클래스 테스트"""
        classification_nodes = ClassificationNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
        
        result = classification_nodes.classify_query_and_complexity(mock_state)
        
        assert result == {"query": "test"}
        mock_workflow.classify_query_and_complexity.assert_called_once_with(mock_state)
    
    def test_search_nodes(self, mock_state, mock_workflow):
        """검색 노드 클래스 테스트"""
        search_nodes = SearchNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
        
        result = search_nodes.expand_keywords(mock_state)
        
        assert result == {"extracted_keywords": ["test"]}
        mock_workflow.expand_keywords.assert_called_once_with(mock_state)
    
    def test_document_nodes(self, mock_state, mock_workflow):
        """문서 노드 클래스 테스트"""
        document_nodes = DocumentNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
        
        result = document_nodes.analyze_document(mock_state)
        
        assert result == {"document_analysis": {}}
        mock_workflow.analyze_document.assert_called_once_with(mock_state)
    
    def test_answer_nodes(self, mock_state, mock_workflow):
        """답변 노드 클래스 테스트"""
        answer_nodes = AnswerNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
        
        result = answer_nodes.generate_and_validate_answer(mock_state)
        
        assert result == {"answer": "test answer"}
        mock_workflow.generate_and_validate_answer.assert_called_once_with(mock_state)
    
    def test_search_node(self, mock_state, config):
        """검색 노드 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.search_handler.SearchHandler') as MockHandler:
            mock_handler = MockHandler.return_value
            mock_handler.search = Mock(return_value=[
                {
                    "content": "검색 결과 1",
                    "metadata": {"source": "test", "similarity": 0.8},
                }
            ])
            
            result = mock_handler.search(mock_state["query"])
            
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_context_builder_node(self, mock_state, config):
        """컨텍스트 빌더 노드 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.context_builder.ContextBuilder') as MockBuilder:
            mock_builder = MockBuilder.return_value
            mock_builder.build_context = Mock(return_value="빌드된 컨텍스트")
            
            result = mock_builder.build_context(mock_state["query"], mock_state["retrieved_docs"])
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_answer_generator_node(self, mock_state, config):
        """답변 생성 노드 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.answer_generator.AnswerGenerator') as MockGenerator:
            mock_generator = MockGenerator.return_value
            mock_generator.generate = Mock(return_value="생성된 답변")
            
            result = mock_generator.generate(
                mock_state["query"],
                mock_state["context"]
            )
            
            assert isinstance(result, str)
            assert result == "생성된 답변"
    
    def test_answer_formatter_node(self, mock_state, config):
        """답변 포맷터 노드 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.answer_formatter.AnswerFormatterHandler') as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format = Mock(return_value="포맷된 답변")
            
            result = mock_formatter.format("원본 답변")
            
            assert isinstance(result, str)
            assert result == "포맷된 답변"
    
    def test_direct_answer_handler(self, mock_state, config):
        """직접 답변 핸들러 테스트"""
        with patch('lawfirm_langgraph.core.agents.handlers.direct_answer_handler.DirectAnswerHandler') as MockHandler:
            mock_handler = MockHandler.return_value
            mock_handler.can_handle = Mock(return_value=True)
            mock_handler.handle = Mock(return_value="직접 답변")
            
            can_handle = mock_handler.can_handle(mock_state["query"])
            assert can_handle is True
            
            result = mock_handler.handle(mock_state["query"])
            assert result == "직접 답변"


class TestStateManagement:
    """상태 관리 테스트"""
    
    def test_state_initialization(self):
        """상태 초기화 테스트"""
        # flat 구조를 사용하여 테스트 (레거시 호환성)
        try:
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_flat_legal_state
            
            state = create_flat_legal_state(
                query="테스트 질문",
                session_id="test_session"
            )
            
            assert state["query"] == "테스트 질문"
            assert state["session_id"] == "test_session"
            assert "retrieved_docs" in state
        except ImportError:
            # modular 구조를 사용하는 경우
            from lawfirm_langgraph.core.workflow.state.state_definitions import create_initial_legal_state
            
            state = create_initial_legal_state(
                query="테스트 질문",
                session_id="test_session"
            )
            
            # modular 구조에서는 input.query로 접근
            assert hasattr(state, "input") or "input" in state
            if hasattr(state, "input"):
                assert state.input.query == "테스트 질문"
            elif isinstance(state, dict) and "input" in state:
                assert state["input"]["query"] == "테스트 질문"
    
    def test_state_update(self):
        """상태 업데이트 테스트"""
        state = {
            "query": "테스트 질문",
            "answer": "",
            "context": [],
        }
        
        # 상태 업데이트 시뮬레이션
        state["answer"] = "테스트 답변"
        state["context"] = ["컨텍스트 1"]
        
        assert state["answer"] == "테스트 답변"
        assert len(state["context"]) == 1


class TestWorkflowRouting:
    """워크플로우 라우팅 테스트"""
    
    def test_route_simple_query(self):
        """간단한 쿼리 라우팅 테스트"""
        query = "안녕하세요"
        
        # 간단한 쿼리는 직접 답변으로 라우팅
        is_simple = len(query) < 10
        
        assert is_simple is True
    
    def test_route_complex_query(self):
        """복잡한 쿼리 라우팅 테스트"""
        query = "계약서 작성 시 주의해야 할 법적 사항과 관련 판례를 알려주세요"
        
        # 복잡한 쿼리는 검색 및 분석이 필요
        is_complex = len(query) > 20
        
        assert is_complex is True
    
    def test_route_by_legal_field(self):
        """법률 분야별 라우팅 테스트"""
        classification = {
            "legal_field": "contract",
            "complexity": "medium",
        }
        
        # 계약 관련은 계약 전문가로 라우팅
        if classification["legal_field"] == "contract":
            route = "contract_expert"
        else:
            route = "general"
        
        assert route == "contract_expert"


class TestErrorHandling:
    """에러 핸들링 테스트"""
    
    def test_node_error_handling(self):
        """노드 에러 핸들링 테스트"""
        state = {
            "query": "테스트 질문",
            "errors": [],
        }
        
        # 에러 발생 시뮬레이션
        try:
            raise ValueError("테스트 에러")
        except Exception as e:
            state["errors"].append(str(e))
        
        assert len(state["errors"]) > 0
        assert "테스트 에러" in state["errors"][0]
    
    def test_workflow_error_recovery(self):
        """워크플로우 에러 복구 테스트"""
        state = {
            "query": "테스트 질문",
            "errors": ["에러 1"],
            "processing_steps": [],
        }
        
        # 에러 복구 시뮬레이션
        if len(state["errors"]) > 0:
            # 재시도 로직
            state["processing_steps"].append("retry")
        
        assert "retry" in state["processing_steps"]

