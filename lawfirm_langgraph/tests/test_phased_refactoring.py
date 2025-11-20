# -*- coding: utf-8 -*-
"""
LangGraph 구조 리팩토링 단계별 테스트
각 Phase별로 구현된 기능을 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.utils.ethical_checker import EthicalChecker
from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
from core.workflow.nodes.classification_nodes import ClassificationNodes
from core.workflow.nodes.search_nodes import SearchNodes
from core.workflow.nodes.document_nodes import DocumentNodes
from core.workflow.nodes.answer_nodes import AnswerNodes
from core.workflow.nodes.agentic_nodes import AgenticNodes
from core.workflow.registry.node_registry import NodeRegistry
from core.workflow.registry.subgraph_registry import SubgraphRegistry
from core.workflow.routes.classification_routes import ClassificationRoutes
from core.workflow.routes.search_routes import SearchRoutes
from core.workflow.routes.answer_routes import AnswerRoutes
from core.workflow.routes.agentic_routes import AgenticRoutes
from core.workflow.edges.classification_edges import ClassificationEdges
from core.workflow.edges.search_edges import SearchEdges
from core.workflow.edges.answer_edges import AnswerEdges
from core.workflow.edges.agentic_edges import AgenticEdges
from core.workflow.builders.modular_graph_builder import ModularGraphBuilder
try:
    from core.workflow.subgraphs.classification_subgraph import ClassificationSubgraph
    from core.workflow.subgraphs.search_subgraph import SearchSubgraph
    from core.workflow.subgraphs.document_preparation_subgraph import DocumentPreparationSubgraph
    from core.workflow.subgraphs.answer_generation_subgraph import AnswerGenerationSubgraph
    SUBGRAPHS_AVAILABLE = True
except ImportError:
    SUBGRAPHS_AVAILABLE = False


class TestPhase0_EthicalCheck:
    """Phase 0: 윤리적 검사 기능 테스트"""
    
    def test_ethical_checker_initialization(self):
        """윤리적 검사기 초기화 테스트"""
        checker = EthicalChecker()
        assert checker is not None
        assert hasattr(checker, 'check_query')
        assert hasattr(checker, 'check_with_llm')
    
    def test_ethical_checker_illegal_keywords(self):
        """불법 행위 키워드 감지 테스트"""
        checker = EthicalChecker()
        
        # 불법 행위 방법을 묻는 질문
        is_problematic, reason, severity = checker.check_query("어떻게 해킹하는지 알려주세요")
        assert is_problematic is True
        assert reason is not None
        assert severity == "high"
        
        # 불법 행위 도움 요청
        is_problematic, reason, severity = checker.check_query("탈세 방법을 도와주세요")
        assert is_problematic is True
    
    def test_ethical_checker_legal_context(self):
        """법적 맥락에서 묻는 질문 허용 테스트"""
        checker = EthicalChecker()
        
        # 법적 처벌에 대한 질문은 허용
        is_problematic, reason, severity = checker.check_query("해킹에 대한 법적 처벌은 무엇인가요?")
        assert is_problematic is False
        
        # 법률 규정에 대한 질문은 허용
        is_problematic, reason, severity = checker.check_query("절도죄의 법률 조항을 알려주세요")
        assert is_problematic is False
    
    def test_ethical_rejection_node(self):
        """윤리적 거부 노드 테스트"""
        node = EthicalRejectionNode()
        
        state: LegalWorkflowState = {
            "query": "해킹 방법 알려주세요",
            "is_ethically_problematic": True,
            "ethical_rejection_reason": "불법 행위 조장"
        }
        
        result_state = node.generate_rejection_response(state, "불법 행위 조장")
        
        assert result_state.get("is_ethically_problematic") is True
        assert result_state.get("answer") is not None
        assert "불법 행위" in result_state.get("answer", "")
        assert result_state.get("ethical_rejection_reason") is not None


class TestPhase1_NodeModularization:
    """Phase 1: 노드 모듈화 테스트"""
    
    @pytest.fixture
    def mock_workflow_instance(self):
        """Mock 워크플로우 인스턴스"""
        mock_workflow = Mock()
        mock_workflow.logger = Mock()
        return mock_workflow
    
    def test_classification_nodes_initialization(self, mock_workflow_instance):
        """분류 노드 초기화 테스트"""
        nodes = ClassificationNodes(
            workflow_instance=mock_workflow_instance,
            logger_instance=Mock()
        )
        assert nodes is not None
        assert hasattr(nodes, 'classify_query_and_complexity')
        assert hasattr(nodes, 'classification_parallel')
        assert hasattr(nodes, 'assess_urgency')
        assert hasattr(nodes, 'resolve_multi_turn')
        assert hasattr(nodes, 'route_expert')
        assert hasattr(nodes, 'direct_answer')
    
    def test_search_nodes_initialization(self, mock_workflow_instance):
        """검색 노드 초기화 테스트"""
        nodes = SearchNodes(
            workflow_instance=mock_workflow_instance,
            logger_instance=Mock()
        )
        assert nodes is not None
        assert hasattr(nodes, 'expand_keywords')
        assert hasattr(nodes, 'prepare_search_query')
        assert hasattr(nodes, 'execute_searches_parallel')
        assert hasattr(nodes, 'process_search_results_combined')
    
    def test_document_nodes_initialization(self, mock_workflow_instance):
        """문서 노드 초기화 테스트"""
        nodes = DocumentNodes(
            workflow_instance=mock_workflow_instance,
            logger_instance=Mock()
        )
        assert nodes is not None
        assert hasattr(nodes, 'analyze_document')
        assert hasattr(nodes, 'prepare_documents_and_terms')
    
    def test_answer_nodes_initialization(self, mock_workflow_instance):
        """답변 노드 초기화 테스트"""
        nodes = AnswerNodes(
            workflow_instance=mock_workflow_instance,
            logger_instance=Mock()
        )
        assert nodes is not None
        assert hasattr(nodes, 'generate_and_validate_answer')
        assert hasattr(nodes, 'generate_answer_stream')
        assert hasattr(nodes, 'generate_answer_final')
        assert hasattr(nodes, 'continue_answer_generation')
    
    def test_agentic_nodes_initialization(self, mock_workflow_instance):
        """Agentic 노드 초기화 테스트"""
        nodes = AgenticNodes(
            workflow_instance=mock_workflow_instance,
            logger_instance=Mock()
        )
        assert nodes is not None
        assert hasattr(nodes, 'agentic_decision_node')


class TestPhase2_SubgraphExpansion:
    """Phase 2: 서브그래프 확대 테스트"""
    
    @pytest.mark.skipif(not SUBGRAPHS_AVAILABLE, reason="서브그래프 모듈을 import할 수 없습니다")
    def test_classification_subgraph_exists(self):
        """분류 서브그래프 존재 확인"""
        # 서브그래프 클래스가 존재하는지 확인
        assert ClassificationSubgraph is not None
    
    @pytest.mark.skipif(not SUBGRAPHS_AVAILABLE, reason="서브그래프 모듈을 import할 수 없습니다")
    def test_search_subgraph_exists(self):
        """검색 서브그래프 존재 확인"""
        assert SearchSubgraph is not None
    
    @pytest.mark.skipif(not SUBGRAPHS_AVAILABLE, reason="서브그래프 모듈을 import할 수 없습니다")
    def test_document_preparation_subgraph_exists(self):
        """문서 준비 서브그래프 존재 확인"""
        assert DocumentPreparationSubgraph is not None
    
    @pytest.mark.skipif(not SUBGRAPHS_AVAILABLE, reason="서브그래프 모듈을 import할 수 없습니다")
    def test_answer_generation_subgraph_exists(self):
        """답변 생성 서브그래프 존재 확인"""
        assert AnswerGenerationSubgraph is not None


class TestPhase3_EdgeModularization:
    """Phase 3: 엣지 모듈화 테스트"""
    
    def test_classification_edges_exists(self):
        """분류 엣지 클래스 존재 확인"""
        assert ClassificationEdges is not None
        assert hasattr(ClassificationEdges, 'add_classification_edges')
    
    def test_search_edges_exists(self):
        """검색 엣지 클래스 존재 확인"""
        assert SearchEdges is not None
        assert hasattr(SearchEdges, 'add_search_edges')
    
    def test_answer_edges_exists(self):
        """답변 엣지 클래스 존재 확인"""
        assert AnswerEdges is not None
        assert hasattr(AnswerEdges, 'add_answer_generation_edges')
    
    def test_agentic_edges_exists(self):
        """Agentic 엣지 클래스 존재 확인"""
        assert AgenticEdges is not None
        assert hasattr(AgenticEdges, 'add_agentic_edges')


class TestPhase4_RegistryPattern:
    """Phase 4: 레지스트리 패턴 테스트"""
    
    def test_node_registry_operations(self):
        """노드 레지스트리 기본 동작 테스트"""
        registry = NodeRegistry()
        
        def test_node(state):
            return state
        
        # 노드 등록
        registry.register("test_node", test_node)
        assert registry.has_node("test_node")
        assert registry.get_node("test_node") == test_node
        
        # 모든 노드 조회
        all_nodes = registry.get_all_nodes()
        assert "test_node" in all_nodes
        
        # 노드 제거
        registry.remove_node("test_node")
        assert not registry.has_node("test_node")
    
    def test_subgraph_registry_operations(self):
        """서브그래프 레지스트리 기본 동작 테스트"""
        registry = SubgraphRegistry()
        
        # Mock subgraph
        mock_subgraph = Mock()
        registry.register("test_subgraph", mock_subgraph)
        
        assert registry.has_subgraph("test_subgraph")
        assert registry.get_subgraph("test_subgraph") == mock_subgraph
        
        # 모든 서브그래프 조회
        all_subgraphs = registry.get_all_subgraphs()
        assert "test_subgraph" in all_subgraphs
    
    def test_modular_graph_builder_exists(self):
        """모듈화 그래프 빌더 존재 확인"""
        assert ModularGraphBuilder is not None
        assert hasattr(ModularGraphBuilder, 'build_graph')


class TestPhase5_RoutingSeparation:
    """Phase 5: 라우팅 함수 분리 테스트"""
    
    def test_classification_routes_exists(self):
        """분류 라우팅 클래스 존재 확인"""
        routes = ClassificationRoutes()
        assert routes is not None
        assert hasattr(routes, 'route_by_complexity')
        assert hasattr(routes, 'route_by_complexity_with_agentic')
    
    def test_search_routes_exists(self):
        """검색 라우팅 클래스 존재 확인"""
        routes = SearchRoutes()
        assert routes is not None
        assert hasattr(routes, 'should_analyze_document')
        assert hasattr(routes, 'should_skip_search_adaptive')
        assert hasattr(routes, 'should_expand_keywords_ai')
    
    def test_answer_routes_exists(self):
        """답변 라우팅 클래스 존재 확인"""
        routes = AnswerRoutes()
        assert routes is not None
        assert hasattr(routes, 'should_retry_validation')
        assert hasattr(routes, 'should_retry_generation')
        assert hasattr(routes, 'should_skip_final_node')
    
    def test_agentic_routes_exists(self):
        """Agentic 라우팅 클래스 존재 확인"""
        routes = AgenticRoutes()
        assert routes is not None
        assert hasattr(routes, 'route_after_agentic')
    
    def test_classification_routes_ethical_reject(self):
        """윤리적 거부 라우팅 테스트"""
        routes = ClassificationRoutes()
        
        state = {
            "is_ethically_problematic": True,
            "query_complexity": "simple"
        }
        result = routes.route_by_complexity(state)
        assert result == "ethical_reject"
    
    def test_classification_routes_complexity_routing(self):
        """복잡도 기반 라우팅 테스트"""
        routes = ClassificationRoutes()
        
        # Simple 쿼리
        state = {
            "is_ethically_problematic": False,
            "query_complexity": "simple"
        }
        result = routes.route_by_complexity(state)
        assert result == "simple"
        
        # Moderate 쿼리
        state = {
            "is_ethically_problematic": False,
            "query_complexity": "moderate"
        }
        result = routes.route_by_complexity(state)
        assert result == "moderate"
        
        # Complex 쿼리
        state = {
            "is_ethically_problematic": False,
            "query_complexity": "complex"
        }
        result = routes.route_by_complexity(state)
        assert result == "complex"


class TestPhase6_TaskNodeClarification:
    """Phase 6: Task와 Node 역할 명확화 테스트"""
    
    def test_task_vs_node_documentation_exists(self):
        """Task vs Node 문서 존재 확인"""
        from pathlib import Path
        doc_path = Path("lawfirm_langgraph/core/workflow/docs/task_vs_node.md")
        assert doc_path.exists(), "Task vs Node 문서가 존재해야 합니다"


class TestIntegration_AllPhases:
    """모든 Phase 통합 테스트"""
    
    def test_workflow_graph_builds_with_all_phases(self):
        """모든 Phase가 적용된 워크플로우 그래프 구축 테스트"""
        # 이 테스트는 실제 워크플로우 인스턴스가 필요하므로
        # 통합 테스트에서 실행
        pass
    
    def test_ethical_check_integration(self):
        """윤리적 검사 통합 테스트"""
        checker = EthicalChecker()
        
        # 정상적인 질문
        is_problematic, reason, severity = checker.check_query("계약 해지 사유에 대해 알려주세요")
        assert is_problematic is False
        
        # 윤리적으로 문제되는 질문
        is_problematic, reason, severity = checker.check_query("어떻게 해킹하는지 알려주세요")
        assert is_problematic is True
    
    def test_node_registry_with_real_nodes(self):
        """실제 노드 클래스를 사용한 레지스트리 테스트"""
        registry = NodeRegistry()
        
        # Mock workflow instance
        mock_workflow = Mock()
        mock_logger = Mock()
        
        # ClassificationNodes 등록
        classification_nodes = ClassificationNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        
        registry.register(
            "classify_query_and_complexity",
            classification_nodes.classify_query_and_complexity
        )
        
        assert registry.has_node("classify_query_and_complexity")
        node_func = registry.get_node("classify_query_and_complexity")
        assert callable(node_func)

