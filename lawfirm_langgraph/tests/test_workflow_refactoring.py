# -*- coding: utf-8 -*-
"""
워크플로우 리팩토링 테스트
"""

import pytest
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


def test_node_registry():
    """노드 레지스트리 테스트"""
    registry = NodeRegistry()
    
    def test_node(state):
        return state
    
    registry.register("test_node", test_node)
    assert registry.has_node("test_node")
    assert registry.get_node("test_node") == test_node
    
    all_nodes = registry.get_all_nodes()
    assert "test_node" in all_nodes
    
    registry.remove_node("test_node")
    assert not registry.has_node("test_node")


def test_subgraph_registry():
    """서브그래프 레지스트리 테스트"""
    registry = SubgraphRegistry()
    
    # Mock subgraph
    class MockSubgraph:
        pass
    
    mock_subgraph = MockSubgraph()
    registry.register("test_subgraph", mock_subgraph)
    
    assert registry.has_subgraph("test_subgraph")
    assert registry.get_subgraph("test_subgraph") == mock_subgraph


def test_classification_routes():
    """분류 라우팅 테스트"""
    routes = ClassificationRoutes()
    
    # 윤리적 문제 감지
    state = {
        "is_ethically_problematic": True,
        "query_complexity": "simple"
    }
    result = routes.route_by_complexity(state)
    assert result == "ethical_reject"
    
    # 정상적인 복잡도 라우팅
    state = {
        "is_ethically_problematic": False,
        "query_complexity": "simple"
    }
    result = routes.route_by_complexity(state)
    assert result == "simple"
    
    state = {
        "is_ethically_problematic": False,
        "query_complexity": "moderate"
    }
    result = routes.route_by_complexity(state)
    assert result == "moderate"


def test_search_routes():
    """검색 라우팅 테스트"""
    routes = SearchRoutes()
    
    # 문서 분석 필요
    state = {
        "uploaded_document": {"type": "contract"}
    }
    result = routes.should_analyze_document(state)
    assert result == "analyze"
    
    # 문서 분석 불필요
    state = {
        "uploaded_document": None
    }
    result = routes.should_analyze_document(state)
    assert result == "skip"
    
    # 검색 스킵 (캐시 히트)
    state = {
        "search_cache_hit": True,
        "needs_search": True
    }
    result = routes.should_skip_search_adaptive(state)
    assert result == "skip"


def test_agentic_routes():
    """Agentic 라우팅 테스트"""
    routes = AgenticRoutes()
    
    # 검색 결과 있음
    state = {
        "search": {
            "results": [{"id": 1, "content": "test"}]
        }
    }
    result = routes.route_after_agentic(state)
    assert result == "has_results"
    
    # 검색 결과 없음
    state = {
        "search": {
            "results": []
        },
        "retrieved_docs": []
    }
    result = routes.route_after_agentic(state)
    assert result == "no_results"

