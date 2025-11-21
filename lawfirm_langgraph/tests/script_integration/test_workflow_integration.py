# -*- coding: utf-8 -*-
"""
워크플로우 리팩토링 통합 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
script_dir = Path(__file__).parent
integration_dir = script_dir.parent
tests_dir = integration_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent
sys.path.insert(0, str(project_root))

from core.workflow.registry.node_registry import NodeRegistry
from core.workflow.registry.subgraph_registry import SubgraphRegistry
from core.workflow.routes.classification_routes import ClassificationRoutes
from core.workflow.routes.search_routes import SearchRoutes
from core.workflow.routes.answer_routes import AnswerRoutes
from core.workflow.routes.agentic_routes import AgenticRoutes


def test_node_registry():
    """노드 레지스트리 테스트"""
    print("=" * 60)
    print("노드 레지스트리 테스트")
    print("=" * 60)
    
    registry = NodeRegistry()
    
    def test_node(state):
        return state
    
    registry.register("test_node", test_node)
    assert registry.has_node("test_node"), "노드가 등록되어야 합니다"
    assert registry.get_node("test_node") == test_node, "등록된 노드를 가져올 수 있어야 합니다"
    
    all_nodes = registry.get_all_nodes()
    assert "test_node" in all_nodes, "모든 노드 목록에 포함되어야 합니다"
    
    registry.remove_node("test_node")
    assert not registry.has_node("test_node"), "노드가 제거되어야 합니다"
    
    print("✅ 노드 레지스트리 테스트 통과")


def test_subgraph_registry():
    """서브그래프 레지스트리 테스트"""
    print("\n" + "=" * 60)
    print("서브그래프 레지스트리 테스트")
    print("=" * 60)
    
    registry = SubgraphRegistry()
    
    # Mock subgraph
    class MockSubgraph:
        pass
    
    mock_subgraph = MockSubgraph()
    registry.register("test_subgraph", mock_subgraph)
    
    assert registry.has_subgraph("test_subgraph"), "서브그래프가 등록되어야 합니다"
    assert registry.get_subgraph("test_subgraph") == mock_subgraph, "등록된 서브그래프를 가져올 수 있어야 합니다"
    
    print("✅ 서브그래프 레지스트리 테스트 통과")


def test_classification_routes():
    """분류 라우팅 테스트"""
    print("\n" + "=" * 60)
    print("분류 라우팅 테스트")
    print("=" * 60)
    
    routes = ClassificationRoutes()
    
    # 윤리적 문제 감지
    state = {
        "is_ethically_problematic": True,
        "query_complexity": "simple"
    }
    result = routes.route_by_complexity(state)
    assert result == "ethical_reject", "윤리적 문제는 ethical_reject로 라우팅되어야 합니다"
    print("✅ 윤리적 문제 감지 라우팅 테스트 통과")
    
    # 정상적인 복잡도 라우팅
    state = {
        "is_ethically_problematic": False,
        "query_complexity": "simple"
    }
    result = routes.route_by_complexity(state)
    assert result == "simple", "simple 복잡도는 simple로 라우팅되어야 합니다"
    print("✅ Simple 복잡도 라우팅 테스트 통과")
    
    state = {
        "is_ethically_problematic": False,
        "query_complexity": "moderate"
    }
    result = routes.route_by_complexity(state)
    assert result == "moderate", "moderate 복잡도는 moderate로 라우팅되어야 합니다"
    print("✅ Moderate 복잡도 라우팅 테스트 통과")


def test_search_routes():
    """검색 라우팅 테스트"""
    print("\n" + "=" * 60)
    print("검색 라우팅 테스트")
    print("=" * 60)
    
    routes = SearchRoutes()
    
    # 문서 분석 필요
    state = {
        "uploaded_document": {"type": "contract"}
    }
    result = routes.should_analyze_document(state)
    assert result == "analyze", "문서가 있으면 analyze로 라우팅되어야 합니다"
    print("✅ 문서 분석 필요 라우팅 테스트 통과")
    
    # 문서 분석 불필요
    state = {
        "uploaded_document": None
    }
    result = routes.should_analyze_document(state)
    assert result == "skip", "문서가 없으면 skip으로 라우팅되어야 합니다"
    print("✅ 문서 분석 불필요 라우팅 테스트 통과")
    
    # 검색 스킵 (캐시 히트)
    state = {
        "search_cache_hit": True,
        "needs_search": True
    }
    result = routes.should_skip_search_adaptive(state)
    assert result == "skip", "캐시 히트 시 skip으로 라우팅되어야 합니다"
    print("✅ 검색 스킵 라우팅 테스트 통과")


def test_agentic_routes():
    """Agentic 라우팅 테스트"""
    print("\n" + "=" * 60)
    print("Agentic 라우팅 테스트")
    print("=" * 60)
    
    routes = AgenticRoutes()
    
    # 검색 결과 있음
    state = {
        "search": {
            "results": [{"id": 1, "content": "test"}]
        }
    }
    result = routes.route_after_agentic(state)
    assert result == "has_results", "검색 결과가 있으면 has_results로 라우팅되어야 합니다"
    print("✅ 검색 결과 있음 라우팅 테스트 통과")
    
    # 검색 결과 없음
    state = {
        "search": {
            "results": []
        },
        "retrieved_docs": []
    }
    result = routes.route_after_agentic(state)
    assert result == "no_results", "검색 결과가 없으면 no_results로 라우팅되어야 합니다"
    print("✅ 검색 결과 없음 라우팅 테스트 통과")


if __name__ == "__main__":
    try:
        test_node_registry()
        test_subgraph_registry()
        test_classification_routes()
        test_search_routes()
        test_agentic_routes()
        print("\n" + "=" * 60)
        print("✅ 모든 통합 테스트 통과!")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

