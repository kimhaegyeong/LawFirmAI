# -*- coding: utf-8 -*-
"""
LangGraph 구조 리팩토링 단계별 수동 테스트
각 Phase별로 구현된 기능을 테스트
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

from lawfirm_langgraph.core.workflow.utils.ethical_checker import EthicalChecker
from lawfirm_langgraph.core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
from lawfirm_langgraph.core.workflow.nodes.classification_nodes import ClassificationNodes
from lawfirm_langgraph.core.workflow.nodes.search_nodes import SearchNodes
from lawfirm_langgraph.core.workflow.nodes.document_nodes import DocumentNodes
from lawfirm_langgraph.core.workflow.nodes.answer_nodes import AnswerNodes
from lawfirm_langgraph.core.workflow.nodes.agentic_nodes import AgenticNodes
from lawfirm_langgraph.core.workflow.registry.node_registry import NodeRegistry
from lawfirm_langgraph.core.workflow.registry.subgraph_registry import SubgraphRegistry
from lawfirm_langgraph.core.workflow.routes.classification_routes import ClassificationRoutes
from lawfirm_langgraph.core.workflow.routes.search_routes import SearchRoutes
from lawfirm_langgraph.core.workflow.routes.answer_routes import AnswerRoutes
from lawfirm_langgraph.core.workflow.routes.agentic_routes import AgenticRoutes
from lawfirm_langgraph.core.workflow.edges.classification_edges import ClassificationEdges
from lawfirm_langgraph.core.workflow.edges.search_edges import SearchEdges
from lawfirm_langgraph.core.workflow.edges.answer_edges import AnswerEdges
from lawfirm_langgraph.core.workflow.edges.agentic_edges import AgenticEdges
from lawfirm_langgraph.core.workflow.builders.modular_graph_builder import ModularGraphBuilder


def print_section(title: str):
    """섹션 제목 출력"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_phase0_ethical_check():
    """Phase 0: 윤리적 검사 기능 테스트"""
    print_section("Phase 0: 윤리적 검사 기능 테스트")
    
    try:
        # 1. EthicalChecker 초기화
        print("\n[1] EthicalChecker 초기화 테스트")
        checker = EthicalChecker()
        assert checker is not None, "EthicalChecker 초기화 실패"
        print("  ✓ EthicalChecker 초기화 성공")
        
        # 2. 불법 행위 키워드 감지
        print("\n[2] 불법 행위 키워드 감지 테스트")
        is_problematic, reason, severity = checker.check_query("어떻게 해킹하는지 알려주세요")
        assert is_problematic is True, "불법 행위 감지 실패"
        assert reason is not None, "거부 사유 없음"
        assert severity == "high", "심각도가 high가 아님"
        print(f"  ✓ 불법 행위 감지 성공: {reason[:50]}...")
        
        # 3. 법적 맥락에서 묻는 질문 허용
        print("\n[3] 법적 맥락 질문 허용 테스트")
        is_problematic, reason, severity = checker.check_query("해킹에 대한 법적 처벌은 무엇인가요?")
        assert is_problematic is False, "법적 맥락 질문이 거부됨"
        print("  ✓ 법적 맥락 질문 허용 성공")
        
        # 4. EthicalRejectionNode 테스트
        print("\n[4] EthicalRejectionNode 테스트")
        node = EthicalRejectionNode()
        state = {
            "query": "해킹 방법 알려주세요",
            "is_ethically_problematic": True,
            "ethical_rejection_reason": "불법 행위 조장"
        }
        result_state = node.generate_rejection_response(state, "불법 행위 조장")
        assert result_state.get("is_ethically_problematic") is True, "윤리적 문제 플래그 설정 실패"
        assert result_state.get("answer") is not None, "거부 메시지 생성 실패"
        print("  ✓ EthicalRejectionNode 동작 성공")
        
        print("\n✅ Phase 0 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 0 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_node_modularization():
    """Phase 1: 노드 모듈화 테스트"""
    print_section("Phase 1: 노드 모듈화 테스트")
    
    try:
        from unittest.mock import Mock
        
        mock_workflow = Mock()
        mock_logger = Mock()
        
        # 1. ClassificationNodes 테스트
        print("\n[1] ClassificationNodes 테스트")
        classification_nodes = ClassificationNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        assert hasattr(classification_nodes, 'classify_query_and_complexity'), "classify_query_and_complexity 없음"
        assert hasattr(classification_nodes, 'classification_parallel'), "classification_parallel 없음"
        assert hasattr(classification_nodes, 'assess_urgency'), "assess_urgency 없음"
        assert hasattr(classification_nodes, 'resolve_multi_turn'), "resolve_multi_turn 없음"
        assert hasattr(classification_nodes, 'route_expert'), "route_expert 없음"
        assert hasattr(classification_nodes, 'direct_answer'), "direct_answer 없음"
        print("  ✓ ClassificationNodes 메서드 확인 성공")
        
        # 2. SearchNodes 테스트
        print("\n[2] SearchNodes 테스트")
        search_nodes = SearchNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        assert hasattr(search_nodes, 'expand_keywords'), "expand_keywords 없음"
        assert hasattr(search_nodes, 'prepare_search_query'), "prepare_search_query 없음"
        assert hasattr(search_nodes, 'execute_searches_parallel'), "execute_searches_parallel 없음"
        assert hasattr(search_nodes, 'process_search_results_combined'), "process_search_results_combined 없음"
        print("  ✓ SearchNodes 메서드 확인 성공")
        
        # 3. DocumentNodes 테스트
        print("\n[3] DocumentNodes 테스트")
        document_nodes = DocumentNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        assert hasattr(document_nodes, 'analyze_document'), "analyze_document 없음"
        assert hasattr(document_nodes, 'prepare_documents_and_terms'), "prepare_documents_and_terms 없음"
        print("  ✓ DocumentNodes 메서드 확인 성공")
        
        # 4. AnswerNodes 테스트
        print("\n[4] AnswerNodes 테스트")
        answer_nodes = AnswerNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        assert hasattr(answer_nodes, 'generate_and_validate_answer'), "generate_and_validate_answer 없음"
        assert hasattr(answer_nodes, 'generate_answer_stream'), "generate_answer_stream 없음"
        assert hasattr(answer_nodes, 'generate_answer_final'), "generate_answer_final 없음"
        assert hasattr(answer_nodes, 'continue_answer_generation'), "continue_answer_generation 없음"
        print("  ✓ AnswerNodes 메서드 확인 성공")
        
        # 5. AgenticNodes 테스트
        print("\n[5] AgenticNodes 테스트")
        agentic_nodes = AgenticNodes(
            workflow_instance=mock_workflow,
            logger_instance=mock_logger
        )
        assert hasattr(agentic_nodes, 'agentic_decision_node'), "agentic_decision_node 없음"
        print("  ✓ AgenticNodes 메서드 확인 성공")
        
        print("\n✅ Phase 1 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_subgraph_expansion():
    """Phase 2: 서브그래프 확대 테스트"""
    print_section("Phase 2: 서브그래프 확대 테스트")
    
    try:
        # 서브그래프 클래스 존재 확인
        print("\n[1] 서브그래프 클래스 존재 확인")
        
        from lawfirm_langgraph.core.workflow.subgraphs.classification_subgraph import ClassificationSubgraph
        assert ClassificationSubgraph is not None, "ClassificationSubgraph 없음"
        print("  ✓ ClassificationSubgraph 존재 확인")
        
        from lawfirm_langgraph.core.workflow.subgraphs.search_subgraph import SearchSubgraph
        assert SearchSubgraph is not None, "SearchSubgraph 없음"
        print("  ✓ SearchSubgraph 존재 확인")
        
        from lawfirm_langgraph.core.workflow.subgraphs.document_preparation_subgraph import DocumentPreparationSubgraph
        assert DocumentPreparationSubgraph is not None, "DocumentPreparationSubgraph 없음"
        print("  ✓ DocumentPreparationSubgraph 존재 확인")
        
        from lawfirm_langgraph.core.workflow.subgraphs.answer_generation_subgraph import AnswerGenerationSubgraph
        assert AnswerGenerationSubgraph is not None, "AnswerGenerationSubgraph 없음"
        print("  ✓ AnswerGenerationSubgraph 존재 확인")
        
        print("\n✅ Phase 2 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_edge_modularization():
    """Phase 3: 엣지 모듈화 테스트"""
    print_section("Phase 3: 엣지 모듈화 테스트")
    
    try:
        # 엣지 클래스 존재 및 메서드 확인
        print("\n[1] 엣지 클래스 존재 및 메서드 확인")
        
        assert ClassificationEdges is not None, "ClassificationEdges 없음"
        assert hasattr(ClassificationEdges, 'add_classification_edges'), "add_classification_edges 없음"
        print("  ✓ ClassificationEdges 확인")
        
        assert SearchEdges is not None, "SearchEdges 없음"
        assert hasattr(SearchEdges, 'add_search_edges'), "add_search_edges 없음"
        print("  ✓ SearchEdges 확인")
        
        assert AnswerEdges is not None, "AnswerEdges 없음"
        assert hasattr(AnswerEdges, 'add_answer_generation_edges'), "add_answer_generation_edges 없음"
        print("  ✓ AnswerEdges 확인")
        
        assert AgenticEdges is not None, "AgenticEdges 없음"
        assert hasattr(AgenticEdges, 'add_agentic_edges'), "add_agentic_edges 없음"
        print("  ✓ AgenticEdges 확인")
        
        print("\n✅ Phase 3 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 3 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase4_registry_pattern():
    """Phase 4: 레지스트리 패턴 테스트"""
    print_section("Phase 4: 레지스트리 패턴 테스트")
    
    try:
        # 1. NodeRegistry 테스트
        print("\n[1] NodeRegistry 테스트")
        registry = NodeRegistry()
        
        def test_node(state):
            return state
        
        registry.register("test_node", test_node)
        assert registry.has_node("test_node"), "노드 등록 실패"
        assert registry.get_node("test_node") == test_node, "노드 조회 실패"
        
        all_nodes = registry.get_all_nodes()
        assert "test_node" in all_nodes, "모든 노드 조회 실패"
        
        registry.remove_node("test_node")
        assert not registry.has_node("test_node"), "노드 제거 실패"
        print("  ✓ NodeRegistry 기본 동작 성공")
        
        # 2. SubgraphRegistry 테스트
        print("\n[2] SubgraphRegistry 테스트")
        subgraph_registry = SubgraphRegistry()
        
        from unittest.mock import Mock
        mock_subgraph = Mock()
        subgraph_registry.register("test_subgraph", mock_subgraph)
        
        assert subgraph_registry.has_subgraph("test_subgraph"), "서브그래프 등록 실패"
        assert subgraph_registry.get_subgraph("test_subgraph") == mock_subgraph, "서브그래프 조회 실패"
        print("  ✓ SubgraphRegistry 기본 동작 성공")
        
        # 3. ModularGraphBuilder 테스트
        print("\n[3] ModularGraphBuilder 테스트")
        assert ModularGraphBuilder is not None, "ModularGraphBuilder 없음"
        assert hasattr(ModularGraphBuilder, 'build_graph'), "build_graph 없음"
        print("  ✓ ModularGraphBuilder 존재 확인")
        
        print("\n✅ Phase 4 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 4 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase5_routing_separation():
    """Phase 5: 라우팅 함수 분리 테스트"""
    print_section("Phase 5: 라우팅 함수 분리 테스트")
    
    try:
        # 1. ClassificationRoutes 테스트
        print("\n[1] ClassificationRoutes 테스트")
        routes = ClassificationRoutes()
        assert hasattr(routes, 'route_by_complexity'), "route_by_complexity 없음"
        assert hasattr(routes, 'route_by_complexity_with_agentic'), "route_by_complexity_with_agentic 없음"
        
        # 윤리적 거부 라우팅 테스트
        state = {
            "is_ethically_problematic": True,
            "query_complexity": "simple"
        }
        result = routes.route_by_complexity(state)
        assert result == "ethical_reject", f"윤리적 거부 라우팅 실패: {result}"
        print("  ✓ ClassificationRoutes 동작 성공")
        
        # 2. SearchRoutes 테스트
        print("\n[2] SearchRoutes 테스트")
        search_routes = SearchRoutes()
        assert hasattr(search_routes, 'should_analyze_document'), "should_analyze_document 없음"
        assert hasattr(search_routes, 'should_skip_search_adaptive'), "should_skip_search_adaptive 없음"
        assert hasattr(search_routes, 'should_expand_keywords_ai'), "should_expand_keywords_ai 없음"
        print("  ✓ SearchRoutes 메서드 확인 성공")
        
        # 3. AnswerRoutes 테스트
        print("\n[3] AnswerRoutes 테스트")
        from unittest.mock import Mock
        mock_retry_manager = Mock()
        answer_routes = AnswerRoutes(retry_manager=mock_retry_manager)
        assert hasattr(answer_routes, 'should_retry_validation'), "should_retry_validation 없음"
        assert hasattr(answer_routes, 'should_skip_final_node'), "should_skip_final_node 없음"
        print("  ✓ AnswerRoutes 메서드 확인 성공")
        
        # 4. AgenticRoutes 테스트
        print("\n[4] AgenticRoutes 테스트")
        agentic_routes = AgenticRoutes()
        assert hasattr(agentic_routes, 'route_after_agentic'), "route_after_agentic 없음"
        print("  ✓ AgenticRoutes 메서드 확인 성공")
        
        print("\n✅ Phase 5 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 5 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase6_task_node_clarification():
    """Phase 6: Task와 Node 역할 명확화 테스트"""
    print_section("Phase 6: Task와 Node 역할 명확화 테스트")
    
    try:
        # 문서 존재 확인
        print("\n[1] Task vs Node 문서 존재 확인")
        doc_path = Path(__file__).parent.parent.parent / "core" / "workflow" / "docs" / "task_vs_node.md"
        assert doc_path.exists(), f"Task vs Node 문서가 없습니다: {doc_path}"
        print(f"  ✓ 문서 존재 확인: {doc_path}")
        
        print("\n✅ Phase 6 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 6 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 80)
    print("  LangGraph 구조 리팩토링 단계별 테스트")
    print("=" * 80)
    
    results = {}
    
    # 각 Phase별 테스트 실행
    results['Phase 0'] = test_phase0_ethical_check()
    results['Phase 1'] = test_phase1_node_modularization()
    results['Phase 2'] = test_phase2_subgraph_expansion()
    results['Phase 3'] = test_phase3_edge_modularization()
    results['Phase 4'] = test_phase4_registry_pattern()
    results['Phase 5'] = test_phase5_routing_separation()
    results['Phase 6'] = test_phase6_task_node_clarification()
    
    # 결과 요약
    print_section("테스트 결과 요약")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for phase, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  {phase}: {status}")
    
    print(f"\n총 {total}개 Phase 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 모든 Phase 테스트 통과!")
        return 0
    else:
        print(f"\n⚠️  {total - passed}개 Phase 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())

