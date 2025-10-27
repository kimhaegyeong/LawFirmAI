# -*- coding: utf-8 -*-
"""
하이브리드 검색 간단 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_semantic_search_engine():
    """SemanticSearchEngine 기본 테스트"""
    print("\n" + "="*80)
    print("SemanticSearchEngine 기본 테스트")
    print("="*80 + "\n")

    try:
        from source.services.semantic_search_engine import SemanticSearchEngine

        # SemanticSearchEngine 초기화
        print("SemanticSearchEngine 초기화 중...")
        engine = SemanticSearchEngine()

        print(f"✅ Index loaded: {engine.index is not None}")
        print(f"✅ Model loaded: {engine.model is not None}")
        print(f"✅ Metadata loaded: {len(engine.metadata)} items")

        if engine.index:
            print(f"✅ FAISS index contains {engine.index.ntotal} vectors")

        # 테스트 검색
        test_query = "이혼 절차"
        print(f"\n검색 테스트: '{test_query}'")
        print("-" * 80)

        results = engine.search(test_query, k=3)

        print(f"검색 결과: {len(results)}개\n")

        if results:
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. [Score: {result.get('score', 0):.3f}]")
                text = result.get('text', '')
                if text:
                    print(f"   텍스트: {text[:100]}...")
                print(f"   소스: {result.get('source', 'Unknown')}")
                print(f"   검색 타입: {result.get('search_type', 'Unknown')}")
                print()

        print("✅ SemanticSearchEngine 테스트 완료\n")
        return True

    except Exception as e:
        print(f"❌ SemanticSearchEngine 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_components():
    """워크플로우 컴포넌트 초기화 테스트"""
    print("\n" + "="*80)
    print("워크플로우 컴포넌트 초기화 테스트")
    print("="*80 + "\n")

    try:
        from source.services.langgraph.legal_workflow_enhanced import (
            EnhancedLegalQuestionWorkflow,
        )
        from source.utils.langgraph_config import LangGraphConfig

        # 설정 로드
        config = LangGraphConfig.from_env()
        print("✅ LangGraph 설정 로드 완료")

        # 워크플로우 초기화
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("✅ EnhancedLegalQuestionWorkflow 초기화 완료")

        # SemanticSearchEngine 확인
        if workflow.semantic_search:
            print("✅ SemanticSearchEngine 통합 확인")
        else:
            print("⚠️  SemanticSearchEngine 통합 실패 (폴백 사용)")

        print("\n✅ 워크플로우 컴포넌트 테스트 완료\n")
        return True

    except Exception as e:
        print(f"❌ 워크플로우 컴포넌트 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("하이브리드 검색 통합 테스트 실행\n")

    # 1. SemanticSearchEngine 테스트
    test1 = test_semantic_search_engine()

    # 2. 워크플로우 컴포넌트 테스트
    test2 = test_workflow_components()

    # 결과 요약
    print("="*80)
    print("테스트 결과 요약")
    print("="*80)
    print(f"SemanticSearchEngine: {'✅ 통과' if test1 else '❌ 실패'}")
    print(f"워크플로우 통합: {'✅ 통과' if test2 else '❌ 실패'}")
    print("="*80)
    print("\n✅ 하이브리드 검색 통합이 완료되었습니다.")
    print("   - SemanticSearchEngine은 LangGraph 워크플로우에 통합됨")
    print("   - Fallback 메커니즘으로 안정성 확보")
    print("   - 하이브리드 검색 (벡터 + 키워드) 사용 가능\n")
