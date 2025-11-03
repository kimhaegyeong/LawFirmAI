# -*- coding: utf-8 -*-
"""빠른 LangGraph 테스트"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("LangGraph 빠른 테스트")
    print("=" * 60)
    print()

    # 1. Python 버전
    print(f"Python 버전: {sys.version}")
    print()

    # 2. Import 테스트
    print("Import 테스트...")
    try:
        import langgraph
        print(f"  ✓ LangGraph: {getattr(langgraph, '__version__', 'N/A')}")
    except Exception as e:
        print(f"  ✗ LangGraph: {e}")
        return False

    try:
        from config.langgraph_config import LangGraphConfig
        print("  ✓ Config 모듈")
    except Exception as e:
        print(f"  ✗ Config 모듈: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from graph import graph, app, create_graph, create_app
        print("  ✓ Graph export 모듈")
    except Exception as e:
        print(f"  ✗ Graph export 모듈: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # 3. 설정 로딩
    print("설정 로딩 테스트...")
    try:
        config = LangGraphConfig.from_env()
        print(f"  ✓ LangGraph 활성화: {config.langgraph_enabled}")
        print(f"  ✓ LLM 제공자: {config.llm_provider}")
        print(f"  ✓ Google 모델: {config.google_model}")
    except Exception as e:
        print(f"  ✗ 설정 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # 4. Graph 생성
    print("Graph 생성 테스트...")
    try:
        g = create_graph()
        print(f"  ✓ Graph 생성: {type(g).__name__}")
    except Exception as e:
        print(f"  ✗ Graph 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # 5. App 생성
    print("App 생성 테스트...")
    try:
        a = create_app()
        print(f"  ✓ App 생성: {type(a).__name__}")
    except Exception as e:
        print(f"  ✗ App 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("✓ 모든 기본 테스트 통과!")
    print("=" * 60)
    print()
    print("다음 단계:")
    print("  1. python test_workflow.py  # 종합 테스트 실행")
    print("  2. langgraph dev            # LangGraph Studio 실행")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
