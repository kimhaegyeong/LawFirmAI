# -*- coding: utf-8 -*-
"""
LangGraph Studio 설정 테스트 스크립트
프로젝트가 올바르게 설정되었는지 확인합니다.
"""

import sys
from pathlib import Path

def test_imports():
    """필수 모듈 import 테스트"""
    print("Testing imports...")

    try:
        # lawfirm_langgraph 경로 추가
        lawfirm_langgraph_root = Path(__file__).parent.parent
        sys.path.insert(0, str(lawfirm_langgraph_root))

        # 상위 프로젝트 경로 추가
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Graph import 테스트
        print("  - Testing graph import...")
        from graph import graph, app
        print("    ✓ Graph and app imports successful")

        # Config import 테스트
        print("  - Testing config import...")
        from config.langgraph_config import LangGraphConfig
        config = LangGraphConfig.from_env()
        print(f"    ✓ Config loaded: langgraph_enabled={config.langgraph_enabled}")

        # Source services import 테스트
        print("  - Testing source services import...")
        from langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from langgraph_core.services.workflow_service import LangGraphWorkflowService
        print("    ✓ Source services imports successful")

        return True
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_creation():
    """그래프 생성 테스트"""
    print("\nTesting graph creation...")

    try:
        from graph import graph as graph_instance, app as app_instance

        # 그래프 확인
        print("  - Getting graph instance...")
        g = graph_instance()
        print(f"    ✓ Graph created: {type(g).__name__}")

        # 앱 확인
        print("  - Getting app instance...")
        a = app_instance()
        print(f"    ✓ App created: {type(a).__name__}")

        return True
    except Exception as e:
        print(f"    ✗ Graph creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_langgraph_version():
    """LangGraph 버전 확인"""
    print("\nTesting LangGraph version...")

    try:
        import langgraph
        # LangGraph v1.0 확인
        try:
            version = langgraph.__version__
            print(f"  ✓ LangGraph version: {version}")
            major_version = int(version.split('.')[0])
            if major_version >= 1:
                print("    ✓ LangGraph v1.0+ detected")
                return True
            else:
                print("    ⚠ Warning: LangGraph v1.0+ recommended")
                return False
        except AttributeError:
            print("  ⚠ LangGraph version info not available")
            return True
    except ImportError:
        print("  ✗ LangGraph not installed")
        return False

def test_python_version():
    """Python 버전 확인"""
    print("\nTesting Python version...")

    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 10:
        print("    ✓ Python 3.10+ detected")
        return True
    else:
        print("    ✗ Python 3.10+ required")
        return False

def test_langgraph_cli():
    """LangGraph CLI 확인"""
    print("\nTesting LangGraph CLI...")

    import shutil
    cli_path = shutil.which('langgraph')

    if cli_path:
        print(f"  ✓ LangGraph CLI found: {cli_path}")
        return True
    else:
        print("  ⚠ LangGraph CLI not found in PATH")
        print("    Install with: pip install 'langgraph-cli[inmem]>=0.4.5'")
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("LangGraph Studio Setup Test")
    print("=" * 60)

    results = []

    # Python 버전 테스트
    results.append(("Python Version", test_python_version()))

    # LangGraph 버전 테스트
    results.append(("LangGraph Version", test_langgraph_version()))

    # Import 테스트
    results.append(("Imports", test_imports()))

    # 그래프 생성 테스트
    results.append(("Graph Creation", test_graph_creation()))

    # CLI 테스트
    results.append(("LangGraph CLI", test_langgraph_cli()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! You can run 'langgraph dev' to start Studio.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
