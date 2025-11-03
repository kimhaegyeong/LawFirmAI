# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 종합 테스트
실제 워크플로우가 정상 동작하는지 테스트합니다.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """필수 모듈 import 테스트"""
    print("\n" + "=" * 60)
    print("1. Import 테스트")
    print("=" * 60)

    try:
        # LangGraph 기본 모듈
        print("  - LangGraph 모듈...")
        import langgraph
        from langgraph.graph import StateGraph
        print(f"    ✓ LangGraph 버전: {getattr(langgraph, '__version__', 'N/A')}")

        # 설정 모듈
        print("  - 설정 모듈...")
        # config는 lawfirm_langgraph/config에 있음
        import sys
        from pathlib import Path
        lawfirm_root = Path(__file__).parent.parent.parent.resolve()
        sys.path.insert(0, str(lawfirm_root))
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        print("    ✓ Config 모듈 로드 성공")

        # 워크플로우 모듈
        print("  - 워크플로우 모듈...")
        from langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from langgraph_core.services.workflow_service import LangGraphWorkflowService
        print("    ✓ 워크플로우 모듈 로드 성공")

        # Graph export
        print("  - Graph export 모듈...")
        from graph import graph, app, create_graph, create_app
        print("    ✓ Graph export 모듈 로드 성공")

        return True
    except Exception as e:
        print(f"    ✗ Import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """설정 로딩 테스트"""
    print("\n" + "=" * 60)
    print("2. 설정 로딩 테스트")
    print("=" * 60)

    try:
        # config는 lawfirm_langgraph/config에 있음
        import sys
        from pathlib import Path
        lawfirm_root = Path(__file__).parent.parent.parent.resolve()
        sys.path.insert(0, str(lawfirm_root))
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

        print("  - 환경 변수에서 설정 로드...")
        config = LangGraphConfig.from_env()
        print(f"    ✓ LangGraph 활성화: {config.langgraph_enabled}")
        print(f"    ✓ LLM 제공자: {config.llm_provider}")
        print(f"    ✓ Google 모델: {config.google_model}")
        print(f"    ✓ 최대 반복 횟수: {config.max_iterations}")
        print(f"    ✓ 재귀 제한: {config.recursion_limit}")

        # 설정 유효성 검사
        print("  - 설정 유효성 검사...")
        errors = config.validate()
        if errors:
            print(f"    ⚠ 경고: {len(errors)}개의 설정 문제 발견")
            for error in errors:
                print(f"      - {error}")
        else:
            print("    ✓ 설정 유효성 검사 통과")

        return True
    except Exception as e:
        print(f"    ✗ 설정 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_creation():
    """그래프 생성 테스트"""
    print("\n" + "=" * 60)
    print("3. 그래프 생성 테스트")
    print("=" * 60)

    try:
        from graph import graph, create_graph
        from langgraph.graph import StateGraph

        print("  - create_graph() 함수 실행...")
        g = create_graph()
        print(f"    ✓ 그래프 생성 성공: {type(g).__name__}")

        print("  - graph 인스턴스 확인...")
        g_instance = graph()
        print(f"    ✓ 그래프 인스턴스 생성: {type(g_instance).__name__}")

        # 그래프 구조 확인
        if hasattr(g_instance, 'nodes'):
            print(f"    ✓ 그래프 노드 수: {len(g_instance.nodes)}")

        if hasattr(g_instance, 'edges'):
            print(f"    ✓ 그래프 엣지 수: {len(g_instance.edges)}")

        return True
    except Exception as e:
        print(f"    ✗ 그래프 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_creation():
    """앱 생성 테스트"""
    print("\n" + "=" * 60)
    print("4. 앱 생성 테스트")
    print("=" * 60)

    try:
        from graph import app, create_app

        print("  - create_app() 함수 실행...")
        a = create_app()
        print(f"    ✓ 앱 생성 성공: {type(a).__name__}")

        print("  - app 인스턴스 확인...")
        a_instance = app()
        print(f"    ✓ 앱 인스턴스 생성: {type(a_instance).__name__}")

        # 앱 메서드 확인
        if hasattr(a_instance, 'invoke'):
            print("    ✓ invoke 메서드 존재")
        if hasattr(a_instance, 'stream'):
            print("    ✓ stream 메서드 존재")
        if hasattr(a_instance, 'get_graph'):
            print("    ✓ get_graph 메서드 존재")

        return True
    except Exception as e:
        print(f"    ✗ 앱 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_workflow_execution():
    """간단한 워크플로우 실행 테스트"""
    print("\n" + "=" * 60)
    print("5. 워크플로우 실행 테스트 (간단한 입력)")
    print("=" * 60)

    try:
        from graph import app

        print("  - 앱 인스턴스 생성...")
        a = app()

        # 테스트 입력 준비
        test_input = {
            "messages": [
                {
                    "role": "user",
                    "content": "안녕하세요. 간단한 테스트 질문입니다."
                }
            ]
        }

        print("  - 워크플로우 실행 (간단한 질문)...")
        print("    입력:", test_input["messages"][0]["content"])

        # 워크플로우 실행 (타임아웃 설정)
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("워크플로우 실행이 타임아웃되었습니다.")

        # Windows에서는 signal.alarm을 지원하지 않으므로 try-except로 처리
        try:
            result = a.invoke(test_input)

            print("    ✓ 워크플로우 실행 완료")

            # 결과 확인
            if isinstance(result, dict):
                print(f"    ✓ 결과 타입: dict")
                if "messages" in result:
                    print(f"    ✓ 메시지 수: {len(result['messages'])}")
                    if result["messages"]:
                        last_message = result["messages"][-1]
                        if hasattr(last_message, "content"):
                            print(f"    ✓ 응답 길이: {len(last_message.content)} 문자")
                            print(f"    ✓ 응답 미리보기: {last_message.content[:100]}...")
                else:
                    print(f"    ✓ 결과 키: {list(result.keys())}")
            else:
                print(f"    ✓ 결과 타입: {type(result).__name__}")

            return True
        except TimeoutError as e:
            print(f"    ⚠ {e}")
            return False
        except Exception as e:
            print(f"    ⚠ 워크플로우 실행 중 오류: {e}")
            print("    (이는 환경 설정 문제일 수 있습니다)")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"    ✗ 워크플로우 실행 준비 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_langgraph_cli():
    """LangGraph CLI 확인"""
    print("\n" + "=" * 60)
    print("6. LangGraph CLI 확인")
    print("=" * 60)

    try:
        import shutil

        cli_path = shutil.which('langgraph')

        if cli_path:
            print(f"  ✓ LangGraph CLI 발견: {cli_path}")

            # CLI 버전 확인
            import subprocess
            try:
                result = subprocess.run(
                    ['langgraph', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"    ✓ CLI 버전: {result.stdout.strip()}")
                else:
                    print("    ⚠ CLI 버전 확인 실패")
            except Exception:
                print("    ⚠ CLI 버전 확인 실패")

            return True
        else:
            print("  ⚠ LangGraph CLI를 PATH에서 찾을 수 없습니다")
            print("    설치: pip install 'langgraph-cli[inmem]>=0.4.5'")
            return False
    except Exception as e:
        print(f"  ✗ CLI 확인 실패: {e}")
        return False

def test_langgraph_json():
    """langgraph.json 설정 확인"""
    print("\n" + "=" * 60)
    print("7. langgraph.json 설정 확인")
    print("=" * 60)

    try:
        import json

        json_path = project_root / "langgraph.json"

        if not json_path.exists():
            print("  ✗ langgraph.json 파일이 없습니다")
            return False

        print(f"  ✓ langgraph.json 파일 발견: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"    ✓ dependencies: {config.get('dependencies', [])}")
        print(f"    ✓ graphs: {list(config.get('graphs', {}).keys())}")
        print(f"    ✓ env: {config.get('env', 'N/A')}")

        # graph.py 파일 확인
        graph_path = project_root / "graph.py"
        if graph_path.exists():
            print(f"    ✓ graph.py 파일 존재")
        else:
            print(f"    ✗ graph.py 파일이 없습니다")
            return False

        return True
    except Exception as e:
        print(f"  ✗ 설정 확인 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("LangGraph 워크플로우 종합 테스트")
    print("=" * 60)

    results = []

    # 각 테스트 실행
    results.append(("Import 테스트", test_imports()))
    results.append(("설정 로딩", test_config_loading()))
    results.append(("그래프 생성", test_graph_creation()))
    results.append(("앱 생성", test_app_creation()))
    results.append(("langgraph.json 설정", test_langgraph_json()))
    results.append(("LangGraph CLI", test_langgraph_cli()))
    results.append(("워크플로우 실행", test_simple_workflow_execution()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n총 {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n✓ 모든 테스트 통과! LangGraph가 정상적으로 동작합니다.")
        print("\n다음 단계:")
        print("  1. 'langgraph dev' 명령어로 LangGraph Studio 실행")
        print("  2. 브라우저에서 http://localhost:8123 접속")
    elif passed >= total - 1:
        print("\n⚠ 대부분의 테스트 통과 (일부 비중요 테스트 실패)")
        print("  LangGraph Studio를 실행해볼 수 있습니다.")
    else:
        print("\n✗ 일부 테스트 실패. 위의 오류를 확인하세요.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
