# -*- coding: utf-8 -*-
"""
수동 테스트 실행 스크립트
테스트를 단계별로 실행하고 결과를 확인합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step(step_name, test_func):
    """테스트 단계 실행"""
    print("\n" + "=" * 60)
    print(f"{step_name}")
    print("=" * 60)
    try:
        result = test_func()
        if result:
            print(f"✓ {step_name} 통과")
        else:
            print(f"✗ {step_name} 실패")
        return result
    except Exception as e:
        print(f"✗ {step_name} 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Import 테스트"""
    import langgraph
    from config.langgraph_config import LangGraphConfig
    from graph import graph, app, create_graph, create_app
    from langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    from langgraph_core.services.workflow_service import LangGraphWorkflowService
    print(f"  LangGraph 버전: {getattr(langgraph, '__version__', 'N/A')}")
    return True

def test_config():
    """설정 로딩 테스트"""
    from config.langgraph_config import LangGraphConfig
    config = LangGraphConfig.from_env()
    print(f"  LangGraph 활성화: {config.langgraph_enabled}")
    print(f"  LLM 제공자: {config.llm_provider}")
    print(f"  Google 모델: {config.google_model}")
    errors = config.validate()
    if errors:
        print(f"  ⚠ 경고: {len(errors)}개의 설정 문제")
        for error in errors[:3]:  # 처음 3개만 표시
            print(f"    - {error}")
    return True

def test_graph():
    """그래프 생성 테스트"""
    from graph import create_graph
    g = create_graph()
    print(f"  그래프 타입: {type(g).__name__}")
    if hasattr(g, 'nodes'):
        print(f"  노드 수: {len(g.nodes)}")
    return True

def test_app():
    """앱 생성 테스트"""
    from graph import create_app
    a = create_app()
    print(f"  앱 타입: {type(a).__name__}")
    if hasattr(a, 'invoke'):
        print("  invoke 메서드 존재")
    return True

def test_workflow_execution():
    """워크플로우 실행 테스트 (선택적)"""
    from graph import create_app
    a = create_app()

    test_input = {
        "messages": [
            {
                "role": "user",
                "content": "안녕하세요. 테스트입니다."
            }
        ]
    }

    print(f"  입력: {test_input['messages'][0]['content']}")
    print("  워크플로우 실행 중...")

    try:
        result = a.invoke(test_input)
        print("  ✓ 워크플로우 실행 완료")
        if isinstance(result, dict) and "messages" in result:
            print(f"  응답 메시지 수: {len(result['messages'])}")
            if result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                    print(f"  응답 길이: {len(content)} 문자")
        return True
    except Exception as e:
        print(f"  ⚠ 워크플로우 실행 실패 (환경 설정 문제일 수 있음): {e}")
        print("  (이것은 치명적이지 않을 수 있습니다. Graph와 App 생성이 성공했다면 정상입니다.)")
        return False  # 치명적이지 않으므로 False를 반환하지만 계속 진행

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("LangGraph 종합 테스트")
    print("=" * 60)

    results = []

    # 각 테스트 단계 실행
    results.append(("Import 테스트", test_step("1. Import 테스트", test_imports)))
    results.append(("설정 로딩", test_step("2. 설정 로딩 테스트", test_config)))
    results.append(("그래프 생성", test_step("3. 그래프 생성 테스트", test_graph)))
    results.append(("앱 생성", test_step("4. 앱 생성 테스트", test_app)))

    # 워크플로우 실행 테스트 (선택적, 실패해도 계속 진행)
    workflow_result = test_step("5. 워크플로우 실행 테스트", test_workflow_execution)
    results.append(("워크플로우 실행", workflow_result))

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

    # 기본 테스트(워크플로우 실행 제외)가 모두 통과했는지 확인
    basic_tests = [r for name, r in results if name != "워크플로우 실행"]
    basic_passed = sum(1 for r in basic_tests if r)
    basic_total = len(basic_tests)

    if basic_passed == basic_total:
        print("\n✓ 기본 테스트 모두 통과! LangGraph가 정상적으로 동작합니다.")
        print("\n다음 단계:")
        print("  1. 'langgraph dev' 명령어로 LangGraph Studio 실행")
        print("  2. 브라우저에서 http://localhost:8123 접속")

        if not workflow_result:
            print("\n⚠ 참고: 워크플로우 실행 테스트는 실패했지만, 이는 LLM API 키 설정 문제일 수 있습니다.")
            print("  Graph와 App 생성이 성공했다면 LangGraph Studio를 사용할 수 있습니다.")

        return True
    else:
        print("\n✗ 일부 기본 테스트 실패. 위의 오류를 확인하세요.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
