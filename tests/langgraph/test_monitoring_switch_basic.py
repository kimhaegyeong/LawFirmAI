# -*- coding: utf-8 -*-
"""
모니터링 전환 기본 테스트
환경변수 전환 로직만 테스트 (워크플로우 의존성 없음)
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch


def test_mode_set_and_restore():
    """환경변수 설정 및 복원 테스트"""
    print("="*80)
    print("테스트 1: 환경변수 설정 및 복원")
    print("="*80)

    # 원본 값 저장
    original_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    original_langfuse = os.environ.get("LANGFUSE_ENABLED")
    original_api_key = os.environ.get("LANGCHAIN_API_KEY")

    # LangSmith 모드로 설정 (API 키도 함께 설정)
    with MonitoringSwitch.set_mode(
        MonitoringMode.LANGSMITH,
        langsmith_api_key="test-api-key-for-testing"
    ):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LangSmith 트레이싱 활성화 확인"
        assert os.environ.get("LANGFUSE_ENABLED") == "false", "Langfuse 비활성화 확인"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.LANGSMITH, f"현재 모드는 LANGSMITH여야 함: {current_mode}"
        print("✅ LangSmith 모드 설정 확인")

    # 복원 확인
    assert os.environ.get("LANGCHAIN_TRACING_V2") == original_tracing, "환경변수 복원 확인"
    assert os.environ.get("LANGFUSE_ENABLED") == original_langfuse, "환경변수 복원 확인"
    print("✅ 환경변수 복원 확인")

    return True


def test_langfuse_mode():
    """Langfuse 모드 테스트"""
    print("\n" + "="*80)
    print("테스트 2: Langfuse 모드 설정")
    print("="*80)

    with MonitoringSwitch.set_mode(MonitoringMode.LANGFUSE):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false", "LangSmith 비활성화 확인"
        assert os.environ.get("LANGFUSE_ENABLED") == "true", "Langfuse 활성화 확인"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.LANGFUSE, f"현재 모드는 LANGFUSE여야 함: {current_mode}"
        print("✅ Langfuse 모드 설정 확인")

    return True


def test_both_mode():
    """Both 모드 테스트"""
    print("\n" + "="*80)
    print("테스트 3: Both 모드 설정")
    print("="*80)

    with MonitoringSwitch.set_mode(
        MonitoringMode.BOTH,
        langsmith_api_key="test-api-key-for-testing"
    ):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LangSmith 활성화 확인"
        assert os.environ.get("LANGFUSE_ENABLED") == "true", "Langfuse 활성화 확인"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.BOTH, f"현재 모드는 BOTH여야 함: {current_mode}"
        print("✅ Both 모드 설정 확인")

    return True


def test_none_mode():
    """None 모드 테스트"""
    print("\n" + "="*80)
    print("테스트 4: None 모드 설정")
    print("="*80)

    with MonitoringSwitch.set_mode(MonitoringMode.NONE):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false", "LangSmith 비활성화 확인"
        assert os.environ.get("LANGFUSE_ENABLED") == "false", "Langfuse 비활성화 확인"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.NONE, f"현재 모드는 NONE이어야 함: {current_mode}"
        print("✅ None 모드 설정 확인")

    return True


def test_mode_switching():
    """모드 전환 테스트"""
    print("\n" + "="*80)
    print("테스트 5: 순차적 모드 전환")
    print("="*80)

    modes_to_test = [
        (MonitoringMode.LANGSMITH, {"langsmith_api_key": "test-api-key"}),
        (MonitoringMode.LANGFUSE, {}),
        (MonitoringMode.BOTH, {"langsmith_api_key": "test-api-key"}),
        (MonitoringMode.NONE, {}),
    ]

    for mode, kwargs in modes_to_test:
        with MonitoringSwitch.set_mode(mode, **kwargs):
            current = MonitoringSwitch.get_current_mode()
            assert current == mode, f"모드 전환 실패: 예상={mode.value}, 실제={current.value}"
            print(f"  ✅ {mode.value} 모드 확인")

    print("✅ 모든 모드 전환 성공")
    return True


def test_mode_from_string():
    """문자열에서 모드 생성 테스트"""
    print("\n" + "="*80)
    print("테스트 6: 문자열에서 모드 생성")
    print("="*80)

    test_cases = [
        ("langsmith", MonitoringMode.LANGSMITH),
        ("LANGSMITH", MonitoringMode.LANGSMITH),
        ("langfuse", MonitoringMode.LANGFUSE),
        ("both", MonitoringMode.BOTH),
        ("none", MonitoringMode.NONE),
    ]

    for string_value, expected_mode in test_cases:
        mode = MonitoringMode.from_string(string_value)
        assert mode == expected_mode, f"문자열 '{string_value}' 파싱 실패"
        print(f"  ✅ '{string_value}' -> {mode.value}")

    # 잘못된 값 테스트
    try:
        MonitoringMode.from_string("invalid")
        assert False, "잘못된 값에 대해 예외가 발생해야 함"
    except ValueError:
        print("  ✅ 잘못된 값에 대한 예외 처리 확인")

    return True


def main():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print("모니터링 전환 기본 테스트 시작")
    print("="*80)

    tests = [
        ("환경변수 설정 및 복원", test_mode_set_and_restore),
        ("Langfuse 모드", test_langfuse_mode),
        ("Both 모드", test_both_mode),
        ("None 모드", test_none_mode),
        ("모드 전환", test_mode_switching),
        ("문자열에서 모드 생성", test_mode_from_string),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name} 실패: {e}")
            import traceback
            traceback.print_exc()

    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)

    passed = sum(1 for _, result, _ in results if result)
    failed = sum(1 for _, result, _ in results if not result)

    for test_name, result, error in results:
        status = "✅ 통과" if result else "❌ 실패"
        if error:
            print(f"  {status}: {test_name} ({error})")
        else:
            print(f"  {status}: {test_name}")

    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")

    if failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
