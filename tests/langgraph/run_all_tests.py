# -*- coding: utf-8 -*-
"""
전체 LangGraph 테스트 실행 스크립트
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Windows 비동기 환경에서 로깅 버퍼 에러 방지
class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 - detached 버퍼 에러 방지"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            # detached buffer 에러나 기타 스트림 에러 무시
            pass

logging.basicConfig(
    level=logging.ERROR,
    handlers=[SafeStreamHandler()],
    force=True  # 기존 설정을 강제로 재설정
)
logging.raiseExceptions = False  # 로깅 예외 무시
logger = logging.getLogger(__name__)

# 테스트 모듈 import
import importlib.util


def import_test_module(module_name):
    """테스트 모듈 동적 import"""
    module_path = Path(__file__).parent / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"테스트 파일을 찾을 수 없습니다: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 80)
    print("전체 LangGraph 테스트 실행")
    print("=" * 80)
    print(f"\n시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = []
    total_start = time.time()

    # 테스트 1: 모든 시나리오 테스트
    print("\n" + "=" * 80)
    print("테스트 1: 모든 시나리오 테스트 (test_all_scenarios)")
    print("=" * 80)
    try:
        module = import_test_module("test_all_scenarios")
        result = await module.main()
        test_results.append(("모든 시나리오 테스트", result == 0))
    except Exception as e:
        print(f"  ❌ [ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("모든 시나리오 테스트", False))

    # 테스트 2: 최적화된 워크플로우 테스트
    print("\n" + "=" * 80)
    print("테스트 2: 최적화된 워크플로우 테스트 (test_optimized_workflow)")
    print("=" * 80)
    try:
        module = import_test_module("test_optimized_workflow")
        await module.main()
        # 이 테스트는 exit code를 반환하지 않으므로 성공으로 간주
        test_results.append(("최적화된 워크플로우 테스트", True))
    except Exception as e:
        print(f"  ❌ [ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("최적화된 워크플로우 테스트", False))

    # 테스트 3: 노드 통합 테스트
    print("\n" + "=" * 80)
    print("테스트 3: 노드 통합 테스트 (test_node_integration)")
    print("=" * 80)
    try:
        module = import_test_module("test_node_integration")
        result = await module.main()
        test_results.append(("노드 통합 테스트", result == 0))
    except Exception as e:
        print(f"  ❌ [ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("노드 통합 테스트", False))

    # 테스트 4: 간단한 노드 통합 테스트
    print("\n" + "=" * 80)
    print("테스트 4: 간단한 노드 통합 테스트 (test_node_integration_simple)")
    print("=" * 80)
    try:
        module = import_test_module("test_node_integration_simple")
        result = await module.test_integration()
        test_results.append(("간단한 노드 통합 테스트", result == 0))
    except Exception as e:
        print(f"  ❌ [ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("간단한 노드 통합 테스트", False))

    # 테스트 5: 프롬프트 개선 테스트
    print("\n" + "=" * 80)
    print("테스트 5: 프롬프트 개선 테스트 (test_prompt_improvements)")
    print("=" * 80)
    test_file_path = Path(__file__).parent / "test_prompt_improvements.py"
    if not test_file_path.exists():
        print("  ⏭️  [SKIP] 테스트 파일이 없어 건너뜁니다.")
        test_results.append(("프롬프트 개선 테스트", None))  # None은 건너뛴 테스트를 의미
    else:
        try:
            module = import_test_module("test_prompt_improvements")
            result = await module.main()
            test_results.append(("프롬프트 개선 테스트", result == 0))
        except Exception as e:
            print(f"  ❌ [ERROR] 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            test_results.append(("프롬프트 개선 테스트", False))

    total_elapsed = time.time() - total_start

    # 최종 결과 요약
    print("\n" + "=" * 80)
    print("📊 전체 테스트 결과 요약")
    print("=" * 80)

    # None은 건너뛴 테스트, True/False는 실행된 테스트 결과
    skipped = sum(1 for _, result in test_results if result is None)
    passed = sum(1 for _, result in test_results if result is True)
    failed = sum(1 for _, result in test_results if result is False)
    total_executed = passed + failed

    for test_name, result in test_results:
        if result is None:
            status = "⏭️  SKIP"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {test_name}: {status}")

    if skipped > 0:
        print(f"\n전체: {passed}/{total_executed} 테스트 통과 ({skipped}개 건너뜀)")
    else:
        print(f"\n전체: {passed}/{total_executed} 테스트 통과")
    print(f"총 실행 시간: {total_elapsed:.2f}초")
    print(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if total_executed > 0 and passed == total_executed:
        print("\n✅ 모든 테스트 통과!")
        return 0
    elif total_executed > 0:
        print(f"\n⚠️ {failed}개 테스트 실패")
        return 1
    else:
        print("\n⚠️ 실행된 테스트가 없습니다.")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
