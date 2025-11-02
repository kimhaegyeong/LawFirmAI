# -*- coding: utf-8 -*-
"""
LangGraph 모니터링 전환 테스트
LangSmith와 Langfuse를 번갈아가며 사용하는 통합 테스트
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(project_root / ".env"))
except ImportError:
    pass

from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch

# WorkflowFactory는 선택적으로 import
try:
    from tests.langgraph.fixtures.workflow_factory import WorkflowFactory
    WORKFLOW_FACTORY_AVAILABLE = True
except ImportError as e:
    WORKFLOW_FACTORY_AVAILABLE = False
    print(f"⚠ WorkflowFactory를 사용할 수 없습니다: {e}")
    print("  langgraph 패키지가 설치되지 않았습니다.")
    print("  워크플로우 테스트를 건너뜁니다.")


class MonitoringSwitchTest:
    """모니터링 전환 테스트 클래스"""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_queries = [
            "계약서 작성 시 주의사항은 무엇인가요?",
            "판례 검색을 도와주세요",
            "법령 해설이 필요합니다"
        ]

    async def test_single_mode(self, mode: MonitoringMode, query: str) -> Dict[str, Any]:
        """
        단일 모니터링 모드로 테스트 실행

        Args:
            mode: 모니터링 모드
            query: 테스트 쿼리

        Returns:
            Dict: 테스트 결과
        """
        print(f"\n{'='*80}")
        print(f"테스트 모드: {mode.value}")
        print(f"쿼리: {query}")
        print(f"{'='*80}")

        result = {
            "mode": mode.value,
            "query": query,
            "success": False,
            "response": None,
            "error": None,
            "verification": None
        }

        try:
            # 모니터링 모드 설정
            with MonitoringSwitch.set_mode(mode):
                # 워크플로우 서비스 생성
                if not WORKFLOW_FACTORY_AVAILABLE or not WorkflowFactory.is_available():
                    result["error"] = "WorkflowFactory를 사용할 수 없습니다. langgraph 패키지를 설치하세요."
                    result["success"] = False
                    print(f"⚠ {result['error']}")
                    return result

                service = WorkflowFactory.get_workflow(mode, force_recreate=True)

                # 검증
                verification = MonitoringSwitch.verify_mode(service, mode)
                result["verification"] = verification

                if verification.get("warnings"):
                    print(f"⚠ 경고: {', '.join(verification['warnings'])}")

                # 쿼리 실행
                print(f"\n쿼리 실행 중...")
                response = await service.process_query(
                    query=query,
                    session_id=f"test_{mode.value}_{hash(query) % 10000}"
                )

                result["response"] = response
                result["success"] = True

                print(f"✅ 테스트 성공")
                if response.get("answer"):
                    print(f"답변 길이: {len(response.get('answer', ''))} 문자")

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def test_switch_between_modes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 모니터링 모드 간 전환 테스트

        Returns:
            Dict: 각 모드별 테스트 결과
        """
        print("\n" + "="*80)
        print("모니터링 모드 전환 테스트 시작")
        print("="*80)

        results_by_mode: Dict[str, List[Dict[str, Any]]] = {}

        # 테스트할 모드 목록
        test_modes = [
            MonitoringMode.LANGSMITH,
            MonitoringMode.LANGFUSE,
            MonitoringMode.BOTH,
            MonitoringMode.NONE
        ]

        for mode in test_modes:
            mode_results = []

            print(f"\n{'='*80}")
            print(f"모니터링 모드: {mode.value.upper()}")
            print(f"{'='*80}")

            # 각 쿼리로 테스트
            for query in self.test_queries[:1]:  # 첫 번째 쿼리만 빠른 테스트
                result = await self.test_single_mode(mode, query)
                mode_results.append(result)

            results_by_mode[mode.value] = mode_results

            # 캐시 정리 (다음 모드를 위해)
            if WORKFLOW_FACTORY_AVAILABLE and WorkflowFactory.is_available():
                WorkflowFactory.clear_cache(mode)

        return results_by_mode

    async def test_sequential_switch(self, query: str) -> List[Dict[str, Any]]:
        """
        순차적으로 모드를 전환하며 동일 쿼리 테스트

        Args:
            query: 테스트 쿼리

        Returns:
            List: 각 모드별 테스트 결과
        """
        print(f"\n{'='*80}")
        print("순차 모드 전환 테스트")
        print(f"쿼리: {query}")
        print(f"{'='*80}")

        results = []
        modes = [MonitoringMode.LANGSMITH, MonitoringMode.LANGFUSE]

        for mode in modes:
            result = await self.test_single_mode(mode, query)
            results.append(result)

            # 캐시 정리
            if WORKFLOW_FACTORY_AVAILABLE and WorkflowFactory.is_available():
                WorkflowFactory.clear_cache(mode)

        return results

    def print_results_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """테스트 결과 요약 출력"""
        print("\n" + "="*80)
        print("테스트 결과 요약")
        print("="*80)

        for mode, mode_results in results.items():
            success_count = sum(1 for r in mode_results if r.get("success"))
            total_count = len(mode_results)

            print(f"\n{mode.upper()}:")
            print(f"  성공: {success_count}/{total_count}")

            for i, result in enumerate(mode_results, 1):
                status = "✅" if result.get("success") else "❌"
                print(f"  {status} 테스트 {i}: {result.get('query', 'N/A')[:50]}...")
                if result.get("error"):
                    print(f"     오류: {result['error']}")
                if result.get("verification", {}).get("warnings"):
                    for warning in result["verification"]["warnings"]:
                        print(f"     ⚠ {warning}")


async def main():
    """메인 테스트 실행"""
    test_runner = MonitoringSwitchTest()

    print("\n" + "="*80)
    print("LangGraph 모니터링 전환 테스트")
    print("="*80)

    # 현재 환경변수 확인
    current_mode = MonitoringSwitch.get_current_mode()
    print(f"\n현재 모니터링 모드: {current_mode.value}")

    # 모드 전환 테스트 실행
    results = await test_runner.test_switch_between_modes()

    # 결과 요약
    test_runner.print_results_summary(results)

    # 성공률 계산
    total_tests = sum(len(mode_results) for mode_results in results.values())
    total_success = sum(
        sum(1 for r in mode_results if r.get("success"))
        for mode_results in results.values()
    )

    print(f"\n{'='*80}")
    print(f"전체 성공률: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        mode_str = sys.argv[1].lower()
        try:
            mode = MonitoringMode.from_string(mode_str)

            # 단일 모드 테스트
            async def test_single():
                test_runner = MonitoringSwitchTest()
                query = test_runner.test_queries[0]
                result = await test_runner.test_single_mode(mode, query)
                print(f"\n테스트 완료: {result.get('success', False)}")
                return result

            asyncio.run(test_single())
        except ValueError:
            print(f"알 수 없는 모니터링 모드: {mode_str}")
            print(f"사용 가능한 모드: {[m.value for m in MonitoringMode]}")
            sys.exit(1)
    else:
        # 전체 테스트 실행
        asyncio.run(main())
