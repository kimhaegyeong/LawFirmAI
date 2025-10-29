#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 테스트 파일 실행 스크립트
모든 테스트를 카테고리별로 실행하고 오류를 수정합니다.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 카테고리별 파일 목록
TEST_CATEGORIES = {
    "langgraph": [
        "test_langgraph.py",
        "test_langgraph_state_optimization.py",
        "test_langgraph_multi_turn.py",
        "test_all_state_systems.py",
        "test_core_state_systems.py",
        "test_state_reduction_performance.py",
        "test_monitoring_switch_basic.py",
        "test_profile_loading.py",
        "test_with_monitoring_switch.py",
    ],
    "integration": [
        "test_comprehensive_system.py",
        "test_integrated_system.py",
    ],
    "search": [
        "test_query_classification.py",
        "test_query_system.py",
        "test_classify_question_type.py",
        "test_hybrid_search_integration.py",
        "test_hybrid_search_simple.py",
        "test_rag_integration.py",
    ],
    "legal": [
        "test_legal_basis_system.py",
        "test_database_keyword_system.py",
        "test_term_integration_workflow.py",
    ],
    "monitoring": [
        "test_langsmith_integration.py",
        "test_langfuse_integration.py",
        "test_unified_prompt_integration.py",
    ],
    "quality_performance": [
        "test_quality_enhancement.py",
        "test_quality_improvement_workflow.py",
        "test_performance_benchmark.py",
        "test_performance_monitor_fix.py",
        "test_optimized_performance.py",
        "test_stress_system.py",
        "test_workflow_execution.py",
    ],
    "phase": [
        "test_phase1_context_enhancement.py",
        "test_phase2_personalization_analysis.py",
        "test_phase3_memory_quality.py",
    ],
}


def run_test_file(category: str, test_file: str) -> Tuple[bool, str]:
    """
    개별 테스트 파일 실행 (출력 억제, 실행 단계만 표시)

    Returns:
        (성공 여부, 에러 요약)
    """
    tests_dir = Path(__file__).parent
    test_path = tests_dir / category / test_file

    if not test_path.exists():
        return False, f"파일 없음: {test_path}"

    try:
        # 테스트 출력을 억제하고 에러만 캡처
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,  # 테스트 출력 억제
            stderr=subprocess.PIPE,      # 에러만 캡처
            text=True,
            timeout=300,  # 5분 타임아웃
            encoding='utf-8',
            errors='replace'
        )

        # 실패한 경우에만 에러 요약 추출
        error_summary = ""
        if result.returncode != 0 and result.stderr:
            error_lines = result.stderr.split('\n')
            # 마지막 의미있는 에러 라인만 추출
            error_summary = ' | '.join([line.strip() for line in error_lines[-3:] if line.strip()])

        if result.returncode == 0:
            return True, ""
        else:
            return False, error_summary

    except subprocess.TimeoutExpired:
        return False, "타임아웃 (5분 초과)"
    except Exception as e:
        return False, f"실행 오류: {str(e)}"


def run_category_tests(category: str) -> Dict[str, Tuple[bool, str]]:
    """카테고리별 테스트 실행"""
    print(f"\n{'='*80}")
    print(f"📁 {category.upper()} 테스트 실행")
    print(f"{'='*80}")

    results = {}
    test_files = TEST_CATEGORIES.get(category, [])

    print(f"총 {len(test_files)}개 테스트 파일 실행 중...\n")

    for i, test_file in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {test_file} 실행 중...", end=" ", flush=True)
        success, output = run_test_file(category, test_file)
        results[test_file] = (success, output)

        if success:
            print("✅ 통과")
        else:
            print("❌ 실패", end="")
            if output:
                print(f" - {output}")
            else:
                print()

    return results


def main():
    """메인 실행 함수"""
    print("="*80)
    print("LawFirmAI 전체 테스트 실행")
    print("="*80)

    all_results = {}
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "categories": {}
    }

    # 각 카테고리별 테스트 실행
    for category in TEST_CATEGORIES.keys():
        category_results = run_category_tests(category)
        all_results[category] = category_results

        # 카테고리별 통계
        passed = sum(1 for success, _ in category_results.values() if success)
        failed = len(category_results) - passed

        summary["categories"][category] = {
            "total": len(category_results),
            "passed": passed,
            "failed": failed
        }

        summary["total"] += len(category_results)
        summary["passed"] += passed
        summary["failed"] += failed

    # 최종 요약
    print(f"\n{'='*80}")
    print("📊 전체 테스트 실행 결과 요약")
    print(f"{'='*80}\n")

    for category, stats in summary["categories"].items():
        status = "✅" if stats["failed"] == 0 else "⚠️"
        print(f"{status} {category:20s}: {stats['passed']}/{stats['total']} 통과 ({stats['failed']} 실패)")

    print(f"\n총계: {summary['passed']}/{summary['total']} 통과 ({summary['failed']} 실패)")

    # 실패한 테스트 상세
    if summary["failed"] > 0:
        print(f"\n{'='*80}")
        print("❌ 실패한 테스트 상세")
        print(f"{'='*80}\n")

        for category, results in all_results.items():
            for test_file, (success, output) in results.items():
                if not success:
                    print(f"📄 {category}/{test_file}")
                    if output:
                        print(f"   {output}\n")

    return summary["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
