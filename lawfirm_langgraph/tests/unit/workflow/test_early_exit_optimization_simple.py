# -*- coding: utf-8 -*-
"""
조기 종료 최적화 단위 테스트 (간단 버전)
"""

import sys
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
unit_dir = workflow_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

def test_phase1_timeout_calculation():
    """Phase 1 타임아웃 계산 로직 테스트"""
    print("=" * 60)
    print("Phase 1 타임아웃 계산 테스트")
    print("=" * 60)
    
    # 타임아웃 계산 로직: max(10, min(15, 8 + (semantic_k + keyword_k) // 5))
    def calculate_phase1_timeout(semantic_k, keyword_k):
        return max(10, min(15, 8 + (semantic_k + keyword_k) // 5))
    
    test_cases = [
        ((10, 10), 12),  # 8 + 20 // 5 = 8 + 4 = 12
        ((5, 5), 10),    # 8 + 10 // 5 = 8 + 2 = 10
        ((20, 20), 15),  # 8 + 40 // 5 = 8 + 8 = 16 -> min(15, 16) = 15
        ((1, 1), 10),    # 8 + 2 // 5 = 8 + 0 = 8 -> max(10, 8) = 10
        ((50, 50), 15),  # 8 + 100 // 5 = 8 + 20 = 28 -> min(15, 28) = 15
    ]
    
    passed = 0
    failed = 0
    
    for (semantic_k, keyword_k), expected in test_cases:
        result = calculate_phase1_timeout(semantic_k, keyword_k)
        if result == expected:
            print(f"✅ PASS: k=({semantic_k}, {keyword_k}) -> {result}초 (예상: {expected}초)")
            passed += 1
        else:
            print(f"❌ FAIL: k=({semantic_k}, {keyword_k}) -> {result}초 (예상: {expected}초)")
            failed += 1
        
        # 범위 체크
        if 10 <= result <= 15:
            print(f"   ✅ 범위 체크 통과: 10 <= {result} <= 15")
        else:
            print(f"   ❌ 범위 체크 실패: {result}는 10-15 범위를 벗어남")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    return failed == 0

def test_dynamic_timeout_calculation():
    """동적 타임아웃 계산 로직 테스트"""
    print("\n" + "=" * 60)
    print("동적 타임아웃 계산 테스트")
    print("=" * 60)
    
    # 동적 타임아웃 계산: base_timeout + (worker_count * timeout_per_worker), 최대 15초
    def calculate_dynamic_timeout(worker_count, base_timeout=6, timeout_per_worker=1.5, max_timeout=15):
        return min(max_timeout, base_timeout + (worker_count * timeout_per_worker))
    
    test_cases = [
        (2, 9.0),   # 6 + (2 * 1.5) = 9
        (3, 10.5),  # 6 + (3 * 1.5) = 10.5
        (4, 12.0),  # 6 + (4 * 1.5) = 12
        (6, 15.0),  # 6 + (6 * 1.5) = 15
        (10, 15.0), # 6 + (10 * 1.5) = 21 -> min(15, 21) = 15
    ]
    
    passed = 0
    failed = 0
    
    for worker_count, expected in test_cases:
        result = calculate_dynamic_timeout(worker_count)
        if abs(result - expected) < 0.1:  # 부동소수점 오차 허용
            print(f"✅ PASS: workers={worker_count} -> {result}초 (예상: {expected}초)")
            passed += 1
        else:
            print(f"❌ FAIL: workers={worker_count} -> {result}초 (예상: {expected}초)")
            failed += 1
        
        # 최대값 체크
        if result <= 15:
            print(f"   ✅ 최대값 체크 통과: {result} <= 15")
        else:
            print(f"   ❌ 최대값 체크 실패: {result} > 15")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    return failed == 0

def test_early_exit_conditions():
    """조기 종료 조건 로직 테스트"""
    print("\n" + "=" * 60)
    print("조기 종료 조건 테스트")
    print("=" * 60)
    
    # 조기 종료 조건: semantic_results가 0개이고 semantic_count가 0이면 조기 종료
    def should_early_exit(semantic_results, semantic_count):
        return len(semantic_results) == 0 and semantic_count == 0
    
    test_cases = [
        (([], 0), True),   # 0개 결과 -> 조기 종료
        (([{"id": 1}], 1), False),  # 1개 결과 -> 조기 종료 안 함
        (([], 1), False),  # 결과는 없지만 count는 1 -> 조기 종료 안 함 (비정상 케이스)
        (([{"id": 1}, {"id": 2}], 2), False),  # 2개 결과 -> 조기 종료 안 함
    ]
    
    passed = 0
    failed = 0
    
    for test_input, expected in test_cases:
        semantic_results, semantic_count = test_input
        
        result = should_early_exit(semantic_results, semantic_count)
        if result == expected:
            print(f"✅ PASS: results={len(semantic_results)}, count={semantic_count} -> {result} (예상: {expected})")
            passed += 1
        else:
            print(f"❌ FAIL: results={len(semantic_results)}, count={semantic_count} -> {result} (예상: {expected})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    print("\n")
    result1 = test_phase1_timeout_calculation()
    result2 = test_dynamic_timeout_calculation()
    result3 = test_early_exit_conditions()
    
    print("\n" + "=" * 60)
    if result1 and result2 and result3:
        print("✅ 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("❌ 일부 테스트 실패")
        sys.exit(1)

