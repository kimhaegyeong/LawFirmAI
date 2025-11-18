#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
직접 테스트 실행 스크립트
pytest 내부 버퍼 오류를 피하기 위해 unittest로 변환하여 실행
"""
import sys
import os
import unittest

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

# 테스트 파일 목록
test_files = [
    "tests.langgraph_core.test_processing_reasoning_extractor",
    "tests.langgraph_core.test_state_modular_states",
    "tests.langgraph_core.test_processing_extractors",
    "tests.langgraph_core.test_processing_quality_validators",
    "tests.langgraph_core.test_processing_response_parsers",
    "tests.langgraph_core.test_prompt_builders",
    "tests.langgraph_core.test_chain_builders",
    "tests.langgraph_core.test_node_input_output_spec",
    "tests.langgraph_core.test_utils_workflow_routes",
    "tests.langgraph_core.test_workflow_service",
    "tests.langgraph_core.test_workflow_legal_workflow_enhanced",
    "tests.langgraph_core.test_state_utils",
    "tests.langgraph_core.test_tools_legal_search_tools",
    "tests.langgraph_core.test_utils_workflow_constants",
    "tests.langgraph_core.test_node_wrappers",
    "tests.langgraph_core.test_workflow_utils",
]

os.chdir(project_root)

print("=" * 80)
print("Running langgraph_core tests")
print("=" * 80)

# 테스트 로더 생성
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# 각 테스트 모듈 로드
passed = 0
failed = 0
errors = 0

for test_module_name in test_files:
    try:
        print(f"\n[INFO] Loading: {test_module_name}")
        module = __import__(test_module_name, fromlist=[''])
        tests = loader.loadTestsFromModule(module)
        suite.addTests(tests)
        print(f"[OK] Loaded {tests.countTestCases()} tests from {test_module_name}")
    except Exception as e:
        print(f"[ERROR] Failed to load {test_module_name}: {e}")
        errors += 1
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Running tests...")
print("=" * 80)

# 테스트 실행
runner = unittest.TextTestRunner(verbosity=2, buffer=True)
result = runner.run(suite)

# 결과 출력
print("\n" + "=" * 80)
print("Test Results Summary")
print("=" * 80)
print(f"Tests run: {result.testsRun}")
print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
print(f"Skipped: {len(result.skipped)}")

if result.failures:
    print("\nFailures:")
    for test, traceback in result.failures:
        print(f"  - {test}: {traceback.split(chr(10))[-2]}")

if result.errors:
    print("\nErrors:")
    for test, traceback in result.errors:
        print(f"  - {test}: {traceback.split(chr(10))[-2]}")

sys.exit(0 if result.wasSuccessful() else 1)

