#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
langgraph_core 테스트 파일 전체 실행
"""
import sys
import os
import subprocess

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

os.chdir(project_root)

# 가상환경 Python 경로 찾기
venv_paths = [
    os.path.join(parent_root, "api", "venv", "Scripts", "python.exe"),
    os.path.join(project_root, "venv", "Scripts", "python.exe"),
]

python_exe = sys.executable
for venv_path in venv_paths:
    if os.path.exists(venv_path):
        python_exe = venv_path
        break

# 테스트 파일 목록
test_files = [
    "test_processing_reasoning_extractor.py",
    "test_state_modular_states.py",
    "test_processing_extractors.py",
    "test_processing_quality_validators.py",
    "test_processing_response_parsers.py",
    "test_prompt_builders.py",
    "test_chain_builders.py",
    "test_node_input_output_spec.py",
    "test_utils_workflow_routes.py",
    "test_workflow_service.py",
    "test_workflow_legal_workflow_enhanced.py",
    "test_state_utils.py",
    "test_tools_legal_search_tools.py",
    "test_utils_workflow_constants.py",
    "test_node_wrappers.py",
    "test_workflow_utils.py",
]

print("=" * 80)
print("Running all langgraph_core tests")
print("=" * 80)
print(f"Python: {python_exe}")
print(f"Working directory: {project_root}")
print("=" * 80)

total_passed = 0
total_failed = 0
total_tests = 0
failed_files = []

for test_file in test_files:
    print(f"\n{'=' * 80}")
    print(f"Testing: {test_file}")
    print('=' * 80)
    
    cmd = [python_exe, "run_single_test.py", test_file]
    result = subprocess.run(
        cmd, 
        cwd=project_root, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # 결과 파싱
    output = result.stdout or ""
    if output and "Results:" in output:
        lines = output.split('\n')
        for line in lines:
            if "Results:" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    results = parts[1].strip()
                    # "X passed, Y failed out of Z tests" 파싱
                    if "passed" in results and "failed" in results:
                        passed = int(results.split("passed")[0].strip())
                        failed = int(results.split("failed")[0].split(",")[1].strip())
                        total_passed += passed
                        total_failed += failed
                        total_tests += (passed + failed)
                        
                        if failed > 0:
                            failed_files.append(test_file)
    
    # 출력 표시
    print(output)
    if result.stderr:
        print("STDERR:", result.stderr)

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Total tests: {total_tests}")
print(f"Passed: {total_passed}")
print(f"Failed: {total_failed}")
print(f"Success rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")

if failed_files:
    print(f"\nFailed test files ({len(failed_files)}):")
    for f in failed_files:
        print(f"  - {f}")

print("=" * 80)

sys.exit(0 if total_failed == 0 else 1)

