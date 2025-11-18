#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 테스트 실행 스크립트
pytest 내부 버퍼 오류를 피하기 위해 직접 테스트 실행
"""
import sys
import os
import subprocess

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

# 가상환경 Python 경로 찾기
venv_paths = [
    os.path.join(parent_root, "api", "venv", "Scripts", "python.exe"),
    os.path.join(project_root, "venv", "Scripts", "python.exe"),
]

python_exe = sys.executable
for venv_path in venv_paths:
    if os.path.exists(venv_path):
        python_exe = venv_path
        print(f"[INFO] Using venv: {venv_path}")
        break

# 테스트 실행
test_dir = os.path.join(project_root, "tests", "langgraph_core")
os.chdir(project_root)

cmd = [
    python_exe,
    "-m", "pytest",
    test_dir,
    "-v",
    "--tb=short",
    "--no-cov",
    "-x"  # 첫 번째 실패에서 중단
]

print(f"[INFO] Running: {' '.join(cmd)}")
print("=" * 80)

result = subprocess.run(cmd, cwd=project_root)
sys.exit(result.returncode)

