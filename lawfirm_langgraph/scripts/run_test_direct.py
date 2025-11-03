# -*- coding: utf-8 -*-
"""직접 테스트 실행 스크립트"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
script_path = project_root / "test_workflow.py"
python_path = project_root / ".venv" / "Scripts" / "python.exe"

if not python_path.exists():
    print(f"오류: Python 실행 파일을 찾을 수 없습니다: {python_path}")
    sys.exit(1)

if not script_path.exists():
    print(f"오류: 테스트 스크립트를 찾을 수 없습니다: {script_path}")
    sys.exit(1)

print("=" * 60)
print("LangGraph 종합 테스트 실행")
print("=" * 60)
print()

# 테스트 실행
try:
    result = subprocess.run(
        [str(python_path), str(script_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # 출력 표시
    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("오류 출력:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    print()
    print("=" * 60)
    print(f"종료 코드: {result.returncode}")
    print("=" * 60)

    sys.exit(result.returncode)

except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
