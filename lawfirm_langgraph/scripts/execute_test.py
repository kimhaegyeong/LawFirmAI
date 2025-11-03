# -*- coding: utf-8 -*-
"""테스트 실행 스크립트"""

import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    test_script = project_root / "test_workflow.py"
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"

    if not python_exe.exists():
        print(f"오류: Python 실행 파일을 찾을 수 없습니다: {python_exe}")
        return 1

    if not test_script.exists():
        print(f"오류: 테스트 스크립트를 찾을 수 없습니다: {test_script}")
        return 1

    print("테스트 실행 중...")
    print(f"Python: {python_exe}")
    print(f"스크립트: {test_script}")
    print()

    try:
        # 테스트 실행
        result = subprocess.run(
            [str(python_exe), str(test_script)],
            cwd=str(project_root),
            capture_output=False,  # 출력을 직접 표시
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        print()
        print(f"종료 코드: {result.returncode}")
        return result.returncode

    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
