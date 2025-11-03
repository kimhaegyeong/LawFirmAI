# -*- coding: utf-8 -*-
"""테스트 실행 및 로그 저장 스크립트"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    test_script = project_root / "test_workflow.py"
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    log_file = project_root / "test_workflow_log.txt"

    if not python_exe.exists():
        print(f"오류: Python 실행 파일을 찾을 수 없습니다: {python_exe}")
        return 1

    if not test_script.exists():
        print(f"오류: 테스트 스크립트를 찾을 수 없습니다: {test_script}")
        return 1

    print("=" * 60)
    print("LangGraph 종합 테스트 실행")
    print("=" * 60)
    print(f"Python: {python_exe}")
    print(f"스크립트: {test_script}")
    print(f"로그 파일: {log_file}")
    print()
    print("테스트 실행 중...")

    try:
        # 테스트 실행 및 로그 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"LangGraph 종합 테스트 실행 로그\n")
            f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            # 테스트 실행
            result = subprocess.run(
                [str(python_exe), str(test_script)],
                cwd=str(project_root),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"종료 코드: {result.returncode}\n")
            f.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

        # 로그 파일 읽기
        print("\n테스트 완료! 로그 파일 내용:")
        print("=" * 60)

        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            print(log_content)

        print("=" * 60)
        print(f"\n로그 파일: {log_file}")
        print(f"종료 코드: {result.returncode}")

        return result.returncode

    except Exception as e:
        error_msg = f"오류 발생: {e}\n"
        print(error_msg)

        # 오류도 로그 파일에 기록
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
                import traceback
                f.write(traceback.format_exc())
        except:
            pass

        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
