# -*- coding: utf-8 -*-
"""테스트 직접 실행 및 로그 저장"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로그 파일 경로
log_file = project_root / "test_workflow_log.txt"

# 로그 파일 초기화
with open(log_file, 'w', encoding='utf-8') as log:
    log.write("=" * 60 + "\n")
    log.write(f"LangGraph 종합 테스트 실행 로그\n")
    log.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("=" * 60 + "\n\n")
    log.flush()

# 출력을 파일로 리다이렉션하는 클래스
class TeeOutput:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# 표준 출력을 파일과 콘솔 모두로 리다이렉션
with open(log_file, 'a', encoding='utf-8') as f:
    tee = TeeOutput(sys.stdout, f)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        # test_workflow.py의 main 함수 직접 실행
        from test_workflow import main

        print("테스트 시작...\n")
        success = main()

        print("\n" + "=" * 60)
        print(f"종료 코드: {0 if success else 1}")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        sys.exit(0 if success else 1)

    except Exception as e:
        error_msg = f"\n치명적 오류: {e}\n"
        print(error_msg)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = original_stdout
        print(f"\n로그 파일이 저장되었습니다: {log_file}")
