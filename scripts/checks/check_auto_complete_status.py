"""
자동 완료 스크립트 상태 확인
"""
import sys
import os
from pathlib import Path
from datetime import datetime

def check_status():
    """자동 완료 스크립트 상태 확인"""
    log_file = Path("logs/auto_complete_dynamic_chunking.log")
    error_file = Path("logs/auto_complete_dynamic_chunking_error.log")
    
    print("=" * 80)
    print("자동 완료 스크립트 상태 확인")
    print("=" * 80)
    print()
    
    # 로그 파일 확인
    if log_file.exists():
        print(f"✓ 로그 파일 존재: {log_file}")
        print(f"  크기: {log_file.stat().st_size} bytes")
        print(f"  수정 시간: {datetime.fromtimestamp(log_file.stat().st_mtime)}")
        print()
        
        # 최근 로그 확인
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print("최근 로그 (마지막 10줄):")
                    print("-" * 80)
                    for line in lines[-10:]:
                        print(line.rstrip())
                    print("-" * 80)
                else:
                    print("로그 파일이 비어있습니다.")
        except Exception as e:
            print(f"로그 파일 읽기 오류: {e}")
    else:
        print(f"✗ 로그 파일 없음: {log_file}")
    
    print()
    
    # 에러 로그 확인
    if error_file.exists():
        print(f"✓ 에러 로그 파일 존재: {error_file}")
        print(f"  크기: {error_file.stat().st_size} bytes")
        if error_file.stat().st_size > 0:
            print("  ⚠️ 에러 로그에 내용이 있습니다. 확인이 필요합니다.")
            try:
                with open(error_file, 'r', encoding='utf-8') as f:
                    error_lines = f.readlines()
                    if error_lines:
                        print("  최근 에러 (마지막 5줄):")
                        for line in error_lines[-5:]:
                            print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"  에러 로그 읽기 오류: {e}")
    else:
        print(f"✓ 에러 로그 파일 없음 (정상)")
    
    print()
    print("=" * 80)
    print("프로세스 확인")
    print("=" * 80)
    print("다음 명령어로 프로세스 확인:")
    print("  Get-Process python | Where-Object {$_.CommandLine -like '*auto_complete*'}")
    print()
    print("재임베딩 진행 상황 확인:")
    print("  python scripts/monitor_re_embedding_progress.py --db data/lawfirm_v2.db --version-id 5")

if __name__ == "__main__":
    check_status()

