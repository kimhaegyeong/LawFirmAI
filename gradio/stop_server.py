#!/usr/bin/env python3
"""
Gradio 서버 종료 스크립트
개발 규칙에 따라 PID 파일을 읽어서 해당 프로세스만 종료합니다.

⚠️ 중요: taskkill /f /im python.exe 사용 금지
✅ 올바른 방법: PID 기반 종료 또는 제공된 스크립트 사용
"""
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path

def stop_by_port():
    """포트 7860을 사용하는 프로세스 찾아서 종료"""
    try:
        # netstat으로 포트 7860을 사용하는 프로세스 찾기
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, encoding='cp949')
        
        pids = []
        for line in result.stdout.split('\n'):
            if ':7860' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.append(int(pid))
        
        if not pids:
            print("ERROR: 포트 7860을 사용하는 프로세스를 찾을 수 없습니다.")
            return False
        
        print(f"포트 7860을 사용하는 프로세스 발견: {pids}")
        
        # 각 PID에 대해 종료 시도
        for pid in pids:
            try:
                # 프로세스가 Python인지 확인
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                      capture_output=True, text=True, encoding='cp949')
                if 'python.exe' in result.stdout:
                    print(f"Python 프로세스 {pid}를 종료하는 중...")
                    
                    # Windows에서 taskkill 사용
                    try:
                        subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                                     capture_output=True, text=True, encoding='cp949')
                        print(f"OK: 프로세스 {pid}가 성공적으로 종료되었습니다.")
                    except Exception as e:
                        print(f"ERROR: 프로세스 {pid} 종료 실패: {e}")
                else:
                    print(f"WARNING: PID {pid}는 Python 프로세스가 아닙니다. 건너뜁니다.")
            except Exception as e:
                print(f"ERROR: 프로세스 {pid} 종료 중 오류: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 포트 기반 종료 중 오류: {e}")
        return False

def stop_gradio_server():
    """Gradio 서버 종료"""
    pid_file = Path("gradio_server.pid")
    
    if not pid_file.exists():
        print("WARNING: PID 파일을 찾을 수 없습니다. 포트 7860에서 실행 중인 프로세스를 찾습니다...")
        return stop_by_port()
    
    try:
        # PID 파일 읽기
        with open(pid_file, 'r', encoding='utf-8') as f:
            pid_data = json.load(f)
        
        pid = pid_data.get('pid')
        start_time = pid_data.get('start_time', 0)
        
        if not pid:
            print("ERROR: PID 정보를 찾을 수 없습니다.")
            return False
        
        print(f"서버 정보:")
        print(f"   PID: {pid}")
        print(f"   시작 시간: {time.ctime(start_time)}")
        
        # 프로세스 존재 확인 (Windows 호환)
        try:
            # tasklist로 프로세스 존재 확인
            check_result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                        capture_output=True, text=True, encoding='cp949')
            
            if str(pid) in check_result.stdout and 'python.exe' in check_result.stdout:
                print(f"OK: 프로세스 {pid}가 실행 중입니다.")
            else:
                print(f"ERROR: 프로세스 {pid}가 존재하지 않습니다.")
                # PID 파일 삭제
                pid_file.unlink()
                return False
        except Exception as e:
            print(f"ERROR: 프로세스 존재 확인 중 오류: {e}")
            return False
        
        # 프로세스 종료 (Windows 호환)
        print(f"프로세스 {pid}를 종료하는 중...")
        try:
            # Windows에서는 subprocess를 사용하여 프로세스 종료
            result = subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                                  capture_output=True, text=True, encoding='cp949')
            
            if result.returncode == 0:
                print("OK: 프로세스가 성공적으로 종료되었습니다.")
            else:
                print(f"WARNING: 프로세스 종료 중 문제 발생: {result.stderr}")
                # 프로세스가 실제로 종료되었는지 확인
                try:
                    # tasklist로 프로세스 존재 확인
                    check_result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                                capture_output=True, text=True, encoding='cp949')
                    if str(pid) not in check_result.stdout:
                        print("OK: 프로세스가 실제로 종료되었습니다.")
                    else:
                        print("ERROR: 프로세스가 여전히 실행 중입니다.")
                        return False
                except Exception as e:
                    print(f"ERROR: 프로세스 상태 확인 중 오류: {e}")
                    return False
            
            # PID 파일 삭제
            if pid_file.exists():
                pid_file.unlink()
                print("OK: PID 파일이 삭제되었습니다.")
            
            return True
            
        except Exception as e:
            print(f"ERROR: 프로세스 종료 중 오류 발생: {e}")
            return False
            
    except Exception as e:
        print(f"ERROR: PID 파일 읽기 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("LawFirmAI Gradio 서버 종료")
    print("=" * 40)
    
    if stop_gradio_server():
        print("\nOK: 서버가 성공적으로 종료되었습니다.")
        sys.exit(0)
    else:
        print("\nERROR: 서버 종료에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
