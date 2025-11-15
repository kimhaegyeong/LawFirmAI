# -*- coding: utf-8 -*-
"""
서버를 시작하고 스트리밍 테스트를 실행하는 스크립트
"""
import subprocess
import sys
import os
import time
import requests
import signal

def check_server():
    """서버 상태 확인"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_server():
    """서버 시작"""
    if check_server():
        print("✅ 서버가 이미 실행 중입니다.")
        return None
    
    print("🚀 API 서버 시작 중...")
    api_dir = os.path.join(os.path.dirname(__file__), "..")
    server_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=api_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    
    # 서버 시작 대기
    print("⏳ 서버 시작 대기 중...")
    for i in range(30):
        time.sleep(1)
        if check_server():
            print(f"✅ 서버가 시작되었습니다! (PID: {server_process.pid})")
            return server_process
        if i % 5 == 0:
            print(f"   대기 중... ({i+1}/30초)")
    
    print("❌ 서버 시작 실패 (30초 타임아웃)")
    server_process.terminate()
    return None

def run_test():
    """테스트 실행"""
    test_file = os.path.join(os.path.dirname(__file__), "test_stream_simple.py")
    print("\n" + "=" * 80)
    print("📡 스트리밍 테스트 실행")
    print("=" * 80 + "\n")
    
    result = subprocess.run([sys.executable, test_file])
    return result.returncode == 0

def main():
    """메인 함수"""
    server_process = None
    
    try:
        # 서버 시작
        server_process = start_server()
        
        if not server_process and not check_server():
            print("❌ 서버를 시작할 수 없습니다.")
            return 1
        
        # 테스트 실행
        success = run_test()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 서버 종료 (선택사항 - 주석 처리하면 서버가 계속 실행됨)
        if server_process:
            print("\n⚠️ 서버를 종료하려면 Ctrl+C를 누르거나 서버 창을 닫으세요.")
            # server_process.terminate()

if __name__ == "__main__":
    sys.exit(main())

