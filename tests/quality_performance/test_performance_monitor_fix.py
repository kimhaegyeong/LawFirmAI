#!/usr/bin/env python3
"""
PerformanceMonitor log_request 오류 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_performance_monitor():
    """PerformanceMonitor 테스트"""
    print("=== PerformanceMonitor log_request 테스트 ===")
    
    try:
        # gradio/app.py의 PerformanceMonitor 클래스 테스트
        from gradio.app import PerformanceMonitor
        
        print("PerformanceMonitor 클래스 임포트 성공")
        
        # 인스턴스 생성
        monitor = PerformanceMonitor()
        print("PerformanceMonitor 인스턴스 생성 성공")
        
        # log_request 메서드 테스트 (기본 매개변수)
        monitor.log_request(response_time=1.5, success=True)
        print("log_request(response_time, success) 호출 성공")
        
        # log_request 메서드 테스트 (operation 매개변수 포함)
        monitor.log_request(response_time=2.0, success=True, operation="test_operation")
        print("log_request(response_time, success, operation) 호출 성공")
        
        # 통계 조회
        stats = monitor.get_stats()
        print(f"통계 조회 성공: {stats}")
        
        print("모든 PerformanceMonitor 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"PerformanceMonitor 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_service_integration():
    """ChatService와 PerformanceMonitor 통합 테스트"""
    print("\n=== ChatService 통합 테스트 ===")
    
    try:
        from source.utils.config import Config
        from source.services.chat_service import ChatService
        import asyncio
        
        print("ChatService 초기화 중...")
        config = Config()
        chat_service = ChatService(config)
        
        # 간단한 메시지 처리 테스트
        test_message = "안녕하세요"
        session_id = "test_session_performance"
        user_id = "test_user_performance"
        
        print(f"테스트 메시지: {test_message}")
        
        # 비동기 함수 실행
        async def run_test():
            result = await chat_service.process_message(test_message, None, session_id, user_id)
            return result
        
        result = asyncio.run(run_test())
        
        if result and isinstance(result, dict):
            print("ChatService 메시지 처리 성공")
            print(f"응답 키: {list(result.keys())}")
            return True
        else:
            print("ChatService 메시지 처리 실패")
            return False
            
    except Exception as e:
        print(f"ChatService 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PerformanceMonitor 오류 수정 테스트 시작...")
    
    # PerformanceMonitor 테스트
    monitor_success = test_performance_monitor()
    
    # ChatService 통합 테스트
    integration_success = test_chat_service_integration()
    
    # 최종 결과
    print(f"\n=== 최종 결과 ===")
    print(f"PerformanceMonitor 테스트: {'통과' if monitor_success else '실패'}")
    print(f"ChatService 통합 테스트: {'통과' if integration_success else '실패'}")
    
    if monitor_success and integration_success:
        print("모든 테스트 통과! log_request 오류가 해결되었습니다.")
    else:
        print("일부 테스트 실패. 추가 수정이 필요합니다.")
