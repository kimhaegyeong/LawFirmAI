#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 모니터링 테스트 스크립트
"""

def test_performance_monitoring():
    """성능 모니터링 서비스 테스트"""
    try:
        from source.services.performance_monitoring import PerformanceMonitor
        print("✅ 성능 모니터링 서비스 임포트 성공")
        
        # 인스턴스 생성
        monitor = PerformanceMonitor()
        print("✅ 인스턴스 생성 성공")
        
        # 모니터링 시작
        monitor.start_monitoring()
        print("✅ 모니터링 시작 성공")
        
        # 이벤트 로깅
        monitor.log_request(0.5, success=True)
        print("✅ 요청 로깅 성공")
        
        # 메트릭 수집
        metrics = monitor.get_current_metrics()
        print(f"✅ 메트릭 수집 성공: {len(metrics)}개")
        
        # 모니터링 중지
        monitor.stop_monitoring()
        print("✅ 모니터링 중지 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 성능 모니터링 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("성능 모니터링 테스트 시작")
    print("=" * 50)
    
    success = test_performance_monitoring()
    
    print("=" * 50)
    if success:
        print("🎉 성능 모니터링 테스트 성공!")
    else:
        print("💥 성능 모니터링 테스트 실패!")
