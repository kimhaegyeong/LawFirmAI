# -*- coding: utf-8 -*-
"""
간단한 Gradio 테스트 스크립트
실제 Gradio 실행 전에 기본 기능이 동작하는지 확인
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_gradio_import():
    """Gradio 라이브러리 import 테스트"""
    try:
        import gradio as gr
        print("[OK] Gradio import 성공")
        print(f"  Gradio 버전: {gr.__version__}")
        return True
    except Exception as e:
        print(f"[FAIL] Gradio import 실패: {e}")
        return False

def test_gradio_interface_creation():
    """Gradio 인터페이스 생성 테스트"""
    try:
        import gradio as gr
        
        def simple_response(message):
            return f"테스트 응답: {message}"
        
        # 간단한 인터페이스 생성
        interface = gr.Interface(
            fn=simple_response,
            inputs="text",
            outputs="text",
            title="LawFirmAI 테스트",
            description="간단한 테스트 인터페이스"
        )
        
        print("[OK] Gradio 인터페이스 생성 성공")
        return True
        
    except Exception as e:
        print(f"[FAIL] Gradio 인터페이스 생성 실패: {e}")
        return False

def test_chat_service_integration():
    """ChatService와 Gradio 통합 테스트"""
    try:
        # 환경 변수 설정
        os.environ.setdefault('USE_LANGGRAPH', 'false')
        os.environ.setdefault('GEMINI_ENABLED', 'false')
        
        # ChatService 초기화
        from source.utils.config import Config
        from source.services.chat_service import ChatService
        
        config = Config()
        chat_service = ChatService(config)
        
        # 간단한 테스트 질의
        test_query = "안녕하세요"
        
        # 비동기 함수를 동기적으로 실행
        async def test_async():
            result = await chat_service.process_message(test_query)
            return result
        
        result = asyncio.run(test_async())
        
        if result and result.get("response"):
            print("[OK] ChatService 통합 테스트 성공")
            print(f"  테스트 질의: {test_query}")
            print(f"  응답 길이: {len(result['response'])} 문자")
            print(f"  신뢰도: {result.get('confidence', 0):.2f}")
            return True
        else:
            print("[FAIL] ChatService 통합 테스트 실패: 응답 없음")
            return False
            
    except Exception as e:
        print(f"[FAIL] ChatService 통합 테스트 실패: {e}")
        return False

def test_gradio_app_components():
    """Gradio 앱의 주요 컴포넌트 테스트"""
    try:
        # gradio/app.py의 주요 클래스 import 테스트
        import sys
        sys.path.append('gradio')
        from app import HuggingFaceSpacesApp
        
        app = HuggingFaceSpacesApp()
        print("[OK] HuggingFaceSpacesApp 클래스 생성 성공")
        
        # 시스템 상태 확인
        status = app.get_system_status()
        print(f"  초기화 상태: {status.get('initialized', False)}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Gradio 앱 컴포넌트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("LawFirmAI Gradio 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("Gradio Import", test_gradio_import),
        ("Gradio Interface Creation", test_gradio_interface_creation),
        ("ChatService Integration", test_chat_service_integration),
        ("Gradio App Components", test_gradio_app_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}] 테스트 중...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name} 테스트 중 예외 발생: {e}")
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"총 테스트: {total}")
    print(f"통과: {passed}")
    print(f"실패: {total - passed}")
    print(f"통과율: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n[SUCCESS] 모든 테스트 통과! Gradio 실행 준비 완료")
        return True
    else:
        print(f"\n[WARNING] {total - passed}개 테스트 실패. 문제 해결 필요")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
