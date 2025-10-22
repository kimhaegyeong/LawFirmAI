#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatService 테스트 스크립트
Gradio에서 chat_service가 정상적으로 작동하는지 확인
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.utils.config import Config

async def test_chat_service():
    """ChatService 기본 기능 테스트"""
    print("=" * 60)
    print("ChatService 테스트 시작")
    print("=" * 60)
    
    try:
        # Config 초기화
        print("1. Config 초기화 중...")
        config = Config()
        print("   ✓ Config 초기화 성공")
        
        # ChatService 초기화
        print("2. ChatService 초기화 중...")
        chat_service = ChatService(config)
        print("   ✓ ChatService 초기화 성공")
        
        # 서비스 상태 확인
        print("3. 서비스 상태 확인 중...")
        status = chat_service.get_service_status()
        print(f"   서비스 상태: {status.get('overall_status', 'unknown')}")
        print(f"   LangGraph 활성화: {status.get('langgraph_enabled', False)}")
        
        # RAG 컴포넌트 상태 확인
        rag_components = status.get('rag_components', {})
        print("   RAG 컴포넌트 상태:")
        for component, available in rag_components.items():
            status_icon = "✓" if available else "✗"
            print(f"     {status_icon} {component}: {available}")
        
        # Phase 컴포넌트 상태 확인
        for phase in ['phase1_components', 'phase2_components', 'phase3_components']:
            if phase in status:
                print(f"   {phase}:")
                for component, available in status[phase].items():
                    status_icon = "✓" if available else "✗"
                    print(f"     {status_icon} {component}: {available}")
        
        # 간단한 메시지 처리 테스트
        print("4. 메시지 처리 테스트 중...")
        test_messages = [
            "안녕하세요",
            "계약서 검토에 대해 알려주세요",
            "법률 자문이 필요합니다"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"   테스트 {i}: '{message}'")
            try:
                result = await chat_service.process_message(message)
                
                if result and 'response' in result:
                    response = result['response']
                    confidence = result.get('confidence', 0.0)
                    processing_time = result.get('processing_time', 0.0)
                    
                    print(f"     ✓ 응답 생성 성공")
                    print(f"     응답 길이: {len(response)} 문자")
                    print(f"     신뢰도: {confidence:.2f}")
                    print(f"     처리 시간: {processing_time:.2f}초")
                    print(f"     응답 미리보기: {response[:100]}...")
                    
                    # Phase 정보 확인
                    phase_info = result.get('phase_info', {})
                    if phase_info:
                        print(f"     Phase 정보:")
                        for phase, info in phase_info.items():
                            enabled = info.get('enabled', False)
                            status_icon = "✓" if enabled else "✗"
                            print(f"       {status_icon} {phase}: {enabled}")
                else:
                    print(f"     ✗ 응답 생성 실패")
                    
            except Exception as e:
                print(f"     ✗ 오류 발생: {str(e)}")
            
            print()
        
        # 성능 메트릭 확인
        print("5. 성능 메트릭 확인 중...")
        try:
            metrics = chat_service.get_performance_metrics()
            print("   ✓ 성능 메트릭 조회 성공")
            if 'error' not in metrics:
                print(f"   메트릭 타임스탬프: {metrics.get('timestamp', 'N/A')}")
            else:
                print(f"   메트릭 오류: {metrics.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   ✗ 성능 메트릭 조회 실패: {str(e)}")
        
        print("=" * 60)
        print("ChatService 테스트 완료")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ ChatService 테스트 실패: {str(e)}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

def test_gradio_integration():
    """Gradio 통합 테스트"""
    print("=" * 60)
    print("Gradio 통합 테스트 시작")
    print("=" * 60)
    
    try:
        # Gradio 앱 임포트 테스트
        print("1. Gradio 앱 임포트 테스트 중...")
        from app_perfect_chatgpt import PerfectChatGPTStyleLawFirmAI
        print("   ✓ Gradio 앱 임포트 성공")
        
        # 앱 인스턴스 생성 테스트
        print("2. 앱 인스턴스 생성 테스트 중...")
        app = PerfectChatGPTStyleLawFirmAI()
        print("   ✓ 앱 인스턴스 생성 성공")
        
        # 초기화 상태 확인
        print("3. 앱 초기화 상태 확인 중...")
        if app.is_initialized:
            print("   ✓ 앱 초기화 완료")
        else:
            print("   ✗ 앱 초기화 실패")
            return False
        
        # ChatService 상태 확인
        print("4. 내장 ChatService 상태 확인 중...")
        if app.chat_service:
            print("   ✓ ChatService 인스턴스 존재")
            
            # 간단한 쿼리 처리 테스트
            print("5. 쿼리 처리 테스트 중...")
            test_query = "테스트 질문입니다"
            result = app.process_query(test_query)
            
            if result and 'answer' in result:
                print("   ✓ 쿼리 처리 성공")
                print(f"   응답: {result['answer'][:100]}...")
                print(f"   신뢰도: {result.get('confidence', 0.0):.2f}")
                print(f"   처리 시간: {result.get('processing_time', 0.0):.2f}초")
            else:
                print("   ✗ 쿼리 처리 실패")
                return False
        else:
            print("   ✗ ChatService 인스턴스 없음")
            return False
        
        print("=" * 60)
        print("Gradio 통합 테스트 완료")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio 통합 테스트 실패: {str(e)}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def main():
    """메인 테스트 함수"""
    print("LawFirmAI ChatService 테스트 시작")
    print("=" * 80)
    
    # ChatService 기본 테스트
    chat_service_success = await test_chat_service()
    
    print("\n")
    
    # Gradio 통합 테스트
    gradio_success = test_gradio_integration()
    
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    print(f"ChatService 기본 테스트: {'✓ 성공' if chat_service_success else '✗ 실패'}")
    print(f"Gradio 통합 테스트: {'✓ 성공' if gradio_success else '✗ 실패'}")
    
    if chat_service_success and gradio_success:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("Gradio에서 chat_service를 사용한 대화가 정상적으로 작동합니다.")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")
        print("문제를 해결한 후 다시 테스트해주세요.")
    
    return chat_service_success and gradio_success

if __name__ == "__main__":
    # 비동기 함수 실행
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
