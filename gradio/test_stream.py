#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
스트림 기능 테스트 스크립트
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.utils.config import Config

async def test_stream_functionality():
    """스트림 기능 테스트"""
    print("🚀 스트림 기능 테스트 시작...")
    
    try:
        # ChatService 초기화
        config = Config()
        chat_service = ChatService(config)
        
        print("✅ ChatService 초기화 완료")
        
        # 테스트 메시지
        test_message = "계약 해제 조건이 무엇인가요?"
        print(f"📝 테스트 메시지: {test_message}")
        
        # 스트림 처리 테스트
        print("\n🔄 스트림 처리 시작...")
        chunk_count = 0
        
        async for chunk in chat_service.process_message_stream(
            test_message,
            session_id="test_session",
            user_id="test_user"
        ):
            chunk_count += 1
            chunk_type = chunk.get("type", "unknown")
            content = chunk.get("content", "")
            timestamp = chunk.get("timestamp", "")
            
            print(f"📦 청크 #{chunk_count}:")
            print(f"   타입: {chunk_type}")
            print(f"   내용: {content}")
            print(f"   시간: {timestamp}")
            print("   " + "-" * 50)
            
            # 최대 10개 청크만 테스트
            if chunk_count >= 10:
                print("⚠️  최대 청크 수에 도달했습니다.")
                break
        
        print(f"\n✅ 스트림 처리 완료! 총 {chunk_count}개 청크 처리됨")
        
        # 일반 처리와 비교
        print("\n🔄 일반 처리 테스트...")
        result = await chat_service.process_message(
            test_message,
            session_id="test_session",
            user_id="test_user"
        )
        
        print(f"📊 일반 처리 결과:")
        print(f"   응답: {result.get('response', '')[:100]}...")
        print(f"   신뢰도: {result.get('confidence', 0):.2f}")
        print(f"   처리 시간: {result.get('processing_time', 0):.2f}초")
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_gradio_stream():
    """Gradio 스트림 기능 테스트"""
    print("🚀 Gradio 스트림 기능 테스트 시작...")
    
    try:
        # Gradio 앱 임포트
        from app_final_production import ProductionLawFirmAI
        
        # 앱 인스턴스 생성
        app = ProductionLawFirmAI()
        
        print("✅ ProductionLawFirmAI 초기화 완료")
        
        # 테스트 메시지
        test_message = "손해배상 관련 판례를 찾아주세요"
        print(f"📝 테스트 메시지: {test_message}")
        
        # 스트림 처리 테스트
        print("\n🔄 스트림 처리 시작...")
        
        async def test_stream():
            chunk_count = 0
            async for chunk in app.process_query_stream(test_message):
                chunk_count += 1
                chunk_type = chunk.get("type", "unknown")
                content = chunk.get("content", "")
                
                print(f"📦 청크 #{chunk_count}:")
                print(f"   타입: {chunk_type}")
                print(f"   내용: {content}")
                print("   " + "-" * 50)
                
                if chunk_count >= 5:
                    break
            
            print(f"\n✅ 스트림 처리 완료! 총 {chunk_count}개 청크 처리됨")
        
        # 비동기 테스트 실행
        asyncio.run(test_stream())
        
        print("\n🎉 Gradio 스트림 테스트 완료!")
        
    except Exception as e:
        print(f"❌ Gradio 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=" * 60)
    print("🔬 LawFirmAI 스트림 기능 테스트")
    print("=" * 60)
    
    # ChatService 스트림 테스트
    print("\n1️⃣ ChatService 스트림 기능 테스트")
    asyncio.run(test_stream_functionality())
    
    print("\n" + "=" * 60)
    
    # Gradio 스트림 테스트
    print("\n2️⃣ Gradio 스트림 기능 테스트")
    test_gradio_stream()
    
    print("\n" + "=" * 60)
    print("🏁 모든 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
