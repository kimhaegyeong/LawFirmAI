# -*- coding: utf-8 -*-
"""
간단한 LangGraph 통합 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_langgraph_integration():
    """LangGraph 통합 테스트"""
    print("🚀 LangGraph 통합 테스트 시작")
    
    try:
        # Enhanced Chat Service 테스트
        from source.utils.config import Config
        from source.services.enhanced_chat_service import EnhancedChatService
        
        print("✅ Enhanced Chat Service import 성공")
        
        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")
        
        # Enhanced Chat Service 초기화
        chat_service = EnhancedChatService(config)
        print("✅ Enhanced Chat Service 초기화 성공")
        
        # LangGraph 활성화 상태 확인
        if hasattr(chat_service, 'use_langgraph'):
            print(f"📊 LangGraph 활성화 상태: {chat_service.use_langgraph}")
        else:
            print("⚠️ LangGraph 속성이 없습니다")
        
        # LangGraph 서비스 상태 확인
        if hasattr(chat_service, 'langgraph_service'):
            if chat_service.langgraph_service:
                print("✅ LangGraph 서비스가 초기화되었습니다")
            else:
                print("⚠️ LangGraph 서비스가 초기화되지 않았습니다")
        else:
            print("⚠️ LangGraph 서비스 속성이 없습니다")
        
        print("\n🎉 LangGraph 통합 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_langgraph_integration()
    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 테스트가 실패했습니다.")
        sys.exit(1)

