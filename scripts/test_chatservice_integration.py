# -*- coding: utf-8 -*-
"""
ChatService Integration Test
ChatService와 법률 제한 시스템 통합 테스트
"""

import sys
import os
import asyncio
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.chat_service import ChatService
from source.utils.config import Config


class ChatServiceIntegrationTest:
    """ChatService 통합 테스트"""
    
    def __init__(self):
        print("🚀 ChatService 통합 테스트 초기화 중...")
        
        try:
            config = Config()
            self.chat_service = ChatService(config)
            print("✅ ChatService가 성공적으로 초기화되었습니다!")
            
            # 법률 제한 시스템이 초기화되었는지 확인
            if hasattr(self.chat_service, 'legal_restriction_system') and self.chat_service.legal_restriction_system:
                print("✅ 법률 제한 시스템이 통합되었습니다!")
            else:
                print("❌ 법률 제한 시스템이 통합되지 않았습니다!")
                
        except Exception as e:
            print(f"❌ ChatService 초기화 실패: {e}")
            self.chat_service = None
    
    async def test_restricted_queries(self):
        """제한된 질문 테스트"""
        print("\n🔍 제한된 질문 테스트")
        print("=" * 50)
        
        test_queries = [
            "제 경우 소송하시겠습니까?",
            "의료사고의 과실이 있나요?",
            "자백하시겠습니까?",
            "세금을 회피하는 방법은?",
            "일반적인 계약서 작성 방법은?"
        ]
        
        for query in test_queries:
            print(f"\n📝 질문: {query}")
            
            try:
                result = await self.chat_service.process_message(
                    message=query,
                    user_id="test_user",
                    session_id="test_session"
                )
                
                # 제한 정보 확인
                if "restriction_info" in result:
                    restriction_info = result["restriction_info"]
                    print(f"   🚫 제한됨: {'예' if restriction_info.get('is_restricted') else '아니오'}")
                    if restriction_info.get('is_restricted'):
                        print(f"   📊 제한 수준: {restriction_info.get('restriction_level', 'unknown')}")
                        print(f"   ⚠️  경고 메시지: {restriction_info.get('warning_message', 'None')}")
                        print(f"   📋 면책 조항: {restriction_info.get('disclaimer', 'None')}")
                
                # 검증 정보 확인
                if "validation_info" in result:
                    validation_info = result["validation_info"]
                    print(f"   ✅ 검증 상태: {validation_info.get('status', 'unknown')}")
                    if validation_info.get('issues'):
                        print(f"   ⚠️  이슈: {', '.join(validation_info['issues'])}")
                
                print(f"   💬 답변: {result.get('response', 'No response')[:100]}...")
                print(f"   📈 신뢰도: {result.get('confidence', 0.0):.2f}")
                
            except Exception as e:
                print(f"   ❌ 오류: {e}")
    
    async def test_safe_queries(self):
        """안전한 질문 테스트"""
        print("\n✅ 안전한 질문 테스트")
        print("=" * 50)
        
        safe_queries = [
            "일반적인 계약서 작성 방법은?",
            "소송 제기 절차는 어떻게 되나요?",
            "관련 법령을 알려주세요",
            "판례를 찾아주세요",
            "법률구조공단은 어디에 있나요?"
        ]
        
        for query in safe_queries:
            print(f"\n📝 질문: {query}")
            
            try:
                result = await self.chat_service.process_message(
                    message=query,
                    user_id="test_user",
                    session_id="test_session"
                )
                
                # 제한 정보 확인
                if "restriction_info" in result:
                    restriction_info = result["restriction_info"]
                    print(f"   🚫 제한됨: {'예' if restriction_info.get('is_restricted') else '아니오'}")
                
                # 검증 정보 확인
                if "validation_info" in result:
                    validation_info = result["validation_info"]
                    print(f"   ✅ 검증 상태: {validation_info.get('status', 'unknown')}")
                
                print(f"   💬 답변: {result.get('response', 'No response')[:100]}...")
                print(f"   📈 신뢰도: {result.get('confidence', 0.0):.2f}")
                
            except Exception as e:
                print(f"   ❌ 오류: {e}")
    
    async def test_system_components(self):
        """시스템 컴포넌트 테스트"""
        print("\n🔧 시스템 컴포넌트 테스트")
        print("=" * 50)
        
        components = [
            ("법률 제한 시스템", self.chat_service.legal_restriction_system),
            ("콘텐츠 필터 엔진", self.chat_service.content_filter_engine),
            ("답변 검증 시스템", self.chat_service.response_validation_system),
            ("안전한 답변 생성기", self.chat_service.safe_response_generator),
            ("법적 준수 모니터", self.chat_service.legal_compliance_monitor),
            ("사용자 교육 시스템", self.chat_service.user_education_system)
        ]
        
        for component_name, component in components:
            if component:
                print(f"   ✅ {component_name}: 초기화됨")
            else:
                print(f"   ❌ {component_name}: 초기화되지 않음")
    
    async def run_test(self):
        """전체 테스트 실행"""
        if not self.chat_service:
            print("❌ ChatService가 초기화되지 않아 테스트를 실행할 수 없습니다.")
            return
        
        print("🎯 ChatService 통합 테스트 시작")
        print("=" * 60)
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 시스템 컴포넌트 테스트
            await self.test_system_components()
            
            # 제한된 질문 테스트
            await self.test_restricted_queries()
            
            # 안전한 질문 테스트
            await self.test_safe_queries()
            
            print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """메인 함수"""
    test = ChatServiceIntegrationTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
