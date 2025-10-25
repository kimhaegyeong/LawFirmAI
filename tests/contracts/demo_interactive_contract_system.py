#!/usr/bin/env python3
"""
Interactive Contract System Demo
대화형 계약서 시스템 데모
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.interactive_contract_assistant import InteractiveContractAssistant


async def demo_interactive_contract_system():
    """대화형 계약서 시스템 데모"""
    print("🎯 대화형 계약서 작성 시스템 데모")
    print("=" * 50)
    
    # 계약서 어시스턴트 초기화
    assistant = InteractiveContractAssistant()
    
    # 데모 시나리오
    demo_scenarios = [
        {
            "step": 1,
            "user_input": "계약서 작성 방법을 알려주세요",
            "description": "초기 질문 - 계약서 작성 도움 요청"
        },
        {
            "step": 2,
            "user_input": "웹사이트 디자인 용역계약이요",
            "description": "계약 유형 선택 - 용역계약"
        },
        {
            "step": 3,
            "user_input": "갑: (주)ABC회사, 대표: 홍길동, 주소: 서울시 강남구\n을: 프리랜서 디자이너 김철수, 주소: 경기도 성남시",
            "description": "계약 당사자 정보 제공"
        },
        {
            "step": 4,
            "user_input": "메인페이지 1개, 상품페이지 5개, 관리자 페이지 1개 디자인",
            "description": "작업 범위 구체화"
        },
        {
            "step": 5,
            "user_input": "500만원, 2개월",
            "description": "계약 금액 및 기간 제공"
        }
    ]
    
    session_id = "demo_session_001"
    user_id = "demo_user_001"
    
    for scenario in demo_scenarios:
        print(f"\n📝 단계 {scenario['step']}: {scenario['description']}")
        print(f"사용자: {scenario['user_input']}")
        print("-" * 30)
        
        # 계약서 처리
        result = await assistant.process_contract_query(
            scenario['user_input'], session_id, user_id
        )
        
        # 응답 출력
        print(f"시스템: {result['response']}")
        
        # 추가 정보 표시
        if 'questions' in result:
            print(f"\n💡 추가 질문 수: {len(result['questions'])}")
        
        if 'contract_generated' in result and result['contract_generated']:
            print("\n🎉 계약서 생성 완료!")
            print(f"📄 계약서 템플릿 길이: {len(result.get('contract_template', ''))} 문자")
            break
        
        print("\n" + "="*50)
    
    # 최종 결과 확인
    session_info = assistant.get_session_info(session_id)
    if session_info:
        print(f"\n📊 세션 정보:")
        print(f"- 수집된 필드: {session_info.collected_fields}")
        print(f"- 대화 상태: {session_info.conversation_state.value}")
        print(f"- 계약 유형: {session_info.contract_info.contract_type.value if session_info.contract_info.contract_type else '미정'}")


async def demo_contract_template_generation():
    """계약서 템플릿 생성 데모"""
    print("\n\n🏗️ 계약서 템플릿 생성 데모")
    print("=" * 50)
    
    from source.services.interactive_contract_assistant import ContractInformation, ContractType
    
    # 완성된 계약 정보 생성
    contract_info = ContractInformation()
    contract_info.contract_type = ContractType.SERVICE
    contract_info.parties = {
        "client": "(주)ABC회사, 대표: 홍길동, 주소: 서울시 강남구 테헤란로 123",
        "contractor": "프리랜서 디자이너 김철수, 주소: 경기도 성남시 분당구 정자역로 456"
    }
    contract_info.purpose = "웹사이트 디자인 작업"
    contract_info.scope = "메인페이지 1개, 상품페이지 5개, 관리자 페이지 1개"
    contract_info.payment_amount = "500만원"
    contract_info.timeline = "2개월 (2024년 1월 1일 ~ 2024년 3월 31일)"
    contract_info.payment_method = "계약금 30% + 중도금 40% + 잔금 30%"
    
    assistant = InteractiveContractAssistant()
    
    # 계약서 템플릿 생성
    template = await assistant._create_contract_template(contract_info)
    print("📄 생성된 계약서 템플릿:")
    print(template)
    
    # 법적 리스크 분석
    risk_analysis = await assistant._analyze_legal_risks(contract_info)
    print(f"\n⚠️ 법적 리스크 분석:")
    print(risk_analysis)
    
    # 권장 조항
    recommended_clauses = await assistant._generate_recommended_clauses(contract_info)
    print(f"\n💡 권장 조항:")
    print(recommended_clauses)


async def main():
    """메인 함수"""
    try:
        await demo_interactive_contract_system()
        await demo_contract_template_generation()
        
        print("\n\n✅ 대화형 계약서 시스템 데모 완료!")
        print("\n🎯 주요 기능:")
        print("- 계약서 관련 질문 자동 감지")
        print("- 단계별 정보 수집")
        print("- 맞춤형 계약서 템플릿 생성")
        print("- 법적 리스크 분석")
        print("- 권장 조항 제안")
        
    except Exception as e:
        print(f"❌ 데모 실행 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
