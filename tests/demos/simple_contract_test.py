#!/usr/bin/env python3
"""
Simple Interactive Contract System Test
간단한 대화형 계약서 시스템 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_contract_detection():
    """계약서 관련 질문 감지 테스트"""
    print("=== 계약서 관련 질문 감지 테스트 ===")
    
    # ContractQueryHandler를 직접 테스트
    from source.services.contract_query_handler import ContractQueryHandler
    
    handler = ContractQueryHandler(None, None)
    
    test_messages = [
        "계약서 작성 방법을 알려주세요",
        "용역계약서를 어떻게 만들까요?",
        "근로계약서 템플릿이 필요해요",
        "부동산 매매계약서 작성 가이드를 원합니다",
        "민법 제750조가 뭐야?",  # 계약서 관련이 아닌 질문
        "일반적인 법률 질문입니다"  # 계약서 관련이 아닌 질문
    ]
    
    for message in test_messages:
        is_contract = handler.is_contract_related_query(message)
        print(f"질문: '{message}' -> 계약서 관련: {is_contract}")

def test_contract_assistant():
    """계약서 어시스턴트 기본 테스트"""
    print("\n=== 계약서 어시스턴트 기본 테스트 ===")
    
    from source.services.interactive_contract_assistant import InteractiveContractAssistant, ContractType
    
    assistant = InteractiveContractAssistant()
    
    # 계약 유형 테스트
    print(f"지원하는 계약 유형: {[ct.value for ct in ContractType]}")
    
    # 질문 템플릿 테스트
    print(f"질문 템플릿 개수: {len(assistant.question_templates)}")
    print(f"질문 템플릿 키: {list(assistant.question_templates.keys())}")

def test_contract_information():
    """계약 정보 클래스 테스트"""
    print("\n=== 계약 정보 클래스 테스트 ===")
    
    from source.services.interactive_contract_assistant import ContractInformation, ContractType
    
    # 계약 정보 생성
    contract_info = ContractInformation()
    contract_info.contract_type = ContractType.SERVICE
    contract_info.parties = {
        "client": "(주)ABC회사, 대표: 홍길동",
        "contractor": "프리랜서 디자이너 김철수"
    }
    contract_info.purpose = "웹사이트 디자인 작업"
    contract_info.scope = "메인페이지 1개, 상품페이지 5개"
    contract_info.payment_amount = "500만원"
    contract_info.timeline = "2개월"
    
    print(f"계약 유형: {contract_info.contract_type.value}")
    print(f"계약 목적: {contract_info.purpose}")
    print(f"작업 범위: {contract_info.scope}")
    print(f"계약 금액: {contract_info.payment_amount}")
    print(f"작업 기간: {contract_info.timeline}")

def main():
    """메인 함수"""
    print("🚀 대화형 계약서 시스템 기본 테스트 시작")
    
    try:
        test_contract_detection()
        test_contract_assistant()
        test_contract_information()
        
        print("\n✅ 모든 기본 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
