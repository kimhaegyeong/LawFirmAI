#!/usr/bin/env python3
"""
Interactive Contract System Test
대화형 계약서 시스템 테스트
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.interactive_contract_assistant import InteractiveContractAssistant
from source.services.contract_query_handler import ContractQueryHandler
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveContractSystemTest:
    """대화형 계약서 시스템 테스트"""
    
    def __init__(self):
        self.contract_assistant = InteractiveContractAssistant()
        self.contract_query_handler = ContractQueryHandler(
            self.contract_assistant, None
        )
    
    async def test_contract_detection(self):
        """계약서 관련 질문 감지 테스트"""
        logger.info("=== 계약서 관련 질문 감지 테스트 ===")
        
        test_messages = [
            "계약서 작성 방법을 알려주세요",
            "용역계약서를 어떻게 만들까요?",
            "근로계약서 템플릿이 필요해요",
            "부동산 매매계약서 작성 가이드를 원합니다",
            "민법 제750조가 뭐야?",  # 계약서 관련이 아닌 질문
            "일반적인 법률 질문입니다"  # 계약서 관련이 아닌 질문
        ]
        
        for message in test_messages:
            is_contract = self.contract_query_handler.is_contract_related_query(message)
            logger.info(f"질문: '{message}' -> 계약서 관련: {is_contract}")
    
    async def test_interactive_contract_flow(self):
        """대화형 계약서 작성 플로우 테스트"""
        logger.info("=== 대화형 계약서 작성 플로우 테스트 ===")
        
        session_id = "test_session_001"
        user_id = "test_user_001"
        
        # 1단계: 초기 질문
        logger.info("1단계: 초기 질문")
        response1 = await self.contract_assistant.process_contract_query(
            "계약서 작성 방법을 알려주세요", session_id, user_id
        )
        logger.info(f"응답1: {response1['response'][:200]}...")
        
        # 2단계: 계약 유형 선택
        logger.info("2단계: 계약 유형 선택")
        response2 = await self.contract_assistant.process_contract_query(
            "웹사이트 디자인 용역계약이요", session_id, user_id
        )
        logger.info(f"응답2: {response2['response'][:200]}...")
        
        # 3단계: 당사자 정보 제공
        logger.info("3단계: 당사자 정보 제공")
        response3 = await self.contract_assistant.process_contract_query(
            "갑: (주)ABC회사, 대표: 홍길동, 주소: 서울시 강남구\n을: 프리랜서 디자이너 김철수, 주소: 경기도 성남시", 
            session_id, user_id
        )
        logger.info(f"응답3: {response3['response'][:200]}...")
        
        # 4단계: 작업 범위 제공
        logger.info("4단계: 작업 범위 제공")
        response4 = await self.contract_assistant.process_contract_query(
            "메인페이지 1개, 상품페이지 5개, 관리자 페이지 1개 디자인", 
            session_id, user_id
        )
        logger.info(f"응답4: {response4['response'][:200]}...")
        
        # 5단계: 금액 및 기간 제공
        logger.info("5단계: 금액 및 기간 제공")
        response5 = await self.contract_assistant.process_contract_query(
            "500만원, 2개월", session_id, user_id
        )
        logger.info(f"응답5: {response5['response'][:200]}...")
        
        # 최종 결과 확인
        if response5.get("contract_generated"):
            logger.info("✅ 계약서 생성 성공!")
            logger.info(f"생성된 계약서 길이: {len(response5.get('contract_template', ''))}")
        else:
            logger.warning("⚠️ 계약서 생성 실패 또는 추가 정보 필요")
    
    async def test_contract_template_generation(self):
        """계약서 템플릿 생성 테스트"""
        logger.info("=== 계약서 템플릿 생성 테스트 ===")
        
        from source.services.interactive_contract_assistant import ContractInformation, ContractType
        
        # 테스트용 계약 정보 생성
        contract_info = ContractInformation()
        contract_info.contract_type = ContractType.SERVICE
        contract_info.parties = {
            "client": "(주)ABC회사, 대표: 홍길동, 주소: 서울시 강남구",
            "contractor": "프리랜서 디자이너 김철수, 주소: 경기도 성남시"
        }
        contract_info.purpose = "웹사이트 디자인 작업"
        contract_info.scope = "메인페이지 1개, 상품페이지 5개, 관리자 페이지 1개"
        contract_info.payment_amount = "500만원"
        contract_info.timeline = "2개월"
        
        # 계약서 템플릿 생성
        template = await self.contract_assistant._create_contract_template(contract_info)
        logger.info("생성된 계약서 템플릿:")
        logger.info(template)
        
        # 법적 리스크 분석
        risk_analysis = await self.contract_assistant._analyze_legal_risks(contract_info)
        logger.info("법적 리스크 분석:")
        logger.info(risk_analysis)
        
        # 권장 조항
        recommended_clauses = await self.contract_assistant._generate_recommended_clauses(contract_info)
        logger.info("권장 조항:")
        logger.info(recommended_clauses)
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🚀 대화형 계약서 시스템 테스트 시작")
        
        try:
            await self.test_contract_detection()
            await self.test_interactive_contract_flow()
            await self.test_contract_template_generation()
            
            logger.info("✅ 모든 테스트 완료!")
            
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {e}")
            raise


async def main():
    """메인 함수"""
    test = InteractiveContractSystemTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
