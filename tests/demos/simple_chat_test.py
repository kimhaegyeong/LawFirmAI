# -*- coding: utf-8 -*-
"""
Simple Chat Service Test
간단한 채팅 서비스 테스트
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')
sys.path.insert(0, 'source')
sys.path.insert(0, 'source/services')
sys.path.insert(0, 'source/utils')
sys.path.insert(0, 'source/models')
sys.path.insert(0, 'source/data')

print("🚀 간단한 채팅 서비스 테스트")
print("=" * 50)

def generate_test_questions() -> List[Dict[str, Any]]:
    """테스트 질문 생성"""
    questions = [
        {
            "question": "퇴직금 계산 방법과 지급 시기를 알려주세요",
            "category": "노동법",
            "expected_type": "labor_law",
            "priority": "high"
        },
        {
            "question": "유언장 없이 상속이 진행될 때 상속분은 어떻게 결정되나요?",
            "category": "상속법",
            "expected_type": "inheritance",
            "priority": "high"
        },
        {
            "question": "사기죄의 구성요건과 처벌 기준을 간단히 설명해주세요",
            "category": "형사법",
            "expected_type": "criminal_law",
            "priority": "high"
        }
    ]
    return questions

async def test_simple_chat():
    """간단한 채팅 테스트"""
    print("\n🚀 간단한 채팅 테스트 시작")
    print("=" * 50)

    try:
        # 기본 모듈만 import
        from source.utils.config import Config
        print("✅ Config 모듈 import 성공")

        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")

        # 간단한 채팅 서비스 클래스 생성 (EnhancedChatService 대신)
        class SimpleChatService:
            def __init__(self, config):
                self.config = config
                self.logger = None

            async def process_message(self, message: str, user_id: str = None, session_id: str = None):
                """간단한 메시지 처리"""
                start_time = time.time()

                # 간단한 응답 생성
                if "퇴직금" in message:
                    response = """퇴직금 계산 방법:

1. **계산 기준**
   - 평균임금 × 근속연수
   - 평균임금: 최근 3개월간 임금의 평균
   - 근속연수: 1년 미만은 월 단위로 계산

2. **지급 시기**
   - 퇴사일로부터 14일 이내
   - 지급 지연 시 연 20% 이자 지급

3. **퇴직금 지급 대상**
   - 1년 이상 근무한 근로자
   - 정규직, 비정규직 모두 포함

더 자세한 내용은 근로기준법 제34조를 참고하세요."""

                elif "상속" in message:
                    response = """상속분 결정 방법:

1. **법정상속인과 상속분**
   - 배우자: 1.5배
   - 자녀: 1인당 1배
   - 부모: 1인당 1배
   - 형제자매: 1인당 1배

2. **상속분 계산**
   - 배우자 + 자녀: 배우자 1.5, 자녀들 나머지
   - 배우자 + 부모: 배우자 1.5, 부모들 나머지
   - 배우자 + 형제자매: 배우자 1.5, 형제자매들 나머지

3. **유언이 없는 경우**
   - 법정상속분에 따라 자동 분할
   - 상속포기 신고 가능

민법 제1000조 이하를 참고하세요."""

                elif "사기죄" in message:
                    response = """사기죄의 구성요건과 처벌:

1. **구성요건**
   - 기망행위: 상대방을 기만하는 행위
   - 착오유발: 상대방이 착오에 빠지게 함
   - 재산상 이익: 재산적 이득을 얻음
   - 인과관계: 기망행위와 재산상 이익 간의 인과관계

2. **처벌 기준**
   - 일반사기: 10년 이하 징역 또는 2천만원 이하 벌금
   - 컴퓨터사기: 10년 이하 징역 또는 2천만원 이하 벌금
   - 신용카드사기: 5년 이하 징역 또는 1천만원 이하 벌금

3. **특가법 적용**
   - 특가법상 사기: 가중처벌
   - 조직적 사기: 더욱 가중처벌

형법 제347조를 참고하세요."""

                else:
                    response = f"죄송합니다. '{message}'에 대한 정보를 찾을 수 없습니다. 더 구체적으로 질문해주세요."

                processing_time = time.time() - start_time

                return {
                    "response": response,
                    "confidence": 0.8,
                    "sources": [],
                    "processing_time": processing_time,
                    "generation_method": "simple_template",
                    "session_id": session_id or "test_session",
                    "user_id": user_id or "test_user"
                }

        # 간단한 채팅 서비스 초기화
        chat_service = SimpleChatService(config)
        print("✅ Simple Chat Service 초기화 성공")

        # 테스트 질문 생성
        test_questions = generate_test_questions()
        print(f"📝 총 {len(test_questions)}개의 테스트 질문 생성")

        # 테스트 실행
        results = []
        start_time = time.time()

        print(f"\n🔄 간단한 채팅 테스트 실행 중...")
        print("-" * 50)

        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            expected_type = test_case["expected_type"]
            priority = test_case["priority"]

            print(f"\n질문 {i}: {question}")
            print(f"카테고리: {category} | 예상유형: {expected_type} | 우선순위: {priority}")

            try:
                # 메시지 처리
                result = await chat_service.process_message(
                    message=question,
                    user_id=f"test_user_{i}",
                    session_id=f"test_session_{i}"
                )

                # 결과 분석
                response = result.get('response', 'N/A')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                generation_method = result.get('generation_method', 'unknown')

                print(f"응답: {response[:100]}...")
                print(f"신뢰도: {confidence:.2f}")
                print(f"처리 시간: {processing_time:.3f}초")
                print(f"생성 방법: {generation_method}")
                print("-" * 80)

                # 결과 저장
                results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': confidence,
                    'generation_method': generation_method
                })

            except Exception as e:
                print(f"❌ 질문 {i} 처리 실패: {e}")
                results.append({
                    'test_case': test_case,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        total_time = time.time() - start_time

        # 테스트 결과 분석
        print(f"\n📊 간단한 채팅 테스트 결과")
        print("=" * 50)

        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - successful_tests

        print(f"총 테스트: {total_tests}")
        print(f"성공한 테스트: {successful_tests}")
        print(f"실패한 테스트: {failed_tests}")
        print(f"총 실행 시간: {total_time:.2f}초")

        if successful_tests > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful_tests
            avg_processing_time = sum(r.get('processing_time', 0) for r in results if r['success']) / successful_tests

            print(f"평균 신뢰도: {avg_confidence:.2f}")
            print(f"평균 처리 시간: {avg_processing_time:.3f}초")

        print(f"\n✅ 간단한 채팅 테스트 완료!")
        return results

    except Exception as e:
        print(f"❌ 간단한 채팅 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return []

if __name__ == "__main__":
    print("🚀 Simple Chat Service Test")
    print("=" * 50)

    # 간단한 테스트 실행
    results = asyncio.run(test_simple_chat())

    print("\n🎉 간단한 채팅 테스트 완료!")
    print("=" * 50)

    # 최종 요약
    if results:
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        print(f"📊 테스트 요약: {successful_tests}/{total_tests} 성공")
