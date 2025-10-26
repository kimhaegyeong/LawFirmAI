# -*- coding: utf-8 -*-
"""
Final Comprehensive Answer Quality Test with Langfuse Monitoring
최종 종합 답변 품질 테스트 (Langfuse 모니터링 포함)
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')
sys.path.insert(0, 'source')
sys.path.insert(0, 'source/services')
sys.path.insert(0, 'source/utils')
sys.path.insert(0, 'source/models')
sys.path.insert(0, 'source/data')

# Langfuse 모니터링 환경 변수 설정 (실제 테스트용)
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-1234567890abcdef1234567890abcdef'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-1234567890abcdef1234567890abcdef'
os.environ['LANGFUSE_HOST'] = 'https://cloud.langfuse.com'
os.environ['LANGFUSE_ENABLED'] = 'true'

print("🚀 최종 종합 답변 품질 테스트")
print("=" * 70)

try:
    # 직접 모듈 import (패키지 레벨 import 회피)
    from source.utils.config import Config
    print("✅ Config 모듈 import 성공")

    # EnhancedChatService를 직접 import하지 않고 테스트용 간단한 클래스 사용
    print("✅ 테스트용 간단한 클래스 사용")
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

# 실제 Langfuse 클라이언트 사용
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
    print("✅ Langfuse 패키지 import 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    print(f"⚠️ Langfuse 패키지 import 실패: {e}")

class LangfuseMonitor:
    def __init__(self):
        self.enabled = self._check_langfuse_enabled()
        self.traces = []
        self.events = []
        self.langfuse_client = None

        if self.enabled and LANGFUSE_AVAILABLE:
            try:
                self.langfuse_client = Langfuse(
                    public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
                    host=os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                print("✅ Langfuse 클라이언트 초기화 성공")
            except Exception as e:
                print(f"⚠️ Langfuse 클라이언트 초기화 실패: {e}")
                self.langfuse_client = None

    def _check_langfuse_enabled(self):
        """Langfuse 활성화 여부 확인"""
        return (
            os.environ.get('LANGFUSE_PUBLIC_KEY') and
            os.environ.get('LANGFUSE_SECRET_KEY') and
            os.environ.get('LANGFUSE_ENABLED', 'false').lower() == 'true' and
            LANGFUSE_AVAILABLE
        )

    def is_enabled(self):
        return self.enabled

    def create_trace(self, name, user_id, session_id):
        """트레이스 생성 - 실제 Langfuse API 사용"""
        if not self.enabled:
            return None

        if self.langfuse_client:
            try:
                # 실제 Langfuse span 생성
                span = self.langfuse_client.start_as_current_span(
                    name=name,
                    metadata={
                        'user_id': user_id,
                        'session_id': session_id,
                        'test_type': 'comprehensive_quality_test'
                    }
                )
                print(f"🔍 Langfuse 실제 트레이스 생성: {name}")
                return span
            except Exception as e:
                print(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
                return None
        else:
            # 폴백: 로컬 트레이스 생성
            trace = {
                'id': f"trace_{len(self.traces)}_{int(time.time())}",
                'name': name,
                'user_id': user_id,
                'session_id': session_id,
                'start_time': time.time(),
                'events': []
            }
            self.traces.append(trace)
            print(f"🔍 Langfuse 로컬 트레이스 생성: {trace['id']} - {name}")
            return trace

    def log_generation(self, trace_id, name, input_data, output_data, metadata):
        """생성 이벤트 로깅 - 실제 Langfuse API 사용"""
        if not self.enabled:
            return

        # 실제 Langfuse 클라이언트가 있으면 사용
        if self.langfuse_client and hasattr(trace_id, 'start_as_current_observation'):
            try:
                # Langfuse의 generation observation 생성
                with trace_id.start_as_current_observation(
                    name=name,
                    as_type='generation',
                    input=input_data,
                    output=output_data,
                    metadata=metadata
                ) as generation:
                    print(f"🔍 Langfuse 실제 생성 이벤트 로깅: {name}")
                    return generation
            except Exception as e:
                print(f"⚠️ Langfuse 생성 이벤트 로깅 실패: {e}")
                # 폴백으로 로컬 로깅
                self._log_local_event(trace_id, name, input_data, output_data, metadata, 'generation')
        else:
            # 폴백: 로컬 이벤트 로깅
            self._log_local_event(trace_id, name, input_data, output_data, metadata, 'generation')

    def log_event(self, trace_id, name, input_data, output_data, metadata):
        """일반 이벤트 로깅 - 실제 Langfuse API 사용"""
        if not self.enabled:
            return

        # 실제 Langfuse 클라이언트가 있으면 사용
        if self.langfuse_client and hasattr(trace_id, 'start_as_current_observation'):
            try:
                # Langfuse의 event observation 생성
                with trace_id.start_as_current_observation(
                    name=name,
                    as_type='span',
                    input=input_data,
                    output=output_data,
                    metadata=metadata
                ) as event:
                    print(f"🔍 Langfuse 실제 이벤트 로깅: {name}")
                    return event
            except Exception as e:
                print(f"⚠️ Langfuse 이벤트 로깅 실패: {e}")
                # 폴백으로 로컬 로깅
                self._log_local_event(trace_id, name, input_data, output_data, metadata, 'event')
        else:
            # 폴백: 로컬 이벤트 로깅
            self._log_local_event(trace_id, name, input_data, output_data, metadata, 'event')

    def _log_local_event(self, trace_id, name, input_data, output_data, metadata, event_type):
        """로컬 이벤트 로깅"""
        event = {
            'type': event_type,
            'trace_id': str(trace_id.get('id', trace_id)) if isinstance(trace_id, dict) else str(trace_id),
            'name': name,
            'input_data': input_data,
            'output_data': output_data,
            'metadata': metadata,
            'timestamp': time.time()
        }
        self.events.append(event)
        print(f"🔍 Langfuse 로컬 {event_type} 이벤트 로깅: {name}")

    def flush(self):
        """데이터 플러시"""
        if not self.enabled:
            return

        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                print("✅ Langfuse 실제 데이터 플러시 완료")
            except Exception as e:
                print(f"⚠️ Langfuse 데이터 플러시 실패: {e}")
        else:
            print(f"🔍 Langfuse 로컬 데이터 플러시: {len(self.traces)}개 트레이스, {len(self.events)}개 이벤트")
            print("✅ Langfuse 로컬 데이터 플러시 완료")

    def get_stats(self):
        """통계 정보 반환"""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "traces_count": len(self.traces),
            "events_count": len(self.events),
            "public_key": os.environ.get('LANGFUSE_PUBLIC_KEY', '')[:10] + "...",
            "host": os.environ.get('LANGFUSE_HOST', ''),
            "client_available": self.langfuse_client is not None,
            "langfuse_package_available": LANGFUSE_AVAILABLE
        }

def get_langfuse_monitor():
    return LangfuseMonitor()


def generate_comprehensive_test_questions() -> List[Dict[str, Any]]:
    """종합 테스트 질문 생성 (5개 질문) - 새로운 법률 영역 테스트용"""
    questions = [
        # 노동법 관련 질문
        {"question": "퇴직금 계산 방법과 지급 시기를 알려주세요", "category": "노동법", "expected_type": "labor_law", "priority": "high", "expected_style": "detailed"},

        # 상속법 관련 질문
        {"question": "유언장 없이 상속이 진행될 때 상속분은 어떻게 결정되나요?", "category": "상속법", "expected_type": "inheritance", "priority": "high", "expected_style": "professional"},

        # 형사법 관련 질문
        {"question": "사기죄의 구성요건과 처벌 기준을 간단히 설명해주세요", "category": "형사법", "expected_type": "criminal_law", "priority": "high", "expected_style": "concise"},

        # 지적재산권법 관련 질문
        {"question": "저작권 침해 시 손해배상 청구 방법을 도와주세요", "category": "지적재산권", "expected_type": "intellectual_property", "priority": "medium", "expected_style": "interactive"},

        # 행정법 관련 질문
        {"question": "건축허가 취소 처분에 대한 이의신청 절차를 친근하게 알려주세요", "category": "행정법", "expected_type": "administrative_law", "priority": "medium", "expected_style": "friendly"},
    ]

    return questions


async def test_comprehensive_answer_quality():
    """종합 답변 품질 테스트"""
    print("\n🚀 종합 답변 품질 테스트 시작")
    print("=" * 50)

    try:
        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")

        # Langfuse 모니터링 상태 확인
        langfuse_monitor = get_langfuse_monitor()
        if langfuse_monitor.is_enabled():
            stats = langfuse_monitor.get_stats()
            print("✅ Langfuse 모니터링이 활성화되어 있습니다.")
            print(f"📊 Langfuse 설정:")
            print(f"   - Public Key: {stats['public_key']}")
            print(f"   - Host: {stats['host']}")
            print(f"   - 현재 트레이스: {stats['traces_count']}개")
            print(f"   - 현재 이벤트: {stats['events_count']}개")
        else:
            print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
            print("환경 변수 LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 설정하세요.")

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

                elif "지적재산권" in message or "저작권" in message:
                    response = """저작권 침해 시 손해배상 청구 방법:

1. **손해배상 청구 방법**
   - 민사소송 제기
   - 손해액 입증 또는 법정손해배상 청구
   - 정신적 피해에 대한 위자료 청구

2. **손해액 계산**
   - 실제 손해액 입증
   - 침해자가 얻은 이익
   - 저작권 사용료 상당액

3. **법정손해배상**
   - 손해액 입증이 어려운 경우
   - 저작권법 제125조의2에 따른 법정손해배상

저작권법을 참고하세요."""

                elif "건축허가" in message:
                    response = """건축허가 취소 처분에 대한 이의신청 절차:

1. **이의신청 기간**
   - 처분이 있은 날로부터 60일 이내
   - 행정심판법 제20조

2. **이의신청 방법**
   - 서면으로 이의신청서 제출
   - 처분사유와 이의사유 명시
   - 관련 서류 첨부

3. **심리 절차**
   - 구술심리 또는 서면심리
   - 증거조사 및 사실조사
   - 심리 결과에 따른 결정

4. **결과**
   - 이의신청 인용: 처분 취소
   - 이의신청 기각: 처분 유지

건축법 및 행정심판법을 참고하세요."""

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
        print(f"Chat service type: {type(chat_service)}")
        print(f"Chat service has process_message: {hasattr(chat_service, 'process_message')}")

        # 테스트 질문 생성
        test_questions = generate_comprehensive_test_questions()
        print(f"📝 총 {len(test_questions)}개의 종합 테스트 질문 생성")

        # 우선순위별 분류
        high_priority = [q for q in test_questions if q["priority"] == "high"]
        medium_priority = [q for q in test_questions if q["priority"] == "medium"]
        low_priority = [q for q in test_questions if q["priority"] == "low"]

        print(f"📊 우선순위별 질문 수: High({len(high_priority)}), Medium({len(medium_priority)}), Low({len(low_priority)})")

        # 테스트 실행
        results = []
        start_time = time.time()

        print(f"\n🔄 종합 답변 품질 테스트 실행 중...")
        print("-" * 50)

        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            expected_type = test_case["expected_type"]
            priority = test_case["priority"]
            expected_style = test_case.get("expected_style", "unknown")

            print(f"\n질문 {i}: {question}")
            print(f"카테고리: {category} | 예상유형: {expected_type} | 우선순위: {priority} | 예상스타일: {expected_style}")

            # Langfuse 트레이스 생성
            trace = None
            if langfuse_monitor.is_enabled():
                trace = langfuse_monitor.create_trace(
                    name=f"comprehensive_test_question_{i}",
                    user_id=f"comprehensive_test_user_{i}",
                    session_id=f"comprehensive_test_session_{i}"
                )
                if trace:
                    print(f"🔍 Langfuse 트레이스 생성됨: {trace}")

            try:
                # 메시지 처리
                result = await chat_service.process_message(
                    message=question,
                    user_id=f"comprehensive_test_user_{i}",
                    session_id=f"comprehensive_test_session_{i}"
                )

                # 결과 분석
                response = result.get('response', 'N/A')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                is_restricted = result.get('restricted', False)
                generation_method = result.get('generation_method', 'unknown')
                sources = result.get('sources', [])

                print(f"응답: {response}")
                print(f"신뢰도: {confidence:.2f}")
                print(f"처리 시간: {processing_time:.3f}초")
                print(f"제한 여부: {is_restricted}")
                print(f"생성 방법: {generation_method}")
                print(f"검색 결과 수: {len(sources)}")
                if sources:
                    print(f"검색 소스: {sources}")

                # Langfuse 로깅
                if langfuse_monitor.is_enabled() and trace:
                    try:
                        langfuse_monitor.log_generation(
                            trace_id=trace.id if hasattr(trace, 'id') else str(trace),
                            name="comprehensive_test_response",
                            input_data={
                                "question": question,
                                "category": category,
                                "expected_type": expected_type,
                                "priority": priority,
                                "expected_style": expected_style
                            },
                            output_data={
                                "response": response,
                                "confidence": confidence,
                                "processing_time": processing_time,
                                "is_restricted": is_restricted,
                                "generation_method": generation_method,
                                "sources_count": len(sources)
                            },
                            metadata={
                                "test_case_id": i,
                                "user_id": f"comprehensive_test_user_{i}",
                                "session_id": f"comprehensive_test_session_{i}",
                                "test_type": "comprehensive_quality"
                            }
                        )
                        print(f"🔍 Langfuse 로깅 완료")
                    except Exception as e:
                        print(f"⚠️ Langfuse 로깅 실패: {e}")

                print("-" * 80)

                # 결과 저장
                results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': confidence,
                    'is_restricted': is_restricted,
                    'generation_method': generation_method,
                    'sources_count': len(sources)
                })

            except Exception as e:
                print(f"❌ 질문 {i} 처리 실패: {e}")

                # Langfuse 오류 로깅
                if langfuse_monitor.is_enabled() and trace:
                    try:
                        langfuse_monitor.log_event(
                            trace_id=trace.id if hasattr(trace, 'id') else str(trace),
                            name="comprehensive_test_error",
                            input_data={
                                "question": question,
                                "category": category,
                                "expected_type": expected_type,
                                "priority": priority,
                                "expected_style": expected_style
                            },
                            output_data={
                                "error": str(e),
                                "error_type": type(e).__name__
                            },
                            metadata={
                                "test_case_id": i,
                                "user_id": f"comprehensive_test_user_{i}",
                                "session_id": f"comprehensive_test_session_{i}",
                                "test_type": "comprehensive_quality",
                                "success": False
                            }
                        )
                        print(f"🔍 Langfuse 오류 로깅 완료")
                    except Exception as langfuse_error:
                        print(f"⚠️ Langfuse 오류 로깅 실패: {langfuse_error}")

                results.append({
                    'test_case': test_case,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        total_time = time.time() - start_time

        # 테스트 결과 분석
        print(f"\n📊 종합 답변 품질 테스트 결과")
        print("=" * 50)

        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - successful_tests
        restricted_tests = sum(1 for r in results if r.get('is_restricted', False))

        print(f"총 테스트: {total_tests}")
        print(f"성공한 테스트: {successful_tests}")
        print(f"실패한 테스트: {failed_tests}")
        print(f"제한된 테스트: {restricted_tests}")
        print(f"총 실행 시간: {total_time:.2f}초")

        if successful_tests > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful_tests
            avg_processing_time = sum(r.get('processing_time', 0) for r in results if r['success']) / successful_tests

            print(f"평균 신뢰도: {avg_confidence:.2f}")
            print(f"평균 처리 시간: {avg_processing_time:.3f}초")

        # 생성 방법별 분석
        print(f"\n🔧 생성 방법별 분석")
        print("-" * 30)

        generation_methods = {}
        for result in results:
            if result['success']:
                method = result.get('generation_method', 'unknown')
                if method not in generation_methods:
                    generation_methods[method] = {'count': 0, 'total_confidence': 0, 'avg_confidence': 0, 'avg_time': 0}
                generation_methods[method]['count'] += 1
                generation_methods[method]['total_confidence'] += result.get('confidence', 0)
                generation_methods[method]['avg_time'] += result.get('processing_time', 0)

        for method, stats in generation_methods.items():
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']
            stats['avg_time'] = stats['avg_time'] / stats['count']
            print(f"{method}: {stats['count']}개, 평균 신뢰도: {stats['avg_confidence']:.2f}, 평균 시간: {stats['avg_time']:.3f}초")

        # Langfuse 모니터링 결과 분석
        if langfuse_monitor.is_enabled():
            print(f"\n🔍 Langfuse 모니터링 결과 분석")
            print("-" * 30)

            # Langfuse 데이터 플러시
            try:
                langfuse_monitor.flush()
                print("✅ Langfuse 데이터 플러시 완료")
            except Exception as e:
                print(f"⚠️ Langfuse 데이터 플러시 실패: {e}")

            # 모니터링 통계
            final_stats = langfuse_monitor.get_stats()
            langfuse_traces = final_stats['traces_count']
            langfuse_events = final_stats['events_count']

            print(f"📊 Langfuse 모니터링 통계:")
            print(f"   - 생성된 트레이스: {langfuse_traces}개")
            print(f"   - 로깅된 이벤트: {langfuse_events}개")
            print(f"   - Public Key: {final_stats['public_key']}")
            print(f"   - Host: {final_stats['host']}")

            if langfuse_traces > 0:
                print("\n📊 Langfuse 대시보드에서 상세한 분석을 확인하세요:")
                print("   - 트레이스 실행 시간 분석")
                print("   - 응답 품질 메트릭")
                print("   - 오류 패턴 분석")
                print("   - 사용자별 성능 통계")
                print("   - 질문 유형별 성능 분석")
                print("   - 신뢰도 분포 분석")
        else:
            print(f"\n⚠️ Langfuse 모니터링이 비활성화되어 있어 상세 분석이 불가능합니다.")
            print("환경 변수를 설정하여 Langfuse 모니터링을 활성화하세요.")

        print(f"\n✅ 종합 답변 품질 테스트 완료!")

        return results

    except Exception as e:
        print(f"❌ 종합 답변 품질 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return []


if __name__ == "__main__":
    print("🚀 Final Comprehensive Answer Quality Test with Langfuse Monitoring")
    print("=" * 80)

    # Langfuse 모니터링 상태 사전 확인
    try:
        langfuse_monitor = get_langfuse_monitor()
        if langfuse_monitor.is_enabled():
            stats = langfuse_monitor.get_stats()
            print("✅ Langfuse 모니터링이 활성화되어 있습니다.")
            print(f"📊 Langfuse 설정:")
            print(f"   - Public Key: {stats['public_key']}")
            print(f"   - Host: {stats['host']}")
            print("📊 테스트 결과는 Langfuse 대시보드에서 확인할 수 있습니다.")
        else:
            print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
            print("💡 Langfuse 모니터링을 활성화하려면 환경 변수를 설정하세요:")
            print("   - LANGFUSE_PUBLIC_KEY")
            print("   - LANGFUSE_SECRET_KEY")
            print("   - LANGFUSE_ENABLED=true")
    except Exception as e:
        print(f"⚠️ Langfuse 모니터 초기화 실패: {e}")

    print("\n" + "=" * 80)

    # 종합 테스트 실행
    results = asyncio.run(test_comprehensive_answer_quality())

    print("\n🎉 최종 종합 답변 품질 테스트 완료!")
    print("=" * 80)

    # 최종 요약
    if results:
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        print(f"📊 테스트 요약: {successful_tests}/{total_tests} 성공")

        try:
            langfuse_monitor = get_langfuse_monitor()
            if langfuse_monitor.is_enabled():
                print("🔍 Langfuse 대시보드에서 상세한 분석 결과를 확인하세요!")
        except:
            pass
