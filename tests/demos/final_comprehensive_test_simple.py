# -*- coding: utf-8 -*-
"""
Final Comprehensive Answer Quality Test with Langfuse Monitoring
최종 종합 답변 품질 테스트 (Langfuse 모니터링 포함)
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

print("🚀 최종 종합 답변 품질 테스트")
print("=" * 70)

try:
    from source.utils.config import Config
    from source.services.enhanced_chat_service import EnhancedChatService
    # from source.utils.langfuse_monitor import get_langfuse_monitor  # 모듈이 없어서 주석 처리
    print("✅ 모든 모듈 import 성공")
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

# Langfuse 모니터 Mock 클래스 (모듈이 없을 때 사용)
class MockLangfuseMonitor:
    def is_enabled(self):
        return False

    def create_trace(self, name, user_id, session_id):
        return None

    def log_generation(self, trace_id, name, input_data, output_data, metadata):
        pass

    def log_event(self, trace_id, name, input_data, output_data, metadata):
        pass

    def flush(self):
        pass

def get_langfuse_monitor():
    return MockLangfuseMonitor()


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
            print("✅ Langfuse 모니터링이 활성화되어 있습니다.")
        else:
            print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
            print("환경 변수 LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 설정하세요.")

        # Enhanced Chat Service 초기화
        chat_service = EnhancedChatService(config)
        print("✅ Enhanced Chat Service 초기화 성공")
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
            langfuse_traces = sum(1 for r in results if r.get('success', False))
            langfuse_errors = sum(1 for r in results if not r.get('success', True))

            print(f"Langfuse 트레이스 생성: {langfuse_traces}개")
            print(f"Langfuse 오류 로깅: {langfuse_errors}개")
            print(f"총 모니터링 이벤트: {langfuse_traces + langfuse_errors}개")

            if langfuse_traces > 0:
                print("📊 Langfuse 대시보드에서 상세한 분석을 확인하세요:")
                print("   - 트레이스 실행 시간 분석")
                print("   - 응답 품질 메트릭")
                print("   - 오류 패턴 분석")
                print("   - 사용자별 성능 통계")
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
            print("✅ Langfuse 모니터링이 활성화되어 있습니다.")
            print("📊 테스트 결과는 Langfuse 대시보드에서 확인할 수 있습니다.")
        else:
            print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
            print("💡 Langfuse 모니터링을 활성화하려면 환경 변수를 설정하세요:")
            print("   - LANGFUSE_PUBLIC_KEY")
            print("   - LANGFUSE_SECRET_KEY")
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
