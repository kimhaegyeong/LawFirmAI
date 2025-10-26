# -*- coding: utf-8 -*-
"""
Enhanced Comprehensive Answer Quality Test with Real AI Models
실제 AI 모델과 RAG 시스템을 사용한 종합 답변 품질 테스트
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

# Langfuse 모니터링 환경 변수 설정 (.env 파일에서 로드)
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 로드
    print("✅ .env 파일에서 환경 변수 로드됨")
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않음. pip install python-dotenv")
except Exception as e:
    print(f"⚠️ .env 파일 로드 실패: {e}")

print("🚀 Enhanced 종합 답변 품질 테스트 (실제 AI 모델 사용)")
print("=" * 70)

try:
    from source.services.chat.enhanced_chat_service import EnhancedChatService
    from source.utils.config import Config
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

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
            except Exception:
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
                span_context = self.langfuse_client.start_as_current_span(
                    name=name,
                    metadata={
                        'user_id': user_id,
                        'session_id': session_id,
                        'test_type': 'enhanced_comprehensive_quality_test'
                    }
                )
                return span_context
            except Exception:
                return None
        else:
            trace = {
                'id': f"trace_{len(self.traces)}_{int(time.time())}",
                'name': name,
                'user_id': user_id,
                'session_id': session_id,
                'start_time': time.time(),
                'events': []
            }
            self.traces.append(trace)
            return trace

    def log_generation(self, trace_id, name, input_data, output_data, metadata):
        """생성 이벤트 로깅"""
        if not self.enabled:
            return

        if self.langfuse_client:
            try:
                return self.langfuse_client.create_event(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata
                )
            except Exception:
                self._log_local_event(trace_id, name, input_data, output_data, metadata, 'generation')
        else:
            self._log_local_event(trace_id, name, input_data, output_data, metadata, 'generation')

    def log_event(self, trace_id, name, input_data, output_data, metadata):
        """일반 이벤트 로깅"""
        if not self.enabled:
            return

        if self.langfuse_client:
            try:
                return self.langfuse_client.create_event(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata
                )
            except Exception:
                self._log_local_event(trace_id, name, input_data, output_data, metadata, 'event')
        else:
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

    def flush(self):
        """데이터 플러시"""
        if not self.enabled:
            return

        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception:
                pass

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


def generate_enhanced_test_questions() -> List[Dict[str, Any]]:
    """향상된 테스트 질문 생성 - 실제 AI 모델 테스트용"""
    questions = [
        # 노동법 관련 질문 (복잡한 시나리오)
        {
            "question": "회사에서 야간 근무를 하는데, 야간 수당과 연장근무 수당이 중복 지급되는지 궁금합니다. 근로기준법상 어떤 규정이 있나요?",
            "category": "노동법",
            "expected_type": "labor_law",
            "priority": "high",
            "expected_style": "detailed",
            "complexity": "high",
            "requires_calculation": True
        },

        # # 상속법 관련 질문 (실제 사례)
        # {
        #     "question": "아버지가 돌아가셨는데 유언장이 없고, 어머니와 형제 2명이 있습니다. 상속분은 어떻게 되고, 상속포기를 하려면 어떤 절차가 필요한가요?",
        #     "category": "상속법",
        #     "expected_type": "inheritance",
        #     "priority": "high",
        #     "expected_style": "professional",
        #     "complexity": "medium",
        #     "requires_calculation": True
        # },

        # # 형사법 관련 질문 (구체적 상황)
        # {
        #     "question": "온라인 쇼핑몰에서 가짜 상품을 판매했다가 고객이 신고했습니다. 이 경우 사기죄가 성립하는지, 그리고 처벌 기준은 어떻게 되나요?",
        #     "category": "형사법",
        #     "expected_type": "criminal_law",
        #     "priority": "high",
        #     "expected_style": "concise",
        #     "complexity": "medium",
        #     "requires_calculation": False
        # },

        # # 지적재산권법 관련 질문 (실무적 질문)
        # {
        #     "question": "유튜브에서 다른 사람의 음악을 배경음악으로 사용했는데 저작권 침해로 신고받았습니다. 어떤 대응 방안이 있고, 손해배상은 얼마나 될 수 있나요?",
        #     "category": "지적재산권",
        #     "expected_type": "intellectual_property",
        #     "priority": "medium",
        #     "expected_style": "interactive",
        #     "complexity": "high",
        #     "requires_calculation": True
        # },

        # # 행정법 관련 질문 (복잡한 절차)
        # {
        #     "question": "아파트 건축허가를 받았는데 인근 주민들이 소음과 일조권 침해를 이유로 이의를 제기했습니다. 행정심판을 신청하려면 어떤 절차와 서류가 필요한가요?",
        #     "category": "행정법",
        #     "expected_type": "administrative_law",
        #     "priority": "medium",
        #     "expected_style": "friendly",
        #     "complexity": "high",
        #     "requires_calculation": False
        # },
    ]

    return questions


class EnhancedComprehensiveTest:
    """실제 AI 모델과 RAG 시스템을 사용한 종합 테스트"""

    def __init__(self):
        self.config = Config()
        self.langfuse_monitor = get_langfuse_monitor()
        self.enhanced_chat_service = None

    async def initialize_services(self):
        """서비스 초기화"""
        try:
            self.enhanced_chat_service = EnhancedChatService(self.config)
            return True
        except Exception as e:
            print(f"❌ 서비스 초기화 실패: {e}")
            return False

    async def test_with_real_ai(self):
        """실제 AI 모델과 RAG 시스템 사용 테스트"""
        print("\n🚀 Enhanced 종합 테스트 시작")
        print("=" * 50)

        if not await self.initialize_services():
            print("❌ 서비스 초기화 실패")
            return []

        langfuse_enabled = self.langfuse_monitor.is_enabled()
        if langfuse_enabled:
            print("📊 Langfuse 모니터링 활성화")

        test_questions = generate_enhanced_test_questions()
        print(f"📝 테스트 질문: {len(test_questions)}개")

        results = []
        start_time = time.time()

        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]

            print(f"\n[{i}/{len(test_questions)}] {category}: {question[:60]}...")

            trace = None
            if self.langfuse_monitor.is_enabled():
                trace = self.langfuse_monitor.create_trace(
                    name=f"enhanced_test_question_{i}",
                    user_id=f"enhanced_test_user_{i}",
                    session_id=f"enhanced_test_session_{i}"
                )

            try:
                result = await self.enhanced_chat_service.process_message(
                    message=question,
                    user_id=f"enhanced_test_user_{i}",
                    session_id=f"enhanced_test_session_{i}"
                )

                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                generation_method = result.get('generation_method', 'unknown')
                langgraph_used = result.get('langgraph_enabled', False)
                sources_count = len(result.get('sources', []))

                print(f"   ✓ 신뢰도: {confidence:.2f} | 시간: {processing_time:.2f}초 | "
                      f"방법: {generation_method} | LangGraph: {langgraph_used} | 소스: {sources_count}개")

                if self.langfuse_monitor.is_enabled() and trace:
                    self.langfuse_monitor.log_generation(
                        trace_id=trace,
                        name="enhanced_ai_response",
                        input_data={"question": question, "category": category},
                        output_data={
                            "confidence": confidence,
                            "processing_time": processing_time,
                            "generation_method": generation_method,
                            "langgraph_used": langgraph_used
                        },
                        metadata={"test_case_id": i}
                    )

                results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': confidence,
                    'generation_method': generation_method,
                    'langgraph_used': langgraph_used,
                    'sources_count': sources_count
                })

            except Exception as e:
                print(f"   ✗ 실패: {e}")

                if self.langfuse_monitor.is_enabled() and trace:
                    self.langfuse_monitor.log_event(
                        trace_id=trace,
                        name="enhanced_test_error",
                        input_data={"question": question, "category": category},
                        output_data={"error": str(e)},
                        metadata={"test_case_id": i, "success": False}
                    )

                results.append({
                    'test_case': test_case,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        total_time = time.time() - start_time
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        langgraph_tests = sum(1 for r in results if r.get('langgraph_used', False))

        print("\n📊 테스트 결과")
        print(f"   총: {total_tests} | 성공: {successful_tests} | 실패: {total_tests - successful_tests}")
        print(f"   LangGraph 사용: {langgraph_tests}/{total_tests} | 총 시간: {total_time:.1f}초")

        if successful_tests > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful_tests
            avg_time = sum(r.get('processing_time', 0) for r in results if r['success']) / successful_tests
            print(f"   평균 신뢰도: {avg_confidence:.2f} | 평균 처리 시간: {avg_time:.2f}초")

        if self.langfuse_monitor.is_enabled():
            self.langfuse_monitor.flush()

        return results


async def main():
    """메인 실행 함수"""
    print("🚀 Enhanced Comprehensive Answer Quality Test")
    print("=" * 70)

    test_instance = EnhancedComprehensiveTest()
    results = await test_instance.test_with_real_ai()

    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
