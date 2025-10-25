# -*- coding: utf-8 -*-
"""
Langfuse 모니터링 통합 예시
기존 서비스에 모니터링 기능 추가하는 방법
"""

import logging
from typing import Dict, Any, Optional
from source.utils.langfuse_monitor import get_langfuse_monitor, observe_function
from source.utils.langchain_monitor import get_monitored_callback_manager

logger = logging.getLogger(__name__)

class MonitoredChatService:
    """모니터링이 적용된 채팅 서비스 예시"""
    
    def __init__(self):
        self.monitor = get_langfuse_monitor()
        self.callback_manager = get_monitored_callback_manager()
    
    @observe_function(name="process_legal_question")
    def process_legal_question(self, question: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """법률 질문 처리 (모니터링 포함)"""
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name="legal_question_processing",
            user_id=user_id,
            session_id=f"session_{user_id}"
        )
        
        if trace:
            try:
                # 질문 분석
                analysis_result = self._analyze_question(question, trace.id)
                
                # 답변 생성
                answer_result = self._generate_answer(question, analysis_result, trace.id)
                
                # 결과 반환
                result = {
                    "question": question,
                    "analysis": analysis_result,
                    "answer": answer_result,
                    "confidence": 0.9,
                    "trace_id": trace.id
                }
                
                # 최종 결과 로깅
                self.monitor.log_generation(
                    trace_id=trace.id,
                    name="legal_question_complete",
                    input_data={"question": question},
                    output_data=result,
                    metadata={"user_id": user_id}
                )
                
                return result
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name="legal_question_error",
                    input_data={"question": question},
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            # 모니터링이 비활성화된 경우 기본 처리
            return self._process_without_monitoring(question)
    
    def _analyze_question(self, question: str, trace_id: str) -> Dict[str, Any]:
        """질문 분석 (모니터링 포함)"""
        
        # 분석 스팬 생성
        span = self.monitor.create_span(
            trace_id=trace_id,
            name="question_analysis",
            input_data={"question": question}
        )
        
        try:
            # 실제 분석 로직 (예시)
            analysis = {
                "category": "civil_law",
                "complexity": "medium",
                "keywords": ["계약", "손해배상"],
                "confidence": 0.8
            }
            
            # 분석 결과 로깅
            self.monitor.log_generation(
                trace_id=trace_id,
                name="question_analysis",
                input_data={"question": question},
                output_data=analysis,
                metadata={"analysis_method": "rule_based"}
            )
            
            return analysis
            
        except Exception as e:
            self.monitor.log_event(
                trace_id=trace_id,
                name="analysis_error",
                input_data={"question": question},
                output_data={"error": str(e)}
            )
            raise
    
    def _generate_answer(self, question: str, analysis: Dict[str, Any], trace_id: str) -> str:
        """답변 생성 (모니터링 포함)"""
        
        try:
            # 실제 답변 생성 로직 (예시)
            answer = f"질문 '{question}'에 대한 답변입니다. {analysis['category']} 분야의 질문으로 분석되었습니다."
            
            # 답변 생성 로깅
            self.monitor.log_generation(
                trace_id=trace_id,
                name="answer_generation",
                input_data={"question": question, "analysis": analysis},
                output_data={"answer": answer},
                metadata={"generation_method": "template_based"}
            )
            
            return answer
            
        except Exception as e:
            self.monitor.log_event(
                trace_id=trace_id,
                name="generation_error",
                input_data={"question": question, "analysis": analysis},
                output_data={"error": str(e)}
            )
            raise
    
    def _process_without_monitoring(self, question: str) -> Dict[str, Any]:
        """모니터링 없이 처리"""
        return {
            "question": question,
            "answer": f"질문 '{question}'에 대한 기본 답변입니다.",
            "confidence": 0.5
        }

# 사용 예시
def example_usage():
    """사용 예시"""
    
    # 서비스 초기화
    service = MonitoredChatService()
    
    # 질문 처리
    questions = [
        "계약서 작성 방법을 알려주세요",
        "손해배상 청구 절차는 어떻게 되나요?",
        "부동산 매매 시 주의사항은 무엇인가요?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 질문 {i}: {question}")
        
        try:
            result = service.process_legal_question(
                question=question,
                user_id=f"user_{i}"
            )
            
            print(f"✅ 답변: {result['answer']}")
            print(f"✅ 신뢰도: {result['confidence']}")
            print(f"✅ 트레이스 ID: {result.get('trace_id', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    example_usage()
