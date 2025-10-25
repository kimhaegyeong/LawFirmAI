# -*- coding: utf-8 -*-
"""
Langfuse 모니터링 데모
LangChain과 LangGraph 모니터링 사용 예시
"""

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# LangChain 관련 import
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain이 설치되지 않았습니다. pip install langchain을 실행하세요.")

# LangGraph 관련 import
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph가 설치되지 않았습니다. pip install langgraph를 실행하세요.")

# 모니터링 모듈 import
from source.utils.langfuse_monitor import get_langfuse_monitor
from source.utils.langchain_monitor import (
    monitor_chain, monitor_llm, monitor_langgraph,
    get_monitored_callback_manager
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_langchain_monitoring():
    """LangChain 모니터링 데모"""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain이 설치되지 않아 데모를 실행할 수 없습니다.")
        return
    
    print("🚀 LangChain 모니터링 데모 시작")
    print("=" * 50)
    
    # 모니터링 상태 확인
    monitor = get_langfuse_monitor()
    if not monitor.is_enabled():
        print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
        print("환경 변수 LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 설정하세요.")
        return
    
    print("✅ Langfuse 모니터링이 활성화되어 있습니다.")
    
    try:
        # OpenAI LLM 초기화 (API 키 필요)
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
            print("실제 LLM 테스트를 위해서는 OpenAI API 키가 필요합니다.")
            return
        
        # LLM 생성 및 모니터링 적용
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        monitored_llm = monitor_llm(llm, name="legal_assistant")
        
        # 프롬프트 템플릿 생성
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="다음 법률 질문에 답변해주세요: {question}"
        )
        
        # 체인 생성 및 모니터링 적용
        chain = LLMChain(llm=monitored_llm, prompt=prompt_template)
        monitored_chain = monitor_chain(chain, name="legal_qa_chain")
        
        # 테스트 질문들
        test_questions = [
            "계약서 작성 시 주의사항은 무엇인가요?",
            "손해배상 청구 방법을 알려주세요.",
            "부동산 매매 계약의 필수 사항은 무엇인가요?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 질문 {i}: {question}")
            
            try:
                # 체인 실행 (모니터링 포함)
                result = monitored_chain.run(
                    question=question,
                    user_id=f"demo_user_{i}",
                    session_id="demo_session"
                )
                
                print(f"✅ 답변: {result[:100]}...")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
        print("\n✅ LangChain 모니터링 데모 완료")
        print("Langfuse 대시보드에서 트레이스를 확인하세요.")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")

def demo_langgraph_monitoring():
    """LangGraph 모니터링 데모"""
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph가 설치되지 않아 데모를 실행할 수 없습니다.")
        return
    
    print("\n🚀 LangGraph 모니터링 데모 시작")
    print("=" * 50)
    
    # 모니터링 상태 확인
    monitor = get_langfuse_monitor()
    if not monitor.is_enabled():
        print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
        return
    
    try:
        # 간단한 상태 정의
        from typing import TypedDict
        
        class LegalState(TypedDict):
            question: str
            analysis: str
            answer: str
            confidence: float
        
        # 노드 함수들 정의
        def analyze_question(state: LegalState) -> LegalState:
            """질문 분석 노드"""
            question = state["question"]
            analysis = f"질문 '{question}'을 분석했습니다. 법률 관련 질문으로 분류됩니다."
            
            return {
                **state,
                "analysis": analysis
            }
        
        def generate_answer(state: LegalState) -> LegalState:
            """답변 생성 노드"""
            question = state["question"]
            analysis = state["analysis"]
            
            # 간단한 답변 생성 (실제로는 LLM 사용)
            answer = f"질문 '{question}'에 대한 답변: 이는 법률 전문가와 상담하시는 것이 좋습니다."
            confidence = 0.8
            
            return {
                **state,
                "answer": answer,
                "confidence": confidence
            }
        
        # 그래프 구성
        workflow = StateGraph(LegalState)
        
        # 노드 추가
        workflow.add_node("analyze", analyze_question)
        workflow.add_node("generate", generate_answer)
        
        # 엣지 추가
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", END)
        
        # 시작점 설정
        workflow.set_entry_point("analyze")
        
        # 그래프 컴파일 및 모니터링 적용
        monitored_graph = monitor_langgraph(workflow, name="legal_workflow")
        compiled_graph = monitored_graph.compile()
        
        # 테스트 실행
        test_inputs = [
            {"question": "계약서 작성 방법을 알려주세요"},
            {"question": "손해배상 청구 절차는 어떻게 되나요?"},
            {"question": "부동산 매매 시 주의사항은 무엇인가요?"}
        ]
        
        for i, input_data in enumerate(test_inputs, 1):
            print(f"\n📝 워크플로우 실행 {i}: {input_data['question']}")
            
            try:
                # 그래프 실행 (모니터링 포함)
                result = monitored_graph.invoke(
                    input_data,
                    user_id=f"demo_user_{i}",
                    session_id="demo_session"
                )
                
                print(f"✅ 분석: {result['analysis']}")
                print(f"✅ 답변: {result['answer']}")
                print(f"✅ 신뢰도: {result['confidence']}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
        print("\n✅ LangGraph 모니터링 데모 완료")
        print("Langfuse 대시보드에서 워크플로우 트레이스를 확인하세요.")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")

def demo_custom_monitoring():
    """커스텀 모니터링 데모"""
    print("\n🚀 커스텀 모니터링 데모 시작")
    print("=" * 50)
    
    monitor = get_langfuse_monitor()
    if not monitor.is_enabled():
        print("⚠️ Langfuse 모니터링이 비활성화되어 있습니다.")
        return
    
    try:
        # 커스텀 트레이스 생성
        trace = monitor.create_trace(
            name="custom_legal_analysis",
            user_id="demo_user",
            session_id="custom_demo"
        )
        
        if trace:
            print(f"✅ 트레이스 생성됨: {trace.id}")
            
            # 커스텀 이벤트 로깅
            monitor.log_event(
                trace_id=trace.id,
                name="question_received",
                input_data={"question": "계약서 검토 요청"},
                metadata={"source": "demo"}
            )
            
            # 커스텀 생성 로깅
            monitor.log_generation(
                trace_id=trace.id,
                name="legal_analysis",
                input_data={"question": "계약서 검토 요청"},
                output_data={"analysis": "계약서 검토가 필요합니다."},
                metadata={"model": "custom", "confidence": 0.9}
            )
            
            # 데이터 플러시
            monitor.flush()
            
            print("✅ 커스텀 모니터링 이벤트가 기록되었습니다.")
        
    except Exception as e:
        print(f"❌ 커스텀 모니터링 중 오류 발생: {e}")

def main():
    """메인 데모 함수"""
    print("🔍 Langfuse 모니터링 데모")
    print("=" * 60)
    
    # 환경 변수 확인
    required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        print("langfuse.env.example 파일을 참고하여 설정하세요.")
        print("\n모니터링 없이 데모를 실행합니다...")
    
    # 데모 실행
    demo_langchain_monitoring()
    demo_langgraph_monitoring()
    demo_custom_monitoring()
    
    print("\n🎉 모든 데모가 완료되었습니다!")
    print("Langfuse 대시보드에서 모니터링 데이터를 확인하세요.")

if __name__ == "__main__":
    main()
