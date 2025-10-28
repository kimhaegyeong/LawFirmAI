# -*- coding: utf-8 -*-
"""
LawFirmAI - Streamlit 애플리케이션
Phase 2의 모든 개선사항을 통합한 최적화된 버전
"""

import gc
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangGraph 활성화 설정
os.environ["USE_LANGGRAPH"] = os.getenv("USE_LANGGRAPH", "true")

# Streamlit 및 기타 라이브러리
import psutil
import torch
from streamlit_chat import message

import streamlit as st

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.services.answer_formatter import AnswerFormatter
from source.services.chat_service import ChatService
from source.services.confidence_calculator import ConfidenceCalculator
from source.services.context_builder import ContextBuilder
from source.services.context_compressor import ContextCompressor
from source.services.conversation_flow_tracker import ConversationFlowTracker
from source.services.dynamic_prompt_updater import create_dynamic_prompt_updater
from source.services.emotion_intent_analyzer import (
    EmotionIntentAnalyzer,
    EmotionType,
    IntentType,
    UrgencyLevel,
)
from source.services.gemini_client import GeminiClient
from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.improved_answer_generator import ImprovedAnswerGenerator

# Phase 1: 대화 맥락 강화 모듈
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.services.legal_term_expander import LegalTermExpander

# AKLS 모듈 (임시 비활성화)
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.optimized_search_engine import OptimizedSearchEngine
from source.services.performance_monitor import PerformanceContext
from source.services.performance_monitor import (
    PerformanceMonitor as SourcePerformanceMonitor,
)
from source.services.prompt_templates import PromptTemplateManager
from source.services.question_classifier import QuestionClassifier, QuestionType
from source.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)

# Phase 2: 개인화 및 지능형 분석 모듈
from source.services.user_profile_manager import (
    DetailLevel,
    ExpertiseLevel,
    UserProfileManager,
)
from source.utils.config import Config
from source.utils.langgraph_config import LangGraphConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/streamlit_app.log')
    ]
)
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """메모리 최적화 클래스"""

    def __init__(self, max_memory_percent: float = 85.0):
        self.max_memory_percent = max_memory_percent
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def memory_efficient_inference(self):
        """메모리 효율적인 추론 컨텍스트"""
        # 추론 전 메모리 정리
        self._cleanup_memory()

        try:
            yield
        finally:
            # 추론 후 메모리 정리
            self._cleanup_memory()

    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def monitor_memory_usage(self) -> float:
        """메모리 사용량 모니터링"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.max_memory_percent:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
            self._cleanup_memory()
        return memory_percent

class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0

    def log_request(self, response_time: float, success: bool = True, operation: str = None):
        """요청 로깅"""
        self.total_requests += 1
        self.response_times.append(response_time)

        if not success:
            self.error_count += 1

        # 최근 100개 요청만 유지
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if not self.response_times:
            return {
                "avg_response_time": 0,
                "error_rate": 0,
                "total_requests": 0
            }

        return {
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "total_requests": self.total_requests
        }

@st.cache_resource
def initialize_app():
    """Streamlit 캐시를 사용한 앱 초기화"""
    app = StreamlitApp()
    if not app.initialize_components():
        logger.error("Failed to initialize components")
    return app

class StreamlitApp:
    """Streamlit 전용 LawFirmAI 애플리케이션"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_optimizer = MemoryOptimizer()
        self.performance_monitor = PerformanceMonitor()

        # 컴포넌트 초기화
        self.db_manager = None
        self.vector_store = None
        self.question_classifier = None
        self.hybrid_search_engine = None
        self.prompt_template_manager = None
        self.unified_prompt_manager = None
        self.dynamic_prompt_updater = None
        self.confidence_calculator = None
        self.legal_term_expander = None
        self.gemini_client = None
        self.improved_answer_generator = None
        self.answer_formatter = None
        self.context_builder = None

        # ChatService 초기화 (LangGraph 통합)
        self.chat_service = None

        # LangGraph 워크플로우 서비스
        self.langgraph_workflow = None

        # Phase 1: 대화 맥락 강화 컴포넌트
        self.session_manager = None
        self.multi_turn_handler = None
        self.context_compressor = None

        # Phase 2: 개인화 및 지능형 분석 컴포넌트
        self.user_profile_manager = None
        self.emotion_intent_analyzer = None
        self.conversation_flow_tracker = None

        # 세션 관리
        self.current_session_id = None
        self.current_user_id = None

        # 초기화 상태
        self.is_initialized = False
        self.initialization_error = None

    def initialize_components(self) -> bool:
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing LawFirmAI components...")
            start_time = time.time()

            with self.memory_optimizer.memory_efficient_inference():
                # 데이터베이스 관리자 초기화
                self.db_manager = DatabaseManager("data/lawfirm.db")
                self.logger.info("Database manager initialized")

                # 벡터 스토어 초기화 (판례 데이터용)
                self.vector_store = LegalVectorStore(
                    model_name='jhgan/ko-sroberta-multitask',
                    dimension=768,
                    index_type='flat'
                )
                # 판례 벡터 인덱스 로드
                if not self.vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
                    self.logger.warning("Failed to load precedent vector index, using law index")
                    self.vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta')
                self.logger.info("Vector store initialized")

                # 질문 분류기 초기화
                self.question_classifier = QuestionClassifier()
                self.logger.info("Question classifier initialized")

                # 하이브리드 검색 엔진 초기화
                self.hybrid_search_engine = HybridSearchEngine()
                self.logger.info("Hybrid search engine initialized")

                # 최적화된 검색 엔진 초기화
                self.optimized_search_engine = OptimizedSearchEngine(
                    vector_store=self.vector_store,
                    hybrid_engine=self.hybrid_search_engine,
                    cache_size=1000,
                    cache_ttl=3600
                )
                self.logger.info("Optimized search engine initialized")

                # 프롬프트 템플릿 관리자 초기화
                self.prompt_template_manager = PromptTemplateManager()
                self.logger.info("Prompt template manager initialized")

                # 통합 프롬프트 관리자
                self.unified_prompt_manager = UnifiedPromptManager()
                self.logger.info("Unified prompt manager initialized")

                # 동적 프롬프트 업데이터
                self.dynamic_prompt_updater = create_dynamic_prompt_updater(self.unified_prompt_manager)
                self.logger.info("Dynamic prompt updater initialized")

                # 신뢰도 계산기 초기화
                self.confidence_calculator = ConfidenceCalculator()
                self.logger.info("Confidence calculator initialized")

                # 법률 용어 확장기 초기화
                self.legal_term_expander = LegalTermExpander()
                self.logger.info("Legal term expander initialized")

                # Gemini 클라이언트 초기화
                if os.getenv('GEMINI_ENABLED', 'true').lower() == 'true':
                    self.gemini_client = GeminiClient()
                    self.logger.info("Gemini client initialized")

                # 답변 포맷터 초기화
                self.answer_formatter = AnswerFormatter()
                self.logger.info("Answer formatter initialized")

                # 컨텍스트 빌더 초기화
                self.context_builder = ContextBuilder()
                self.logger.info("Context builder initialized")

                # 개선된 답변 생성기 초기화
                self.improved_answer_generator = ImprovedAnswerGenerator(
                    gemini_client=self.gemini_client,
                    prompt_template_manager=self.prompt_template_manager,
                    confidence_calculator=self.confidence_calculator,
                    answer_formatter=self.answer_formatter,
                    context_builder=self.context_builder
                )
                self.logger.info("Improved answer generator initialized")

                # ChatService 초기화 (LangGraph 통합)
                config = Config()
                self.chat_service = ChatService(config)
                self.logger.info("ChatService initialized with LangGraph integration")

                # Phase 1: 대화 맥락 강화 컴포넌트 초기화
                self.session_manager = IntegratedSessionManager("data/conversations.db")
                self.logger.info("Integrated session manager initialized")

                self.multi_turn_handler = MultiTurnQuestionHandler()
                self.logger.info("Multi-turn question handler initialized")

                self.context_compressor = ContextCompressor(max_tokens=2000)
                self.logger.info("Context compressor initialized")

                # Phase 2: 개인화 및 지능형 분석 컴포넌트 초기화
                self.user_profile_manager = UserProfileManager(self.session_manager.conversation_store)
                self.logger.info("User profile manager initialized")

                self.emotion_intent_analyzer = EmotionIntentAnalyzer()
                self.logger.info("Emotion intent analyzer initialized")

                self.conversation_flow_tracker = ConversationFlowTracker()
                self.logger.info("Conversation flow tracker initialized")

                # 세션 ID 생성
                self.current_session_id = f"streamlit_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_user_id = "streamlit_user"  # 기본 사용자 ID
                self.logger.info(f"Session initialized: {self.current_session_id}")

            initialization_time = time.time() - start_time
            self.logger.info(f"All components initialized successfully in {initialization_time:.2f} seconds")

            self.is_initialized = True
            return True

        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    def process_query(self, query: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """질의 처리 (Phase 1 대화 맥락 강화 적용)"""
        if not self.is_initialized:
            return {
                "answer": "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "error": "System not initialized",
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"}
            }

        # 세션 ID 설정
        if not session_id:
            session_id = self.current_session_id
        if not user_id:
            user_id = self.current_user_id

        start_time = time.time()

        # 성능 모니터링
        try:
            with self.memory_optimizer.memory_efficient_inference():
                # Phase 1: 대화 맥락 처리
                context = None
                resolved_query = query
                multi_turn_info = {}

                if self.session_manager:
                    # 세션 컨텍스트 로드
                    context = self.session_manager.get_or_create_session(session_id, user_id)

                    # 다중 턴 질문 처리
                    if self.multi_turn_handler and context:
                        is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, context)
                        if is_multi_turn:
                            multi_turn_result = self.multi_turn_handler.build_complete_query(query, context)
                            resolved_query = multi_turn_result["resolved_query"]
                            multi_turn_info = {
                                "is_multi_turn": True,
                                "original_query": query,
                                "resolved_query": resolved_query,
                                "confidence": multi_turn_result["confidence"],
                                "reasoning": multi_turn_result["reasoning"]
                            }
                            self.logger.info(f"Multi-turn question resolved: '{query}' -> '{resolved_query}'")

                # Phase 2: 개인화 및 지능형 분석
                personalized_context = {}
                emotion_intent_info = {}
                flow_tracking_info = {}

                if self.user_profile_manager:
                    # 개인화된 컨텍스트 생성
                    personalized_context = self.user_profile_manager.get_personalized_context(user_id, resolved_query)
                    self.logger.info(f"Personalized context created for user {user_id}")

                if self.emotion_intent_analyzer:
                    # 감정 및 의도 분석
                    emotion_result = self.emotion_intent_analyzer.analyze_emotion(resolved_query)
                    intent_result = self.emotion_intent_analyzer.analyze_intent(resolved_query, context)

                    # 응답 톤 결정
                    response_tone = self.emotion_intent_analyzer.get_contextual_response_tone(
                        emotion_result, intent_result, personalized_context
                    )

                    emotion_intent_info = {
                        "emotion": {
                            "primary_emotion": emotion_result.primary_emotion.value,
                            "confidence": emotion_result.confidence,
                            "intensity": emotion_result.intensity,
                            "reasoning": emotion_result.reasoning
                        },
                        "intent": {
                            "primary_intent": intent_result.primary_intent.value,
                            "confidence": intent_result.confidence,
                            "urgency_level": intent_result.urgency_level.value,
                            "reasoning": intent_result.reasoning
                        },
                        "response_tone": {
                            "tone_type": response_tone.tone_type,
                            "empathy_level": response_tone.empathy_level,
                            "formality_level": response_tone.formality_level,
                            "urgency_response": response_tone.urgency_response,
                            "explanation_depth": response_tone.explanation_depth
                        }
                    }
                    self.logger.info(f"Emotion: {emotion_result.primary_emotion.value}, Intent: {intent_result.primary_intent.value}")

                if self.conversation_flow_tracker and context:
                    # 대화 흐름 추적
                    from source.services.conversation_manager import ConversationTurn
                    self.conversation_flow_tracker.track_conversation_flow(session_id,
                        ConversationTurn(
                            user_query=query,
                            bot_response="",
                            timestamp=datetime.now(),
                            question_type="general_question"
                        )
                    )

                    # 다음 의도 예측 및 후속 질문 제안
                    predicted_intents = self.conversation_flow_tracker.predict_next_intent(context)
                    suggested_questions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
                    conversation_state = self.conversation_flow_tracker.get_conversation_state(context)

                    flow_tracking_info = {
                        "predicted_intents": predicted_intents,
                        "suggested_questions": suggested_questions,
                        "conversation_state": conversation_state
                    }
                    self.logger.info(f"Flow tracking: state={conversation_state}, suggestions={len(suggested_questions)}")

                # ChatService를 사용한 처리 (실제 RAG 시스템)
                if self.chat_service and hasattr(self.chat_service, 'improved_answer_generator') and self.chat_service.improved_answer_generator:
                    import asyncio
                    result = asyncio.run(self.chat_service.process_message(resolved_query, session_id=session_id))

                    response_time = time.time() - start_time
                    self.performance_monitor.log_request(response_time, success=True, operation="process_query")

                    # Phase 1: 세션에 턴 추가
                    if self.session_manager and context:
                        self.session_manager.add_turn(
                            session_id,
                            query,
                            result.get("response", ""),
                            result.get("question_type", "general_question"),
                            user_id
                        )

                    # 컨텍스트 압축 확인
                    compression_info = {}
                    if self.context_compressor and context:
                        current_tokens = self.context_compressor.calculate_tokens(context)
                        if current_tokens > 1500:
                            compression_result = self.context_compressor.compress_long_conversation(context)
                            compression_info = {
                                "compressed": True,
                                "original_tokens": compression_result.original_tokens,
                                "compressed_tokens": compression_result.compressed_tokens,
                                "compression_ratio": compression_result.compression_ratio,
                                "summary": compression_result.summary
                            }
                            self.logger.info(f"Context compressed: {compression_result.compression_ratio:.2f} ratio")

                    return {
                        "answer": result.get("response", ""),
                        "confidence": {
                            "confidence": result.get("confidence", 0.0),
                            "reliability_level": "HIGH" if result.get("confidence", 0) > 0.7 else "MEDIUM" if result.get("confidence", 0) > 0.4 else "LOW"
                        },
                        "processing_time": response_time,
                        "memory_usage": self.memory_optimizer.monitor_memory_usage(),
                        "question_type": result.get("question_type", "general_question"),
                        "session_id": session_id,
                        "user_id": user_id,
                        "multi_turn_info": multi_turn_info,
                        "compression_info": compression_info,
                        "context_stats": {
                            "total_turns": len(context.turns) if context else 0,
                            "total_entities": sum(len(entities) for entities in context.entities.values()) if context else 0,
                            "topics": list(context.topic_stack) if context else []
                        },
                        "personalized_context": personalized_context,
                        "emotion_intent_info": emotion_intent_info,
                        "flow_tracking_info": flow_tracking_info,
                        "session_id": result.get("session_id"),
                        "query_type": result.get("query_type", ""),
                        "legal_references": result.get("legal_references", []),
                        "processing_steps": result.get("processing_steps", []),
                        "metadata": result.get("metadata", {}),
                        "errors": result.get("errors", [])
                    }
                else:
                    # 기존 방식으로 폴백
                    return self._process_query_legacy(query, start_time)

        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.log_request(response_time, success=False, operation="process_query_error")
            self.logger.error(f"Error processing query: {e}")

            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "error": str(e),
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                "processing_time": response_time
            }

    def _process_query_legacy(self, query: str, start_time: float) -> Dict[str, Any]:
        """기존 방식으로 질의 처리 (폴백)"""
        try:
            with self.memory_optimizer.memory_efficient_inference():
                # 질문 분류
                question_classification = self.question_classifier.classify_question(query)

                # 지능형 검색 실행
                search_results = self.hybrid_search_engine.search_with_question_type(
                    query=query,
                    question_type=question_classification,
                    max_results=10
                )

                # 답변 생성
                answer_result = self.improved_answer_generator.generate_answer(
                    query=query,
                    question_type=question_classification,
                    context="",
                    sources=search_results,
                    conversation_history=None
                )

                response_time = time.time() - start_time
                self.performance_monitor.log_request(response_time, success=True, operation="legacy_process_query")

                return {
                    "answer": answer_result.answer,
                    "question_type": question_classification.question_type.value,
                    "confidence": {
                        "confidence": answer_result.confidence.confidence,
                        "reliability_level": answer_result.confidence.level.value
                    },
                    "processing_time": response_time,
                    "memory_usage": self.memory_optimizer.monitor_memory_usage()
                }

        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.log_request(response_time, success=False, operation="legacy_process_query_error")
            self.logger.error(f"Error in legacy processing: {e}")

            return {
                "answer": "기존 처리 방식에서 오류가 발생했습니다.",
                "error": str(e),
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                "processing_time": response_time
            }

def main():
    """메인 함수"""
    st.set_page_config(
        page_title="LawFirmAI - 법률 AI 어시스턴트",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚖️ LawFirmAI - 법률 AI 어시스턴트")

    # 시스템 초기화
    if 'app' not in st.session_state:
        with st.spinner('시스템을 초기화하는 중...'):
            st.session_state.app = initialize_app()

    app = st.session_state.app

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")

        # 탭 생성
        tabs = st.tabs(["💬 채팅", "👤 프로필", "📊 상태"])

        with tabs[0]:
            st.subheader("채팅 설정")
            user_id = st.text_input("사용자 ID", value=app.current_user_id)

        with tabs[1]:
            st.subheader("사용자 프로필")
            expertise_level = st.selectbox("전문성 수준", ["beginner", "intermediate", "advanced", "expert"])
            detail_level = st.selectbox("답변 상세도", ["simple", "medium", "detailed"])

        with tabs[2]:
            st.subheader("시스템 상태")
            if app.is_initialized:
                st.success("✅ 시스템이 정상적으로 초기화되었습니다.")
                stats = app.performance_monitor.get_stats()
                st.json(stats)
            else:
                st.error("❌ 시스템 초기화 실패")
                st.error(app.initialization_error)

    # 채팅 인터페이스
    st.markdown("### 법률 관련 질문에 정확하고 신뢰할 수 있는 답변을 제공합니다.")

    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 채팅 표시
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

                # 신뢰도 정보 표시
                if "confidence" in msg:
                    conf_info = msg["confidence"]
                    st.caption(f"신뢰도: {conf_info.get('confidence', 0):.1%} | "
                              f"수준: {conf_info.get('reliability_level', 'Unknown')} | "
                              f"처리 시간: {msg.get('processing_time', 0):.2f}초")

    # 질문 입력
    user_input = st.chat_input("법률 관련 질문을 입력하세요...")

    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    # 답변 생성
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_msg = st.session_state.chat_history[-1]["content"]

        with st.spinner('답변을 생성하는 중...'):
            result = app.process_query(last_msg, user_id=user_id)

            answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")

            # 답변 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "confidence": result.get("confidence", {}),
                "processing_time": result.get("processing_time", 0)
            })

            st.rerun()

if __name__ == "__main__":
    main()
