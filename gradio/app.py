# -*- coding: utf-8 -*-
"""
LawFirmAI - HuggingFace Spaces 전용 Gradio 애플리케이션
Phase 2의 모든 개선사항을 통합한 최적화된 버전
"""

import os
import sys
import logging
import time
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangGraph 활성화 설정
os.environ["USE_LANGGRAPH"] = os.getenv("USE_LANGGRAPH", "true")

# Gradio 및 기타 라이브러리
import gradio as gr
import torch
import psutil

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.services.question_classifier import QuestionClassifier, QuestionType
from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.optimized_search_engine import OptimizedSearchEngine
from source.services.prompt_templates import PromptTemplateManager
from source.services.unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType
from source.services.dynamic_prompt_updater import create_dynamic_prompt_updater
from source.services.prompt_optimizer import create_prompt_optimizer
from source.services.confidence_calculator import ConfidenceCalculator
from source.services.legal_term_expander import LegalTermExpander
from source.services.gemini_client import GeminiClient
from source.services.improved_answer_generator import ImprovedAnswerGenerator
from source.services.answer_formatter import AnswerFormatter
from source.services.context_builder import ContextBuilder
from source.services.chat_service import ChatService
from source.services.performance_monitor import PerformanceMonitor as SourcePerformanceMonitor, PerformanceContext

# Phase 1: 대화 맥락 강화 모듈
from source.services.integrated_session_manager import IntegratedSessionManager

# AKLS 모듈 (임시 비활성화)
# from gradio.components.akls_search_interface import create_akls_interface
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor

# Phase 2: 개인화 및 지능형 분석 모듈
from source.services.user_profile_manager import UserProfileManager, ExpertiseLevel, DetailLevel
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer, EmotionType, IntentType, UrgencyLevel
from source.services.conversation_flow_tracker import ConversationFlowTracker

from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/huggingface_spaces_app.log')
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

class HuggingFaceSpacesApp:
    """HuggingFace Spaces 전용 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_optimizer = MemoryOptimizer()
        self.performance_monitor = PerformanceMonitor()
        # 성능 모니터링 시작 (옵션)
        if hasattr(self.performance_monitor, 'start_monitoring'):
            self.performance_monitor.start_monitoring()
        
        # 컴포넌트 초기화
        self.db_manager = None
        self.vector_store = None
        self.question_classifier = None
        self.hybrid_search_engine = None
        self.prompt_template_manager = None
        self.unified_prompt_manager = None
        self.dynamic_prompt_updater = None
        self.prompt_optimizer = None
        self.confidence_calculator = None
        self.legal_term_expander = None
        self.gemini_client = None
        self.improved_answer_generator = None
        self.answer_formatter = None
        self.context_builder = None
        
        # ChatService 초기화 (LangGraph 통합)
        self.chat_service = None
        
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
        
        # HuggingFace Spaces 환경 설정
        self._setup_huggingface_spaces_env()
    
    def _setup_huggingface_spaces_env(self):
        """HuggingFace Spaces 환경 설정"""
        # 환경 변수 설정
        os.environ.setdefault('GRADIO_SERVER_NAME', '0.0.0.0')
        os.environ.setdefault('GRADIO_SERVER_PORT', '7860')
        os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
        
        # 로깅 레벨 설정
        if os.getenv('HUGGINGFACE_SPACES', '').lower() == 'true':
            logging.getLogger().setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)
    
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
                
                # 프롬프트 최적화기
                self.prompt_optimizer = create_prompt_optimizer(self.unified_prompt_manager)
                self.logger.info("Prompt optimizer initialized")
                
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
                self.current_session_id = f"gradio_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_user_id = "gradio_user"  # 기본 사용자 ID
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
        
        # 성능 모니터링 (PerformanceContext 대신 직접 처리)
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
                                bot_response="",  # 아직 생성되지 않음
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
                                query,  # 원본 질문 저장
                                result.get("response", ""),
                                result.get("question_type", "general_question"),
                                user_id
                            )
                        
                        # 컨텍스트 압축 확인
                        compression_info = {}
                        if self.context_compressor and context:
                            current_tokens = self.context_compressor.calculate_tokens(context)
                            if current_tokens > 1500:  # 압축 임계값
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
                            # Phase 2: 개인화 및 지능형 분석 정보
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        stats = self.performance_monitor.get_stats()
        memory_usage = self.memory_optimizer.monitor_memory_usage()
        
        status = {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "memory_usage_percent": memory_usage,
            "performance_stats": stats,
            "timestamp": datetime.now().isoformat(),
            "prompt_system": {
                "unified_manager_available": self.unified_prompt_manager is not None,
                "dynamic_updater_available": self.dynamic_prompt_updater is not None,
                "prompt_optimizer_available": self.prompt_optimizer is not None,
                "prompt_analytics": self.unified_prompt_manager.get_prompt_analytics() if self.unified_prompt_manager else {},
                "optimization_recommendations": self.prompt_optimizer.get_optimization_recommendations() if self.prompt_optimizer else []
            }
        }
        
        # ChatService 상태 추가
        if self.chat_service:
            try:
                chat_status = self.chat_service.get_service_status()
                status["chat_service"] = chat_status
            except Exception as e:
                status["chat_service_error"] = str(e)
        
        # Phase 1: 대화 맥락 강화 상태 추가
        try:
            phase1_status = {
                "session_manager_available": self.session_manager is not None,
                "multi_turn_handler_available": self.multi_turn_handler is not None,
                "context_compressor_available": self.context_compressor is not None,
                "current_session_id": self.current_session_id,
                "current_user_id": self.current_user_id
            }
            
            # 세션 통계 추가
            if self.session_manager:
                session_stats = self.session_manager.get_session_stats()
                phase1_status["session_stats"] = session_stats
            
            status["phase1_context_enhancement"] = phase1_status
            
        except Exception as e:
            status["phase1_error"] = str(e)
        
        # Phase 2: 개인화 및 지능형 분석 상태 추가
        try:
            phase2_status = {
                "user_profile_manager_available": self.user_profile_manager is not None,
                "emotion_intent_analyzer_available": self.emotion_intent_analyzer is not None,
                "conversation_flow_tracker_available": self.conversation_flow_tracker is not None,
                "current_user_id": self.current_user_id
            }
            
            # 사용자 프로필 통계 추가
            if self.user_profile_manager:
                try:
                    user_stats = self.user_profile_manager.get_user_statistics(self.current_user_id)
                    phase2_status["user_profile_stats"] = user_stats
                except Exception as e:
                    phase2_status["user_profile_error"] = str(e)
            
            status["phase2_personalization_analysis"] = phase2_status
            
        except Exception as e:
            status["phase2_error"] = str(e)
        
        # Phase 3: 장기 기억 및 품질 모니터링 상태 추가
        try:
            phase3_status = {
                "contextual_memory_manager_available": self.contextual_memory_manager is not None,
                "quality_monitor_available": self.quality_monitor is not None,
                "current_user_id": self.current_user_id
            }
            
            # 메모리 통계 추가
            if self.contextual_memory_manager:
                try:
                    memory_stats = self.contextual_memory_manager.get_memory_statistics(self.current_user_id)
                    phase3_status["memory_stats"] = memory_stats
                except Exception as e:
                    phase3_status["memory_error"] = str(e)
            
            # 품질 대시보드 데이터 추가
            if self.quality_monitor:
                try:
                    quality_dashboard = self.quality_monitor.get_quality_dashboard_data(self.current_user_id)
                    phase3_status["quality_dashboard"] = quality_dashboard
                except Exception as e:
                    phase3_status["quality_error"] = str(e)
            
            status["phase3_memory_quality"] = phase3_status
            
        except Exception as e:
            status["phase3_error"] = str(e)
        
        return status

# 전역 앱 인스턴스
app_instance = HuggingFaceSpacesApp()

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 컴포넌트 초기화
    if not app_instance.initialize_components():
        logger.error("Failed to initialize components")
    
    # 커스텀 CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    
    /* 스트림 모드용 CSS */
    .stream-toggle {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    
    .stream-toggle:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* 스트림 애니메이션 */
    @keyframes stream-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .streaming {
        animation: stream-pulse 1.5s infinite !important;
    }
    
    /* 타이핑 효과 */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    .typing-effect {
        overflow: hidden;
        white-space: nowrap;
        animation: typing 2s steps(40, end);
    }
    """
    
    # HTML 헤드에 매니페스트 및 메타 태그 추가
    head_html = """
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="LawFirmAI">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """
    
    with gr.Blocks(
        css=css, 
        title="LawFirmAI - 법률 AI 어시스턴트",
        head=head_html
    ) as interface:
        gr.Markdown("""
        # ⚖️ LawFirmAI - 법률 AI 어시스턴트
        
        **Phase 1 완료**: 대화 맥락 강화, 다중 턴 질문 처리, 컨텍스트 압축, 영구적 세션 저장
        **Phase 2 완료**: 개인화 및 지능형 분석, 감정/의도 분석, 대화 흐름 추적, 사용자 프로필 관리
        
        법률 관련 질문에 정확하고 신뢰할 수 있는 답변을 제공합니다.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 탭 구조 추가
                with gr.Tabs():
                    with gr.Tab("💬 채팅"):
                        # 채팅 인터페이스
                        chatbot = gr.Chatbot(
                            label="법률 AI 어시스턴트",
                            height=500,
                            show_label=True,
                            type="messages"  # 최신 Gradio 형식 사용
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="법률 관련 질문을 입력하세요...",
                                label="질문",
                                scale=4
                            )
                            submit_btn = gr.Button("전송", scale=1, variant="primary")
                        
                        # 스트림 모드 토글
                        with gr.Row():
                            stream_mode = gr.Checkbox(label="🔄 스트림 모드 (실시간 답변)", value=True, elem_classes=["stream-toggle"])
                        
                        # 예시 질문
                        gr.Examples(
                            examples=[
                                "계약 해제 조건이 무엇인가요?",
                                "손해배상 관련 판례를 찾아주세요",
                                "불법행위의 법적 근거를 알려주세요",
                                "이혼 절차는 어떻게 진행하나요?",
                                "민법 제750조의 내용이 무엇인가요?"
                            ],
                            inputs=msg
                        )
                    
                    with gr.Tab("👤 사용자 프로필"):
                        with gr.Row():
                            with gr.Column():
                                user_id_input = gr.Textbox(
                                    label="사용자 ID",
                                    value="gradio_user",
                                    placeholder="사용자 ID를 입력하세요"
                                )
                                expertise_level = gr.Radio(
                                    choices=["beginner", "intermediate", "advanced", "expert"],
                                    value="beginner",
                                    label="전문성 수준"
                                )
                                detail_level = gr.Radio(
                                    choices=["simple", "medium", "detailed"],
                                    value="medium",
                                    label="답변 상세도"
                                )
                                interest_areas = gr.CheckboxGroup(
                                    choices=["민법", "형법", "상법", "근로기준법", "부동산", "금융", "지적재산권", "세법", "환경법", "의료법"],
                                    label="관심 분야",
                                    value=[]
                                )
                                save_profile_btn = gr.Button("프로필 저장", variant="primary")
                            
                            with gr.Column():
                                profile_status = gr.JSON(label="프로필 상태")
                                user_statistics = gr.JSON(label="사용자 통계")
                    
                    with gr.Tab("🧠 지능형 분석"):
                        with gr.Row():
                            with gr.Column():
                                emotion_analysis = gr.JSON(label="감정 분석")
                                intent_analysis = gr.JSON(label="의도 분석")
                                response_tone = gr.JSON(label="응답 톤")
                            
                            with gr.Column():
                                flow_tracking = gr.JSON(label="대화 흐름 추적")
                                suggested_questions = gr.JSON(label="제안된 후속 질문")
                                conversation_state = gr.Textbox(label="대화 상태", interactive=False)
                    
                    with gr.Tab("📊 대화 이력"):
                        session_list = gr.Dataframe(
                            label="세션 목록",
                            headers=["세션 ID", "생성일", "마지막 업데이트", "턴 수", "주제"],
                            interactive=False
                        )
                        load_session_btn = gr.Button("세션 불러오기")
                        session_details = gr.JSON(label="세션 상세 정보")
                    
                        with gr.Tab("🧠 장기 기억"):
                            with gr.Row():
                                with gr.Column():
                                    memory_search_query = gr.Textbox(
                                        label="메모리 검색",
                                        placeholder="검색할 내용을 입력하세요"
                                    )
                                    search_memory_btn = gr.Button("메모리 검색", variant="primary")
                                    memory_search_results = gr.JSON(label="검색 결과")
                                
                                with gr.Column():
                                    memory_statistics = gr.JSON(label="메모리 통계")
                                    consolidate_memory_btn = gr.Button("메모리 통합", variant="secondary")
                                    cleanup_memory_btn = gr.Button("오래된 메모리 정리", variant="secondary")
                        
                        with gr.Tab("📊 품질 모니터링"):
                            with gr.Row():
                                with gr.Column():
                                    quality_assessment = gr.JSON(label="대화 품질 평가")
                                    quality_trends = gr.JSON(label="품질 트렌드")
                                    quality_issues = gr.JSON(label="감지된 문제점")
                                
                                with gr.Column():
                                    improvement_suggestions = gr.JSON(label="개선 제안")
                                    quality_dashboard = gr.JSON(label="품질 대시보드")
                                    refresh_quality_btn = gr.Button("품질 분석 새로고침", variant="secondary")
                        
                        with gr.Tab("⚙️ 고급 설정"):
                            with gr.Row():
                                with gr.Column():
                                    show_follow_ups = gr.Checkbox(label="후속 질문 제안 표시", value=True)
                                    enable_emotion_analysis = gr.Checkbox(label="감정 분석 활성화", value=True)
                                    enable_flow_tracking = gr.Checkbox(label="대화 흐름 추적 활성화", value=True)
                                    enable_memory_management = gr.Checkbox(label="장기 기억 관리 활성화", value=True)
                                    enable_quality_monitoring = gr.Checkbox(label="품질 모니터링 활성화", value=True)
                                    max_context_turns = gr.Slider(
                                        minimum=5, maximum=20, value=10, step=1,
                                        label="최대 컨텍스트 턴 수"
                                    )
                                
                                with gr.Column():
                                    compression_threshold = gr.Slider(
                                        minimum=1000, maximum=5000, value=2000, step=100,
                                        label="컨텍스트 압축 임계값 (토큰)"
                                    )
                                    session_timeout = gr.Slider(
                                        minimum=1, maximum=24, value=24, step=1,
                                        label="세션 타임아웃 (시간)"
                                    )
                                    memory_retention_days = gr.Slider(
                                        minimum=7, maximum=90, value=30, step=1,
                                        label="메모리 보관 기간 (일)"
                                    )
                                    auto_backup = gr.Checkbox(label="자동 백업 활성화", value=True)
                    
                    # AKLS 표준판례 검색 탭 추가 (임시 비활성화)
                    # akls_tab = create_akls_interface()
            
            with gr.Column(scale=1):
                # 시스템 상태
                status_output = gr.JSON(
                    label="시스템 상태",
                    value=app_instance.get_system_status()
                )
                
                # 신뢰도 정보
                confidence_output = gr.JSON(
                    label="신뢰도 정보"
                )
                
                # 성능 통계
                performance_output = gr.JSON(
                    label="성능 통계"
                )
        
        # 이벤트 핸들러
        def respond(message, history):
            """응답 생성 (Phase 1 대화 맥락 강화 적용)"""
            if not message.strip():
                return history, "", {}
            
            # 질의 처리 (Phase 1 기능 포함)
            result = app_instance.process_query(message)
            
            # 메시지 형식 변환 (type="messages"에 맞게)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result["answer"]})
            
            # 신뢰도 정보 (Phase 1 및 Phase 2 정보 포함)
            confidence_info = {
                "신뢰도": f"{result['confidence']['confidence']:.1%}",
                "수준": result['confidence']['reliability_level'],
                "처리 시간": f"{result.get('processing_time', 0):.2f}초",
                "질문 유형": result.get('question_type', 'Unknown'),
                "세션 ID": result.get('session_id', 'Unknown'),
                "사용자 ID": result.get('user_id', 'Unknown')
            }
        
        def respond_stream(message, history):
            """스트림 응답 생성"""
            if not message.strip():
                return history, "", {}
            
            # 사용자 메시지 추가
            history.append({"role": "user", "content": message})
            
            # 스트림 응답 시뮬레이션
            import time
            
            # 초기 상태 메시지
            history.append({"role": "assistant", "content": "🔄 질문을 분석하고 있습니다..."})
            yield history, "", {}
            time.sleep(0.5)
            
            # 검색 상태 메시지
            history[-1] = {"role": "assistant", "content": "🔍 관련 법령과 판례를 검색하고 있습니다..."}
            yield history, "", {}
            time.sleep(0.5)
            
            # 답변 생성 상태 메시지
            history[-1] = {"role": "assistant", "content": "📝 답변을 생성하고 있습니다..."}
            yield history, "", {}
            time.sleep(0.5)
            
            # 실제 답변 생성
            result = app_instance.process_query(message)
            answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            # 답변을 단어별로 스트림
            words = answer.split()
            current_response = ""
            
            for i, word in enumerate(words):
                current_response += word + " "
                history[-1] = {"role": "assistant", "content": current_response.strip()}
                
                # 신뢰도 정보 업데이트
                confidence_info = {
                    "신뢰도": f"{result['confidence']['confidence']:.1%}",
                    "수준": result['confidence']['reliability_level'],
                    "처리 시간": f"{result.get('processing_time', 0):.2f}초",
                    "질문 유형": result.get('question_type', 'Unknown'),
                    "세션 ID": result.get('session_id', 'Unknown'),
                    "사용자 ID": result.get('user_id', 'Unknown'),
                    "스트림 모드": "활성화",
                    "진행률": f"{((i + 1) / len(words) * 100):.0f}%"
                }
                
                yield history, "", confidence_info
                time.sleep(0.1)  # 스트림 효과를 위한 지연
            
            # 다중 턴 질문 정보 추가
            if result.get('multi_turn_info', {}).get('is_multi_turn'):
                multi_turn = result['multi_turn_info']
                confidence_info["다중 턴 질문"] = "예"
                confidence_info["원본 질문"] = multi_turn.get('original_query', '')
                confidence_info["해결된 질문"] = multi_turn.get('resolved_query', '')
                confidence_info["해결 신뢰도"] = f"{multi_turn.get('confidence', 0):.1%}"
            else:
                confidence_info["다중 턴 질문"] = "아니오"
            
            # 컨텍스트 압축 정보 추가
            if result.get('compression_info', {}).get('compressed'):
                compression = result['compression_info']
                confidence_info["컨텍스트 압축"] = "예"
                confidence_info["압축 비율"] = f"{compression.get('compression_ratio', 0):.1%}"
                confidence_info["압축 요약"] = compression.get('summary', '')[:100] + "..."
            else:
                confidence_info["컨텍스트 압축"] = "아니오"
            
            # 컨텍스트 통계 추가
            context_stats = result.get('context_stats', {})
            confidence_info["총 대화 턴"] = context_stats.get('total_turns', 0)
            confidence_info["총 엔티티"] = context_stats.get('total_entities', 0)
            confidence_info["주제"] = ", ".join(context_stats.get('topics', []))
            
            # Phase 2: 개인화 및 지능형 분석 정보 추가
            personalized_context = result.get('personalized_context', {})
            if personalized_context:
                confidence_info["개인화 점수"] = f"{personalized_context.get('personalization_score', 0):.1%}"
                confidence_info["전문성 수준"] = personalized_context.get('expertise_level', 'Unknown')
                confidence_info["관심 분야"] = ", ".join(personalized_context.get('interest_areas', []))
            
            emotion_intent_info = result.get('emotion_intent_info', {})
            if emotion_intent_info:
                emotion = emotion_intent_info.get('emotion', {})
                intent = emotion_intent_info.get('intent', {})
                response_tone = emotion_intent_info.get('response_tone', {})
                
                confidence_info["감정"] = emotion.get('primary_emotion', 'Unknown')
                confidence_info["의도"] = intent.get('primary_intent', 'Unknown')
                confidence_info["긴급도"] = intent.get('urgency_level', 'Unknown')
                confidence_info["응답 톤"] = response_tone.get('tone_type', 'Unknown')
                confidence_info["공감 수준"] = f"{response_tone.get('empathy_level', 0):.1%}"
                confidence_info["격식 수준"] = f"{response_tone.get('formality_level', 0):.1%}"
            
            flow_tracking_info = result.get('flow_tracking_info', {})
            if flow_tracking_info:
                confidence_info["대화 상태"] = flow_tracking_info.get('conversation_state', 'Unknown')
                suggested_questions = flow_tracking_info.get('suggested_questions', [])
                confidence_info["제안된 질문 수"] = len(suggested_questions)
                if suggested_questions:
                    confidence_info["제안된 질문"] = suggested_questions[:3]  # 최대 3개만 표시
            
            # Phase 3: 장기 기억 및 품질 모니터링 정보 추가
            memory_search_results = result.get('memory_search_results', [])
            if memory_search_results:
                confidence_info["관련 메모리 수"] = len(memory_search_results)
                confidence_info["최고 관련성 점수"] = f"{max([m.get('relevance_score', 0) for m in memory_search_results]):.2f}"
            
            quality_assessment = result.get('quality_assessment', {})
            if quality_assessment:
                confidence_info["품질 점수"] = f"{quality_assessment.get('overall_score', 0):.2f}"
                confidence_info["완결성 점수"] = f"{quality_assessment.get('completeness_score', 0):.2f}"
                confidence_info["만족도 점수"] = f"{quality_assessment.get('satisfaction_score', 0):.2f}"
                confidence_info["정확성 점수"] = f"{quality_assessment.get('accuracy_score', 0):.2f}"
                
                issues = quality_assessment.get('issues', [])
                if issues:
                    confidence_info["감지된 문제점"] = issues[:3]  # 최대 3개만 표시
                
                suggestions = quality_assessment.get('suggestions', [])
                if suggestions:
                    confidence_info["개선 제안"] = suggestions[:2]  # 최대 2개만 표시
            
            # Phase별 상태 정보 추가
            phase_info = result.get('phase_info', {})
            if phase_info:
                phase1_status = phase_info.get('phase1', {})
                phase2_status = phase_info.get('phase2', {})
                phase3_status = phase_info.get('phase3', {})
                
                confidence_info["Phase 1 활성화"] = "예" if phase1_status.get('enabled') else "아니오"
                confidence_info["Phase 2 활성화"] = "예" if phase2_status.get('enabled') else "아니오"
                confidence_info["Phase 3 활성화"] = "예" if phase3_status.get('enabled') else "아니오"
                
                # 에러 정보 추가
                phase1_errors = phase1_status.get('errors', [])
                phase2_errors = phase2_status.get('errors', [])
                phase3_errors = phase3_status.get('errors', [])
                
                if phase1_errors:
                    confidence_info["Phase 1 에러"] = phase1_errors[0]  # 첫 번째 에러만 표시
                if phase2_errors:
                    confidence_info["Phase 2 에러"] = phase2_errors[0]
                if phase3_errors:
                    confidence_info["Phase 3 에러"] = phase3_errors[0]
            
            return history, "", confidence_info
        
        def update_status():
            """상태 업데이트"""
            return app_instance.get_system_status()
        
        def update_performance():
            """성능 통계 업데이트"""
            return app_instance.performance_monitor.get_stats()
        
        # Phase 2 기능들을 위한 이벤트 핸들러들
        def save_user_profile(user_id, expertise_level, detail_level, interest_areas):
            """사용자 프로필 저장"""
            try:
                profile_data = {
                    "expertise_level": expertise_level,
                    "preferred_detail_level": detail_level,
                    "interest_areas": interest_areas
                }
                
                success = app_instance.user_profile_manager.create_profile(user_id, profile_data)
                if success:
                    return {"status": "success", "message": "프로필이 저장되었습니다."}
                else:
                    return {"status": "error", "message": "프로필 저장에 실패했습니다."}
            except Exception as e:
                return {"status": "error", "message": f"오류: {str(e)}"}
        
        def get_user_statistics(user_id):
            """사용자 통계 조회"""
            try:
                stats = app_instance.user_profile_manager.get_user_statistics(user_id)
                return stats
            except Exception as e:
                return {"error": str(e)}
        
        def load_session_history():
            """세션 이력 로드"""
            try:
                sessions = app_instance.session_manager.get_user_sessions(app_instance.current_user_id, limit=10)
                session_data = []
                for session in sessions:
                    session_data.append([
                        session["session_id"],
                        session["created_at"],
                        session["last_updated"],
                        session.get("turn_count", 0),
                        ", ".join(session.get("topics", []))
                    ])
                return session_data
            except Exception as e:
                return []
        
        def get_session_details(session_id):
            """세션 상세 정보 조회"""
            try:
                session_data = app_instance.session_manager.conversation_store.load_session(session_id)
                return session_data
            except Exception as e:
                return {"error": str(e)}
        
        def search_memory(query):
            """메모리 검색"""
            try:
                if app_instance.contextual_memory_manager:
                    results = app_instance.contextual_memory_manager.retrieve_relevant_memory(
                        app_instance.current_session_id, query, app_instance.current_user_id
                    )
                    return [
                        {
                            "memory_id": result.memory.memory_id,
                            "content": result.memory.content,
                            "memory_type": result.memory.memory_type,
                            "importance_score": result.memory.importance_score,
                            "relevance_score": result.relevance_score,
                            "match_reason": result.match_reason
                        } for result in results[:10]  # 상위 10개만
                    ]
                else:
                    return []
            except Exception as e:
                return {"error": str(e)}
        
        def get_memory_statistics():
            """메모리 통계 조회"""
            try:
                if app_instance.contextual_memory_manager:
                    stats = app_instance.contextual_memory_manager.get_memory_statistics(app_instance.current_user_id)
                    return stats
                else:
                    return {"error": "Memory manager not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def consolidate_memories():
            """메모리 통합"""
            try:
                if app_instance.contextual_memory_manager:
                    count = app_instance.contextual_memory_manager.consolidate_memories(app_instance.current_user_id)
                    return {"status": "success", "consolidated_count": count}
                else:
                    return {"error": "Memory manager not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def cleanup_old_memories():
            """오래된 메모리 정리"""
            try:
                if app_instance.contextual_memory_manager:
                    count = app_instance.contextual_memory_manager.cleanup_old_memories(days=30)
                    return {"status": "success", "cleaned_count": count}
                else:
                    return {"error": "Memory manager not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def get_quality_assessment():
            """품질 평가 조회"""
            try:
                if app_instance.quality_monitor and app_instance.session_manager:
                    context = app_instance.session_manager.get_or_create_session(
                        app_instance.current_session_id, app_instance.current_user_id
                    )
                    assessment = app_instance.quality_monitor.assess_conversation_quality(context)
                    return assessment
                else:
                    return {"error": "Quality monitor not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def get_quality_trends():
            """품질 트렌드 조회"""
            try:
                if app_instance.quality_monitor:
                    # 최근 세션들 조회
                    sessions = app_instance.session_manager.get_user_sessions(app_instance.current_user_id, limit=10)
                    session_ids = [session["session_id"] for session in sessions]
                    trends = app_instance.quality_monitor.analyze_quality_trends(session_ids)
                    return trends
                else:
                    return {"error": "Quality monitor not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def get_quality_dashboard():
            """품질 대시보드 데이터 조회"""
            try:
                if app_instance.quality_monitor:
                    dashboard_data = app_instance.quality_monitor.get_quality_dashboard_data(app_instance.current_user_id)
                    return dashboard_data
                else:
                    return {"error": "Quality monitor not available"}
            except Exception as e:
                return {"error": str(e)}
        
        def detect_quality_issues():
            """품질 문제점 감지"""
            try:
                if app_instance.quality_monitor and app_instance.session_manager:
                    context = app_instance.session_manager.get_or_create_session(
                        app_instance.current_session_id, app_instance.current_user_id
                    )
                    issues = app_instance.quality_monitor.detect_conversation_issues(context)
                    suggestions = app_instance.quality_monitor.suggest_improvements(context)
                    return {
                        "issues": issues,
                        "suggestions": suggestions
                    }
                else:
                    return {"error": "Quality monitor not available"}
            except Exception as e:
                return {"error": str(e)}
        
        # 이벤트 연결
        def handle_submit(message, history, use_stream):
            """제출 처리 (스트림/일반 모드 선택)"""
            if use_stream:
                return respond_stream(message, history)
            else:
                return respond(message, history)
        
        submit_btn.click(
            handle_submit,
            inputs=[msg, chatbot, stream_mode],
            outputs=[chatbot, msg, confidence_output]
        )
        
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot, stream_mode],
            outputs=[chatbot, msg, confidence_output]
        )
        
        # Phase 2 이벤트 연결
        save_profile_btn.click(
            save_user_profile,
            inputs=[user_id_input, expertise_level, detail_level, interest_areas],
            outputs=[profile_status]
        )
        
        user_id_input.change(
            get_user_statistics,
            inputs=[user_id_input],
            outputs=[user_statistics]
        )
        
        load_session_btn.click(
            load_session_history,
            outputs=[session_list]
        )
        
        session_list.select(
            get_session_details,
            inputs=[session_list],
            outputs=[session_details]
        )
        
        # Phase 3 이벤트 연결
        search_memory_btn.click(
            search_memory,
            inputs=[memory_search_query],
            outputs=[memory_search_results]
        )
        
        consolidate_memory_btn.click(
            consolidate_memories,
            outputs=[memory_statistics]
        )
        
        cleanup_memory_btn.click(
            cleanup_old_memories,
            outputs=[memory_statistics]
        )
        
        refresh_quality_btn.click(
            get_quality_assessment,
            outputs=[quality_assessment]
        )
        
        # 자동 로드 이벤트들
        user_id_input.change(
            get_memory_statistics,
            outputs=[memory_statistics]
        )
        
        user_id_input.change(
            get_quality_dashboard,
            outputs=[quality_dashboard]
        )
        
        # 상태 업데이트 버튼 추가 (주기적 업데이트 대신 수동 업데이트)
        with gr.Row():
            refresh_status_btn = gr.Button("상태 새로고침", variant="secondary")
            refresh_performance_btn = gr.Button("성능 통계 새로고침", variant="secondary")
        
        # 수동 상태 업데이트 이벤트
        refresh_status_btn.click(
            update_status,
            outputs=status_output
        )
        
        refresh_performance_btn.click(
            update_performance,
            outputs=performance_output
        )
    
        return interface
    
    def get_performance_dashboard(self):
        """성능 모니터링 대시보드"""
        with gr.Blocks(title="성능 모니터링 대시보드") as dashboard:
            gr.Markdown("# LawFirmAI 성능 모니터링 대시보드")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 시스템 상태")
                    system_health = gr.JSON(label="시스템 상태", value=self.performance_monitor.get_system_health())
                    
                    refresh_btn = gr.Button("새로고침", variant="secondary")
                    
                with gr.Column():
                    gr.Markdown("## 성능 요약 (최근 24시간)")
                    performance_summary = gr.JSON(label="성능 요약", value=self.performance_monitor.get_performance_summary())
            
            with gr.Row():
                gr.Markdown("## 성능 메트릭 내보내기")
                export_btn = gr.Button("메트릭 내보내기", variant="primary")
                export_status = gr.Textbox(label="내보내기 상태", interactive=False)
            
            def refresh_data():
                return (
                    self.performance_monitor.get_system_health(),
                    self.performance_monitor.get_performance_summary()
                )
            
            def export_metrics():
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"performance_metrics_{timestamp}.json"
                    self.performance_monitor.export_metrics(filepath)
                    return f"메트릭이 {filepath}에 저장되었습니다."
                except Exception as e:
                    return f"내보내기 실패: {str(e)}"
            
            refresh_btn.click(
                fn=refresh_data,
                outputs=[system_health, performance_summary]
            )
            
            export_btn.click(
                fn=export_metrics,
                outputs=[export_status]
            )
        
        return dashboard

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI HuggingFace Spaces application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # 안정적인 실행 설정
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=40,
        # 공유 기능 완전 비활성화
        show_api=False,
        # 정적 파일 서빙 설정
        favicon_path="gradio/static/favicon.ico" if os.path.exists("gradio/static/favicon.ico") else None
    )

if __name__ == "__main__":
    main()
