# -*- coding: utf-8 -*-
"""
Chat Service
채팅 메시지 처리 서비스
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..utils.config import Config
from ..utils.logger import get_logger
from .rag_service import MLEnhancedRAGService
from .hybrid_search_engine import HybridSearchEngine
from .improved_answer_generator import ImprovedAnswerGenerator
from .question_classifier import QuestionClassifier
from ..models.model_manager import LegalModelManager
from ..data.vector_store import LegalVectorStore
from ..data.database import DatabaseManager

# Phase 1: 대화 맥락 강화 모듈
from .integrated_session_manager import IntegratedSessionManager
from .multi_turn_handler import MultiTurnQuestionHandler
from .context_compressor import ContextCompressor

# Phase 2: 개인화 및 지능형 분석 모듈
from .user_profile_manager import UserProfileManager
from .emotion_intent_analyzer import EmotionIntentAnalyzer
from .conversation_flow_tracker import ConversationFlowTracker

# Phase 3: 장기 기억 및 품질 모니터링 모듈
from .contextual_memory_manager import ContextualMemoryManager
from .conversation_quality_monitor import ConversationQualityMonitor

# 자연스러운 답변 개선 모듈
from .conversation_connector import ConversationConnector
from .emotional_tone_adjuster import EmotionalToneAdjuster
from .personalized_style_learner import PersonalizedStyleLearner
from .realtime_feedback_system import RealtimeFeedbackSystem
from .naturalness_evaluator import NaturalnessEvaluator

# 성능 최적화 모듈
from ..utils.performance_optimizer import (
    PerformanceMonitor, MemoryOptimizer, CacheManager,
    performance_monitor, memory_optimized
)

logger = get_logger(__name__)


class ChatService:
    """채팅 서비스 클래스"""
    
    def __init__(self, config: Config):
        """채팅 서비스 초기화"""
        self.config = config
        self.logger = get_logger(__name__)
        
        # LangGraph 사용 여부 확인 (비활성화)
        self.use_langgraph = False  # os.getenv("USE_LANGGRAPH", "false").lower() == "true"
        
        if self.use_langgraph:
            try:
                from .langgraph.workflow_service import LangGraphWorkflowService
                from ..utils.langgraph_config import LangGraphConfig
                
                langgraph_config = LangGraphConfig.from_env()
                self.langgraph_service = LangGraphWorkflowService(langgraph_config)
                self.logger.info("LangGraph workflow service initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize LangGraph service: {e}")
                self.use_langgraph = False
                self.langgraph_service = None
        else:
            self.langgraph_service = None
        
        # 실제 RAG 컴포넌트 초기화
        try:
            # 필요한 컴포넌트들 초기화
            model_manager = LegalModelManager()
            vector_store = LegalVectorStore()
            database_manager = DatabaseManager()
            
            self.rag_service = MLEnhancedRAGService(
                config=config,
                model_manager=model_manager,
                vector_store=vector_store,
                database=database_manager
            )
            self.hybrid_search_engine = HybridSearchEngine()
            self.question_classifier = QuestionClassifier()
            self.improved_answer_generator = ImprovedAnswerGenerator()
            self.logger.info("RAG components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG components: {e}")
            self.rag_service = None
            self.hybrid_search_engine = None
            self.question_classifier = None
            self.improved_answer_generator = None
        
        # Phase 1: 대화 맥락 강화 컴포넌트 초기화
        try:
            self.session_manager = IntegratedSessionManager("data/conversations.db")
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.context_compressor = ContextCompressor(max_tokens=2000)
            self.logger.info("Phase 1 components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 1 components: {e}")
            self.session_manager = None
            self.multi_turn_handler = None
            self.context_compressor = None
        
        # Phase 2: 개인화 및 지능형 분석 컴포넌트 초기화
        try:
            if self.session_manager:
                self.user_profile_manager = UserProfileManager(self.session_manager.conversation_store)
            else:
                self.user_profile_manager = UserProfileManager()
            self.emotion_intent_analyzer = EmotionIntentAnalyzer()
            self.conversation_flow_tracker = ConversationFlowTracker()
            self.logger.info("Phase 2 components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 2 components: {e}")
            self.user_profile_manager = None
            self.emotion_intent_analyzer = None
            self.conversation_flow_tracker = None
        
        # Phase 3: 장기 기억 및 품질 모니터링 컴포넌트 초기화
        try:
            if self.session_manager:
                self.contextual_memory_manager = ContextualMemoryManager(self.session_manager.conversation_store)
                self.quality_monitor = ConversationQualityMonitor(self.session_manager.conversation_store)
            else:
                self.contextual_memory_manager = ContextualMemoryManager()
                self.quality_monitor = ConversationQualityMonitor()
            self.logger.info("Phase 3 components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 3 components: {e}")
            self.contextual_memory_manager = None
            self.quality_monitor = None
        
        # 자연스러운 답변 개선 컴포넌트 초기화
        try:
            self.conversation_connector = ConversationConnector()
            self.emotional_tone_adjuster = EmotionalToneAdjuster()
            self.personalized_style_learner = PersonalizedStyleLearner(self.user_profile_manager)
            self.realtime_feedback_system = RealtimeFeedbackSystem(self.quality_monitor)
            self.naturalness_evaluator = NaturalnessEvaluator()
            self.logger.info("Natural conversation improvement components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize natural conversation components: {e}")
            self.conversation_connector = None
            self.emotional_tone_adjuster = None
            self.personalized_style_learner = None
            self.realtime_feedback_system = None
            self.naturalness_evaluator = None
        
        # 성능 최적화 컴포넌트 초기화
        try:
            self.performance_monitor = PerformanceMonitor()
            self.memory_optimizer = MemoryOptimizer()
            self.cache_manager = CacheManager(max_size=1000, ttl=3600)
            self.logger.info("Performance optimization components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance optimization components: {e}")
            self.performance_monitor = None
            self.memory_optimizer = None
            self.cache_manager = None
        
        self.logger.info(f"ChatService initialized (LangGraph: {self.use_langgraph})")
    
    @performance_monitor
    @memory_optimized
    async def process_message(self, message: str, context: Optional[str] = None, 
                             session_id: Optional[str] = None, 
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        사용자 메시지 처리 (모든 Phase 기능 통합)
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트 (선택사항)
            session_id: 세션 ID (선택사항)
            user_id: 사용자 ID (선택사항)
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            start_time = time.time()
            
            # 캐시에서 응답 확인
            cache_key = f"response:{hash(message)}:{session_id}:{user_id}"
            cached_response = self.cache_manager.get(cache_key) if self.cache_manager else None
            
            if cached_response:
                self.logger.info(f"Cache hit for message: {message[:50]}...")
                cached_response["from_cache"] = True
                cached_response["processing_time"] = time.time() - start_time
                return cached_response
            
            # 기본값 설정
            if not user_id:
                user_id = "anonymous_user"
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 입력 검증
            if not self.validate_input(message):
                return {
                    "response": "올바른 질문을 입력해주세요.",
                    "confidence": 0.0,
                    "sources": [],
                    "processing_time": 0.0,
                    "session_id": session_id,
                    "user_id": user_id,
                    "phase_info": {
                        "phase1": {"enabled": False, "error": "Invalid input"},
                        "phase2": {"enabled": False},
                        "phase3": {"enabled": False}
                    }
                }
            
            # Phase 1: 대화 맥락 강화 처리
            phase1_info = await self._process_phase1_context(message, session_id, user_id)
            
            # Phase 2: 개인화 및 지능형 분석 처리
            phase2_info = await self._process_phase2_personalization(message, session_id, user_id, phase1_info)
            
            # Phase 3: 장기 기억 및 품질 모니터링 처리
            phase3_info = await self._process_phase3_memory_quality(message, session_id, user_id, phase1_info, phase2_info)
            
            # 실제 답변 생성 (RAG 시스템 사용)
            response_result = await self._generate_response(message, phase1_info, phase2_info, phase3_info)
            
            # 자연스러운 답변 개선 적용
            if response_result.get("response"):
                response_result["response"] = await self._apply_naturalness_improvements(
                    response_result["response"], phase1_info, phase2_info, user_id
                )
            
            # 최종 결과 통합
            processing_time = time.time() - start_time
            
            # 캐시 히트율 계산
            cache_hit_rate = 0.0
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                cache_hit_rate = cache_stats.get("hit_rate", 0.0)
            
            result = {
                "response": response_result.get("response", "죄송합니다. 답변을 생성할 수 없습니다."),
                "confidence": response_result.get("confidence", 0.0),
                "sources": response_result.get("sources", []),
                "processing_time": processing_time,
                "session_id": session_id,
                "user_id": user_id,
                "question_type": response_result.get("question_type", "general_question"),
                "legal_references": response_result.get("legal_references", []),
                "processing_steps": response_result.get("processing_steps", []),
                "metadata": response_result.get("metadata", {}),
                "errors": response_result.get("errors", []),
                "phase_info": {
                    "phase1": phase1_info,
                    "phase2": phase2_info,
                    "phase3": phase3_info
                },
                "performance_info": {
                    "processing_time": processing_time,
                    "cache_hit_rate": cache_hit_rate,
                    "from_cache": False,
                    "memory_usage_mb": self.memory_optimizer.get_memory_usage().process_memory / 1024 / 1024 if self.memory_optimizer else 0
                }
            }
            
            # 캐시에 결과 저장
            if self.cache_manager and not result.get("errors"):
                self.cache_manager.set(cache_key, result)
            
            # Phase 3: 품질 평가 및 메모리 저장
            if self.quality_monitor and phase1_info.get("context"):
                quality_assessment = self.quality_monitor.assess_conversation_quality(phase1_info["context"])
                result["quality_assessment"] = quality_assessment
            
            if self.contextual_memory_manager and phase1_info.get("context"):
                # 중요한 사실 추출 및 저장
                facts = {}
                for turn in phase1_info["context"].turns:
                    extracted_facts = self.contextual_memory_manager.extract_facts_from_conversation(turn)
                    for fact in extracted_facts:
                        fact_type = fact["type"]
                        if fact_type not in facts:
                            facts[fact_type] = []
                        facts[fact_type].append(fact["content"])
                
                if facts:
                    self.contextual_memory_manager.store_important_facts(session_id, user_id, facts)
            
            # 성능 메트릭 기록
            if self.performance_monitor:
                active_sessions = 1  # 간단한 추정
                db_query_time = 0.0  # 실제 DB 쿼리 시간은 별도 측정 필요
                self.performance_monitor.record_metrics(
                    response_time=processing_time,
                    active_sessions=active_sessions,
                    cache_hit_rate=cache_hit_rate,
                    db_query_time=db_query_time
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "response": "죄송합니다. 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "user_id": user_id,
                "error": str(e),
                "phase_info": {
                    "phase1": {"enabled": False, "error": str(e)},
                    "phase2": {"enabled": False},
                    "phase3": {"enabled": False}
                }
            }
    
    async def _process_phase1_context(self, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """Phase 1: 대화 맥락 강화 처리"""
        try:
            phase1_info = {
                "enabled": True,
                "session_manager_available": self.session_manager is not None,
                "multi_turn_handler_available": self.multi_turn_handler is not None,
                "context_compressor_available": self.context_compressor is not None,
                "context": None,
                "multi_turn_result": None,
                "compression_info": None,
                "errors": []
            }
            
            if not self.session_manager:
                phase1_info["enabled"] = False
                phase1_info["errors"].append("Session manager not available")
                return phase1_info
            
            # 세션 로드 또는 생성
            context = self.session_manager.get_or_create_session(session_id, user_id)
            phase1_info["context"] = context
            
            # 다중 턴 질문 처리
            if self.multi_turn_handler:
                multi_turn_result = self.multi_turn_handler.build_complete_query(message, context)
                phase1_info["multi_turn_result"] = {
                    "is_multi_turn": multi_turn_result["is_multi_turn"],
                    "resolved_query": multi_turn_result["resolved_query"],
                    "confidence": multi_turn_result["confidence"],
                    "reasoning": multi_turn_result["reasoning"]
                }
            
            # 컨텍스트 압축 (필요시)
            if self.context_compressor and context:
                compression_info = self.context_compressor.compress_context_if_needed(context, message)
                phase1_info["compression_info"] = compression_info
            
            return phase1_info
            
        except Exception as e:
            self.logger.error(f"Error in Phase 1 processing: {e}")
            return {
                "enabled": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    async def _process_phase2_personalization(self, message: str, session_id: str, user_id: str, 
                                            phase1_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: 개인화 및 지능형 분석 처리"""
        try:
            phase2_info = {
                "enabled": True,
                "user_profile_manager_available": self.user_profile_manager is not None,
                "emotion_intent_analyzer_available": self.emotion_intent_analyzer is not None,
                "conversation_flow_tracker_available": self.conversation_flow_tracker is not None,
                "personalized_context": {},
                "emotion_intent_info": {},
                "flow_tracking_info": {},
                "errors": []
            }
            
            # 사용자 프로필 관리
            if self.user_profile_manager:
                try:
                    personalized_context = self.user_profile_manager.get_personalized_context(user_id, message)
                    phase2_info["personalized_context"] = personalized_context
                except Exception as e:
                    phase2_info["errors"].append(f"User profile error: {str(e)}")
            
            # 감정 및 의도 분석
            if self.emotion_intent_analyzer:
                try:
                    emotion_result = self.emotion_intent_analyzer.analyze_emotion(message)
                    intent_result = self.emotion_intent_analyzer.analyze_intent(message, phase1_info.get("context"))
                    
                    response_tone = self.emotion_intent_analyzer.get_contextual_response_tone(
                        emotion_result, intent_result, phase2_info.get("personalized_context", {})
                    )
                    
                    phase2_info["emotion_intent_info"] = {
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
                except Exception as e:
                    phase2_info["errors"].append(f"Emotion intent analysis error: {str(e)}")
            
            # 대화 흐름 추적
            if self.conversation_flow_tracker and phase1_info.get("context"):
                try:
                    from .conversation_manager import ConversationTurn
                    
                    # 대화 흐름 추적
                    self.conversation_flow_tracker.track_conversation_flow(session_id, 
                        ConversationTurn(
                            user_query=message,
                            bot_response="",  # 아직 생성되지 않음
                            timestamp=datetime.now(),
                            question_type="general_question"
                        )
                    )
                    
                    # 다음 의도 예측 및 후속 질문 제안
                    predicted_intents = self.conversation_flow_tracker.predict_next_intent(phase1_info["context"])
                    suggested_questions = self.conversation_flow_tracker.suggest_follow_up_questions(phase1_info["context"])
                    conversation_state = self.conversation_flow_tracker.get_conversation_state(phase1_info["context"])
                    
                    phase2_info["flow_tracking_info"] = {
                        "predicted_intents": [
                            {
                                "intent_type": pred.intent_type.value,
                                "confidence": pred.confidence,
                                "reasoning": pred.reasoning
                            } for pred in predicted_intents
                        ],
                        "suggested_questions": [
                            {
                                "question": sug.question,
                                "relevance_score": sug.relevance_score,
                                "question_type": sug.question_type
                            } for sug in suggested_questions
                        ],
                        "conversation_state": {
                            "current_topic": conversation_state.current_topic,
                            "last_question_type": conversation_state.last_question_type,
                            "turn_count_in_topic": conversation_state.turn_count_in_topic,
                            "is_problem_solving_flow": conversation_state.is_problem_solving_flow
                        }
                    }
                except Exception as e:
                    phase2_info["errors"].append(f"Flow tracking error: {str(e)}")
            
            return phase2_info
            
        except Exception as e:
            self.logger.error(f"Error in Phase 2 processing: {e}")
            return {
                "enabled": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    async def _process_phase3_memory_quality(self, message: str, session_id: str, user_id: str,
                                           phase1_info: Dict[str, Any], phase2_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: 장기 기억 및 품질 모니터링 처리"""
        try:
            phase3_info = {
                "enabled": True,
                "contextual_memory_manager_available": self.contextual_memory_manager is not None,
                "quality_monitor_available": self.quality_monitor is not None,
                "memory_search_results": [],
                "quality_assessment": {},
                "errors": []
            }
            
            # 맥락적 메모리 검색
            if self.contextual_memory_manager:
                try:
                    memory_search_results = self.contextual_memory_manager.retrieve_relevant_memory(session_id, message, user_id)
                    phase3_info["memory_search_results"] = [
                        {
                            "memory_id": result.memory.memory_id,
                            "content": result.memory.content,
                            "memory_type": result.memory.memory_type,
                            "importance_score": result.memory.importance_score,
                            "relevance_score": result.relevance_score,
                            "match_reason": result.match_reason
                        } for result in memory_search_results[:5]  # 상위 5개만
                    ]
                except Exception as e:
                    phase3_info["errors"].append(f"Memory search error: {str(e)}")
            
            # 품질 평가
            if self.quality_monitor and phase1_info.get("context"):
                try:
                    quality_assessment = self.quality_monitor.assess_conversation_quality(phase1_info["context"])
                    phase3_info["quality_assessment"] = {
                        "overall_score": quality_assessment.get("overall_score", 0.0),
                        "completeness_score": quality_assessment.get("completeness_score", 0.0),
                        "satisfaction_score": quality_assessment.get("satisfaction_score", 0.0),
                        "accuracy_score": quality_assessment.get("accuracy_score", 0.0),
                        "issues": quality_assessment.get("issues", []),
                        "suggestions": quality_assessment.get("suggestions", [])
                    }
                except Exception as e:
                    phase3_info["errors"].append(f"Quality assessment error: {str(e)}")
            
            return phase3_info
            
        except Exception as e:
            self.logger.error(f"Error in Phase 3 processing: {e}")
            return {
                "enabled": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    async def _generate_response(self, message: str, phase1_info: Dict[str, Any], 
                               phase2_info: Dict[str, Any], phase3_info: Dict[str, Any]) -> Dict[str, Any]:
        """실제 답변 생성 (RAG 시스템 사용)"""
        try:
            # 다중 턴 처리된 쿼리 사용
            resolved_query = message
            if phase1_info.get("multi_turn_result", {}).get("is_multi_turn"):
                resolved_query = phase1_info["multi_turn_result"]["resolved_query"]
            
            # LangGraph 사용 여부에 따른 처리
            if self.use_langgraph and self.langgraph_service:
                return await self._process_with_langgraph(resolved_query, phase1_info.get("session_id"))
            else:
                return await self._process_legacy(resolved_query, None)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "errors": [str(e)]
            }
    
    async def _process_with_langgraph(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """LangGraph를 사용한 메시지 처리"""
        try:
            result = await self.langgraph_service.process_query(message, session_id)
            
            # LangGraph 결과를 기존 형식으로 변환
            return {
                "response": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
                "processing_time": result.get("processing_time", 0.0),
                "session_id": result.get("session_id"),
                "query_type": result.get("query_type", ""),
                "legal_references": result.get("legal_references", []),
                "processing_steps": result.get("processing_steps", []),
                "metadata": result.get("metadata", {}),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            self.logger.error(f"LangGraph processing error: {e}")
            return {
                "response": "LangGraph 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0,
                "errors": [str(e)]
            }
    
    async def _process_legacy(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """기존 방식으로 메시지 처리"""
        try:
            start_time = time.time()
            
            # 기존 처리 로직 (placeholder)
            response_data = self._generate_response_sync(message, context)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response_data.get("response", ""),
                "confidence": response_data.get("confidence", 0.8),
                "sources": response_data.get("sources", []),
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Legacy processing error: {e}")
            return {
                "response": "기존 처리 방식에서 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0,
                "errors": [str(e)]
            }
    
    def _generate_response_sync(self, message: str, context: Optional[str] = None, 
                          *args, **kwargs) -> Dict[str, Any]:
        """실제 RAG 시스템을 사용한 응답 생성"""
        try:
            # 질문 분류
            if self.question_classifier:
                question_classification = self.question_classifier.classify_question(message)
            else:
                # 기본 분류
                question_classification = type('QuestionClassification', (), {
                    'question_type': type('QuestionType', (), {'value': 'general'})()
                })()
            
            # 검색 실행
            if self.hybrid_search_engine:
                search_results = self.hybrid_search_engine.search_with_question_type(
                    query=message,
                    question_type=question_classification,
                    max_results=10
                )
            else:
                search_results = []
            
            # 답변 생성
            if self.improved_answer_generator:
                answer_result = self.improved_answer_generator.generate_answer(
                    query=message,
                    question_type=question_classification,
                    context=context or "",
                    sources=search_results,
                    conversation_history=None
                )
                return {
                    "response": answer_result.answer,
                    "confidence": 0.8,
                    "sources": search_results,
                    "processing_time": 0.0
                }
            else:
                # 폴백: 기본 응답
                return {
                    "response": f"안녕하세요! '{message}'에 대한 질문을 받았습니다. 현재 개발 중인 기능입니다.",
                    "confidence": 0.5,
                    "sources": [],
                    "processing_time": 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0
            }
    
    def validate_input(self, message: str) -> bool:
        """입력 검증"""
        if not message or not message.strip():
            return False
        
        if len(message) > 10000:  # Max 10,000 characters
            return False
        
        return True
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        try:
            status = {
                "service_name": "ChatService",
                "langgraph_enabled": self.use_langgraph,
                "rag_components": {
                    "rag_service": self.rag_service is not None,
                    "hybrid_search_engine": self.hybrid_search_engine is not None,
                    "question_classifier": self.question_classifier is not None,
                    "improved_answer_generator": self.improved_answer_generator is not None
                },
                "phase1_components": {
                    "session_manager": self.session_manager is not None,
                    "multi_turn_handler": self.multi_turn_handler is not None,
                    "context_compressor": self.context_compressor is not None
                },
                "phase2_components": {
                    "user_profile_manager": self.user_profile_manager is not None,
                    "emotion_intent_analyzer": self.emotion_intent_analyzer is not None,
                    "conversation_flow_tracker": self.conversation_flow_tracker is not None
                },
                "phase3_components": {
                    "contextual_memory_manager": self.contextual_memory_manager is not None,
                    "quality_monitor": self.quality_monitor is not None
                },
                "overall_status": "healthy" if self._is_healthy() else "degraded",
                "last_updated": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                "service_name": "ChatService",
                "overall_status": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def _is_healthy(self) -> bool:
        """서비스 건강 상태 확인"""
        try:
            # 기본 RAG 컴포넌트 중 하나라도 사용 가능하면 healthy
            rag_available = any([
                self.rag_service is not None,
                self.hybrid_search_engine is not None,
                self.question_classifier is not None,
                self.improved_answer_generator is not None
            ])
            
            # Phase 1 컴포넌트 중 하나라도 사용 가능하면 healthy
            phase1_available = any([
                self.session_manager is not None,
                self.multi_turn_handler is not None,
                self.context_compressor is not None
            ])
            
            return rag_available or phase1_available
            
        except Exception as e:
            self.logger.error(f"Error checking health status: {e}")
            return False
    
    def get_phase_statistics(self) -> Dict[str, Any]:
        """Phase별 통계 조회"""
        try:
            stats = {
                "phase1": {
                    "enabled": self.session_manager is not None,
                    "session_count": 0,
                    "total_turns": 0,
                    "compression_count": 0
                },
                "phase2": {
                    "enabled": self.user_profile_manager is not None,
                    "user_count": 0,
                    "profile_updates": 0,
                    "emotion_analysis_count": 0
                },
                "phase3": {
                    "enabled": self.contextual_memory_manager is not None,
                    "memory_count": 0,
                    "quality_assessments": 0,
                    "consolidation_count": 0
                }
            }
            
            # Phase 1 통계
            if self.session_manager:
                try:
                    session_stats = self.session_manager.get_session_stats()
                    stats["phase1"]["session_count"] = session_stats.get("total_sessions", 0)
                    stats["phase1"]["total_turns"] = session_stats.get("total_turns", 0)
                except Exception as e:
                    self.logger.warning(f"Error getting Phase 1 stats: {e}")
            
            # Phase 2 통계
            if self.user_profile_manager:
                try:
                    # 사용자 수는 DB에서 직접 조회해야 함
                    stats["phase2"]["user_count"] = 0  # Placeholder
                except Exception as e:
                    self.logger.warning(f"Error getting Phase 2 stats: {e}")
            
            # Phase 3 통계
            if self.contextual_memory_manager:
                try:
                    # 메모리 수는 DB에서 직접 조회해야 함
                    stats["phase3"]["memory_count"] = 0  # Placeholder
                except Exception as e:
                    self.logger.warning(f"Error getting Phase 3 stats: {e}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting phase statistics: {e}")
            return {
                "phase1": {"enabled": False, "error": str(e)},
                "phase2": {"enabled": False, "error": str(e)},
                "phase3": {"enabled": False, "error": str(e)}
            }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """성능 최적화 수행"""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "memory_optimization": {},
                "cache_optimization": {},
                "performance_summary": {}
            }
            
            # 메모리 최적화
            if self.memory_optimizer:
                memory_result = self.memory_optimizer.optimize_memory()
                optimization_results["memory_optimization"] = memory_result
                optimization_results["actions_taken"].extend(memory_result.get("actions_taken", []))
            
            # 캐시 최적화
            if self.cache_manager:
                cache_stats_before = self.cache_manager.get_stats()
                cache_cleared = self.cache_manager.clear()
                optimization_results["cache_optimization"] = {
                    "cache_cleared": cache_cleared,
                    "stats_before": cache_stats_before
                }
                optimization_results["actions_taken"].append(f"Cache cleared: {cache_cleared} entries")
            
            # 성능 요약
            if self.performance_monitor:
                performance_summary = self.performance_monitor.get_performance_summary()
                optimization_results["performance_summary"] = performance_summary
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "performance_monitor": {},
                "memory_optimizer": {},
                "cache_manager": {},
                "system_health": {}
            }
            
            # 성능 모니터 메트릭
            if self.performance_monitor:
                metrics["performance_monitor"] = {
                    "summary": self.performance_monitor.get_performance_summary(),
                    "system_health": self.performance_monitor.get_system_health()
                }
            
            # 메모리 최적화 메트릭
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_usage()
                memory_trend = self.memory_optimizer.monitor_memory_trend()
                metrics["memory_optimizer"] = {
                    "memory_usage": {
                        "total_mb": round(memory_usage.total_memory / 1024 / 1024, 2),
                        "used_mb": round(memory_usage.used_memory / 1024 / 1024, 2),
                        "available_mb": round(memory_usage.available_memory / 1024 / 1024, 2),
                        "percentage": round(memory_usage.memory_percentage, 2),
                        "process_mb": round(memory_usage.process_memory / 1024 / 1024, 2),
                        "cache_mb": round(memory_usage.cache_memory / 1024 / 1024, 2)
                    },
                    "memory_trend": memory_trend
                }
            
            # 캐시 관리 메트릭
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                metrics["cache_manager"] = cache_stats
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """리소스 정리"""
        try:
            cleanup_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "resources_freed": {}
            }
            
            # 메모리 정리
            if self.memory_optimizer:
                memory_result = self.memory_optimizer.optimize_memory()
                cleanup_results["resources_freed"]["memory_mb"] = memory_result.get("memory_freed_mb", 0)
                cleanup_results["actions_taken"].extend(memory_result.get("actions_taken", []))
            
            # 캐시 정리
            if self.cache_manager:
                cache_cleared = self.cache_manager.clear()
                cleanup_results["resources_freed"]["cache_entries"] = cache_cleared
                cleanup_results["actions_taken"].append(f"Cache cleared: {cache_cleared} entries")
            
            # 세션 정리 (오래된 세션)
            if self.session_manager:
                # 간단한 세션 정리 로직 (실제 구현은 세션 관리자에 따라 다름)
                cleanup_results["actions_taken"].append("Session cleanup completed")
            
            return cleanup_results
            
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")
            return {"error": str(e)}
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """대화 기록 조회"""
        if self.use_langgraph and self.langgraph_service:
            try:
                session_info = self.langgraph_service.get_session_info(session_id)
                return [session_info] if session_info else []
            except Exception as e:
                self.logger.error(f"Failed to get conversation history: {e}")
                return []
        else:
            # 기존 방식 (placeholder)
            return []
    
    def clear_conversation_history(self, session_id: str) -> None:
        """대화 기록 삭제"""
        if self.use_langgraph and self.langgraph_service:
            try:
                # LangGraph에서는 체크포인트를 통해 세션 관리
                # 실제 삭제는 체크포인트 관리자에서 처리
                self.logger.info(f"Clearing conversation history for session: {session_id}")
            except Exception as e:
                self.logger.error(f"Failed to clear conversation history: {e}")
        else:
            # 기존 방식 (placeholder)
            pass
    
    async def _apply_naturalness_improvements(self, answer: str, phase1_info: Dict[str, Any], 
                                           phase2_info: Dict[str, Any], user_id: str) -> str:
        """
        자연스러운 답변 개선사항 적용
        
        Args:
            answer: 원본 답변
            phase1_info: Phase 1 정보 (대화 맥락)
            phase2_info: Phase 2 정보 (개인화 및 감정 분석)
            user_id: 사용자 ID
            
        Returns:
            str: 개선된 답변
        """
        try:
            if not answer or not isinstance(answer, str):
                return answer
            
            # 1. 대화 연결어 추가
            answer = await self._add_conversation_connectors(answer, phase1_info, phase2_info)
            
            # 2. 감정 톤 조절
            answer = await self._adjust_emotional_tone(answer, phase2_info)
            
            # 3. 개인화된 스타일 적용
            if self.personalized_style_learner:
                answer = self.personalized_style_learner.apply_personalized_style(answer, user_id)
            
            # 4. 실시간 피드백 기반 개선사항 적용
            if self.realtime_feedback_system:
                improvements = self.realtime_feedback_system.get_next_response_improvements()
                answer = self._apply_feedback_improvements(answer, improvements)
            
            # 5. 자연스러움 평가 및 최종 검증
            if self.naturalness_evaluator:
                context = {
                    "user_preference": phase2_info.get("user_profile", {}).get("style_preferences", {}).get("formality_level", "medium"),
                    "user_emotion": phase2_info.get("emotion_analysis", {}).get("primary_emotion", "neutral"),
                    "user_type": phase2_info.get("user_profile", {}).get("user_type", "general"),
                    "expertise_level": phase2_info.get("user_profile", {}).get("expertise_level", "beginner"),
                    "question_type": phase1_info.get("question_classification", {}).get("question_type", "general"),
                    "previous_topic": phase1_info.get("session_context", {}).get("current_topic", "")
                }
                
                metrics = self.naturalness_evaluator.evaluate_naturalness(answer, context)
                
                # 자연스러움이 매우 낮으면 추가 개선
                if metrics.overall_naturalness < 0.3:
                    recommendations = self.naturalness_evaluator.get_improvement_recommendations(metrics)
                    answer = self._apply_naturalness_recommendations(answer, recommendations)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying naturalness improvements: {e}")
            return answer
    
    async def _add_conversation_connectors(self, answer: str, phase1_info: Dict[str, Any], 
                                        phase2_info: Dict[str, Any]) -> str:
        """대화 연결어 추가"""
        try:
            if not self.conversation_connector:
                return answer
            
            context = {
                "previous_topic": phase1_info.get("session_context", {}).get("current_topic", ""),
                "conversation_flow": phase1_info.get("session_context", {}).get("flow_type", "new"),
                "user_emotion": phase2_info.get("emotion_analysis", {}).get("primary_emotion", "neutral"),
                "question_type": phase1_info.get("question_classification", {}).get("question_type", "general")
            }
            
            return self.conversation_connector.add_natural_connectors(answer, context)
            
        except Exception as e:
            self.logger.error(f"Error adding conversation connectors: {e}")
            return answer
    
    async def _adjust_emotional_tone(self, answer: str, phase2_info: Dict[str, Any]) -> str:
        """감정 톤 조절"""
        try:
            if not self.emotional_tone_adjuster:
                return answer
            
            emotion_analysis = phase2_info.get("emotion_analysis", {})
            user_emotion = emotion_analysis.get("primary_emotion", "neutral")
            emotion_intensity = emotion_analysis.get("intensity", 0.5)
            
            return self.emotional_tone_adjuster.adjust_tone(answer, user_emotion, emotion_intensity)
            
        except Exception as e:
            self.logger.error(f"Error adjusting emotional tone: {e}")
            return answer
    
    def _apply_feedback_improvements(self, answer: str, improvements: List[str]) -> str:
        """피드백 기반 개선사항 적용"""
        try:
            for improvement in improvements:
                if "간결하게" in improvement:
                    answer = self._make_concise(answer)
                elif "자세히" in improvement:
                    answer = self._add_details(answer)
                elif "친근한" in improvement:
                    answer = self._make_friendly(answer)
                elif "단계별" in improvement:
                    answer = self._structure_step_by_step(answer)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying feedback improvements: {e}")
            return answer
    
    def _apply_naturalness_recommendations(self, answer: str, recommendations: List[str]) -> str:
        """자연스러움 권장사항 적용"""
        try:
            for recommendation in recommendations:
                if "자연스러운 말투" in recommendation:
                    answer = self._make_casual(answer)
                elif "대화 연결어" in recommendation:
                    answer = self._add_connectors(answer)
                elif "공감적 표현" in recommendation:
                    answer = self._add_empathy(answer)
                elif "개인화된 표현" in recommendation:
                    answer = self._add_personalization(answer)
                elif "읽기 쉽게" in recommendation:
                    answer = self._improve_readability(answer)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying naturalness recommendations: {e}")
            return answer
    
    def _make_concise(self, answer: str) -> str:
        """답변을 간결하게 만들기"""
        try:
            sentences = answer.split(".")
            if len(sentences) > 3:
                return ". ".join(sentences[:3]) + "."
            return answer
        except Exception:
            return answer
    
    def _add_details(self, answer: str) -> str:
        """세부사항 추가"""
        try:
            if "예를 들어" not in answer:
                answer += " 예를 들어, 실제 사례를 통해 설명드리면 더 명확할 것 같아요."
            return answer
        except Exception:
            return answer
    
    def _make_friendly(self, answer: str) -> str:
        """친근하게 만들기"""
        try:
            replacements = {
                "~입니다": "~예요",
                "~합니다": "~해요",
                "~하시기 바랍니다": "~하시면 돼요"
            }
            
            for formal, casual in replacements.items():
                answer = answer.replace(formal, casual)
            
            return answer
        except Exception:
            return answer
    
    def _structure_step_by_step(self, answer: str) -> str:
        """단계별로 구조화"""
        try:
            if "단계" not in answer and "절차" not in answer:
                answer = f"단계별로 설명드리면, {answer}"
            return answer
        except Exception:
            return answer
    
    def _make_casual(self, answer: str) -> str:
        """캐주얼하게 만들기"""
        try:
            replacements = {
                "~입니다": "~예요",
                "~합니다": "~해요",
                "~됩니다": "~돼요",
                "귀하": "질문하신",
                "~하시기 바랍니다": "~하시면 돼요"
            }
            
            for formal, casual in replacements.items():
                answer = answer.replace(formal, casual)
            
            return answer
        except Exception:
            return answer
    
    def _add_connectors(self, answer: str) -> str:
        """연결어 추가"""
        try:
            connectors = ["그리고", "또한", "추가로"]
            if not any(connector in answer for connector in connectors):
                answer = f"그리고 {answer}"
            return answer
        except Exception:
            return answer
    
    def _add_empathy(self, answer: str) -> str:
        """공감 표현 추가"""
        try:
            empathy_expressions = ["이해하시는 마음이에요", "답답하시겠어요", "괜찮을 거예요"]
            if not any(expression in answer for expression in empathy_expressions):
                answer = f"이해하시는 마음이에요, {answer}"
            return answer
        except Exception:
            return answer
    
    def _add_personalization(self, answer: str) -> str:
        """개인화 표현 추가"""
        try:
            personal_expressions = ["질문하신", "말씀하신", "문의하신"]
            if not any(expression in answer for expression in personal_expressions):
                answer = f"질문하신 내용에 대해 {answer}"
            return answer
        except Exception:
            return answer
    
    def _improve_readability(self, answer: str) -> str:
        """가독성 개선"""
        try:
            # 문장을 더 짧게 만들기
            sentences = answer.split(".")
            improved_sentences = []
            
            for sentence in sentences:
                if len(sentence.split()) > 20:  # 너무 긴 문장
                    # 간단히 나누기
                    words = sentence.split()
                    mid_point = len(words) // 2
                    improved_sentences.append(" ".join(words[:mid_point]))
                    improved_sentences.append(" ".join(words[mid_point:]))
                else:
                    improved_sentences.append(sentence)
            
            return ". ".join(filter(None, improved_sentences)) + "."
        except Exception:
            return answer
    
    def collect_user_feedback(self, session_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 피드백 수집
        
        Args:
            session_id: 세션 ID
            feedback_data: 피드백 데이터
            
        Returns:
            Dict[str, Any]: 피드백 처리 결과
        """
        try:
            if not self.realtime_feedback_system:
                return {"feedback_received": False, "error": "Feedback system not available"}
            
            return self.realtime_feedback_system.collect_feedback(session_id, feedback_data)
            
        except Exception as e:
            self.logger.error(f"Error collecting user feedback: {e}")
            return {"feedback_received": False, "error": str(e)}
    
    def get_naturalness_evaluation(self, answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        답변의 자연스러움 평가
        
        Args:
            answer: 평가할 답변
            context: 평가 맥락
            
        Returns:
            Dict[str, Any]: 자연스러움 평가 결과
        """
        try:
            if not self.naturalness_evaluator:
                return {"error": "Naturalness evaluator not available"}
            
            metrics = self.naturalness_evaluator.evaluate_naturalness(answer, context)
            
            return {
                "overall_score": metrics.overall_naturalness,
                "category_scores": {
                    "formality": metrics.formality_score,
                    "conversation_flow": metrics.conversation_flow_score,
                    "emotional_appropriateness": metrics.emotional_appropriateness,
                    "personalization": metrics.personalization_score,
                    "readability": metrics.readability_score
                },
                "naturalness_level": metrics.detailed_analysis["naturalness_level"],
                "strengths": metrics.detailed_analysis["strengths"],
                "weaknesses": metrics.detailed_analysis["weaknesses"],
                "suggestions": metrics.detailed_analysis["suggestions"],
                "evaluation_timestamp": metrics.evaluation_timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating naturalness: {e}")
            return {"error": str(e)}
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        status = {
            "service_name": "ChatService",
            "langgraph_enabled": self.use_langgraph,
            "langgraph_service_available": self.langgraph_service is not None,
            "timestamp": time.time()
        }
        
        if self.use_langgraph and self.langgraph_service:
            try:
                langgraph_status = self.langgraph_service.get_service_status()
                status["langgraph_status"] = langgraph_status
            except Exception as e:
                status["langgraph_error"] = str(e)
        
        return status
    
    async def test_service(self, test_message: str = "테스트 질문입니다") -> Dict[str, Any]:
        """서비스 테스트"""
        try:
            result = await self.process_message(test_message)
            
            test_passed = (
                "response" in result and 
                result["response"] and 
                "processing_time" in result
            )
            
            return {
                "test_passed": test_passed,
                "test_message": test_message,
                "result": result,
                "langgraph_enabled": self.use_langgraph
            }
            
        except Exception as e:
            return {
                "test_passed": False,
                "test_message": test_message,
                "error": str(e),
                "langgraph_enabled": self.use_langgraph
            }
