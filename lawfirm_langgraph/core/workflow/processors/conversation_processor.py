# -*- coding: utf-8 -*-
"""
Conversation Processor
대화 처리 관련 로직 (긴급도 평가, 멀티턴 처리)
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import time
from typing import Any, Dict, Optional, Tuple

from core.workflow.state.state_definitions import LegalWorkflowState

logger = get_logger(__name__)


class ConversationProcessor:
    """대화 처리 프로세서 (긴급도 평가, 멀티턴 처리)"""

    def __init__(
        self,
        logger,
        emotion_analyzer=None,
        multi_turn_handler=None,
        conversation_manager=None,
        llm=None,
        get_state_value_func=None,
        set_state_value_func=None,
        update_processing_time_func=None,
        handle_error_func=None
    ):
        self.logger = logger
        self.emotion_analyzer = emotion_analyzer
        self.multi_turn_handler = multi_turn_handler
        self.conversation_manager = conversation_manager
        self.llm = llm
        self._get_state_value = get_state_value_func
        self._set_state_value = set_state_value_func
        self._update_processing_time = update_processing_time_func
        self._handle_error = handle_error_func

    def assess_urgency(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """긴급도 평가 노드"""
        try:
            start_time = time.time()
            query = self._get_state_value(state, "query", "")

            if not query:
                self.logger.warning("Empty query in assess_urgency")
                self._set_state_value(state, "urgency_level", "normal")
                self._set_state_value(state, "urgency_reasoning", "쿼리가 비어있습니다.")
                return state

            urgency_level, urgency_reasoning = self.assess_urgency_internal(query)

            self._set_state_value(state, "urgency_level", urgency_level)
            self._set_state_value(state, "urgency_reasoning", urgency_reasoning)

            self._update_processing_time(state, start_time)

        except Exception as e:
            self.logger.error(f"Error in assess_urgency: {e}")
            self._set_state_value(state, "urgency_level", "normal")
            self._set_state_value(state, "urgency_reasoning", "긴급도 평가 중 오류 발생")
            self._handle_error(state, str(e), "긴급도 평가 중 오류 발생")

        return state

    def assess_urgency_internal(self, query: str) -> Tuple[str, str]:
        """긴급도 평가 (내부 로직)"""
        try:
            if self.emotion_analyzer:
                try:
                    intent_result = self.emotion_analyzer.analyze_intent(query, None)
                    
                    if intent_result and hasattr(intent_result, 'emergency_level'):
                        if hasattr(intent_result.emergency_level, 'value'):
                            urgency_level = intent_result.emergency_level.value
                        elif hasattr(intent_result.emergency_level, 'lower'):
                            urgency_level = intent_result.emergency_level.lower()
                        else:
                            urgency_level = str(intent_result.emergency_level).lower()
                        
                        urgency_reasoning = getattr(intent_result, 'reasoning', None) or "긴급도 분석 완료"
                        return urgency_level, urgency_reasoning
                except Exception as e:
                    self.logger.warning(f"Emotion analyzer failed: {e}. Using fallback.")
            
            return self.assess_urgency_fallback(query)
        except Exception as e:
            self.logger.error(f"Error in assess_urgency_internal: {e}")
            return "medium", "오류 발생, 기본값 사용"

    def assess_urgency_fallback(self, query: str) -> Tuple[str, str]:
        """폴백 긴급도 평가"""
        urgent_keywords = ["긴급", "급해", "빨리", "즉시", "당장"]
        high_keywords = ["오늘", "내일", "이번주", "마감"]
        
        query_lower = query.lower()
        if any(k in query_lower for k in urgent_keywords):
            return "critical", "긴급 키워드 감지"
        elif any(k in query_lower for k in high_keywords):
            return "high", "높은 긴급도 키워드 감지"
        else:
            return "medium", "일반 긴급도"

    def resolve_multi_turn(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """멀티턴 질문 해결 노드"""
        try:
            start_time = time.time()

            if not self.multi_turn_handler or not self.conversation_manager:
                self._set_state_value(state, "is_multi_turn", False)
                query = self._get_state_value(state, "query", "")
                self._set_state_value(state, "search_query", query)
                self.logger.debug("Multi-turn handler not available, skipping multi-turn resolution")
                return state

            query = self._get_state_value(state, "query", "")
            session_id = self._get_state_value(state, "session_id", "")

            conversation_context = self._get_or_create_conversation_context(session_id)

            if conversation_context and conversation_context.turns:
                is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, conversation_context)
                self._set_state_value(state, "is_multi_turn", is_multi_turn)

                if is_multi_turn:
                    multi_turn_result = self.multi_turn_handler.build_complete_query(query, conversation_context)

                    resolved_query = multi_turn_result.get("resolved_query", query) if isinstance(multi_turn_result, dict) else (
                        multi_turn_result.complete_query if hasattr(multi_turn_result, 'complete_query') else query
                    )
                    self._set_state_value(state, "multi_turn_confidence", 
                        multi_turn_result.get("confidence", 1.0) if isinstance(multi_turn_result, dict) else 
                        (getattr(multi_turn_result, 'confidence', 1.0) if hasattr(multi_turn_result, 'confidence') else 1.0))

                    self._set_state_value(state, "conversation_context", self._build_conversation_context_dict(conversation_context))
                    self._set_state_value(state, "search_query", resolved_query)

                    self.logger.info(f"Multi-turn question resolved: '{query}' -> '{resolved_query}'")
                else:
                    self._set_state_value(state, "multi_turn_confidence", 1.0)
                    self._set_state_value(state, "search_query", query)
            else:
                self._set_state_value(state, "is_multi_turn", False)
                self._set_state_value(state, "multi_turn_confidence", 1.0)
                self._set_state_value(state, "search_query", query)

            self._update_processing_time(state, start_time)

        except Exception as e:
            self.logger.error(f"Error in resolve_multi_turn: {e}")
            self._set_state_value(state, "is_multi_turn", False)
            search_query = self._get_state_value(state, "search_query")
            if not search_query:
                search_query = self._get_state_value(state, "query", "")
            self._set_state_value(state, "search_query", search_query)
            self._handle_error(state, str(e), "멀티턴 처리 중 오류 발생")

        return state

    def resolve_multi_turn_internal(self, query: str, session_id: str) -> Tuple[bool, str]:
        """멀티턴 처리 (내부 로직)"""
        try:
            if not self.multi_turn_handler or not self.conversation_manager:
                return False, query

            conversation_context = self._get_or_create_conversation_context(session_id)

            if conversation_context and conversation_context.turns:
                is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, conversation_context)

                if is_multi_turn:
                    multi_turn_result = self.multi_turn_handler.build_complete_query(query, conversation_context)
                    search_query = (
                        multi_turn_result.complete_query if hasattr(multi_turn_result, 'complete_query') else
                        (multi_turn_result.get("resolved_query", query) if isinstance(multi_turn_result, dict) else query)
                    )
                    return True, search_query

            return False, query
        except Exception as e:
            self.logger.error(f"멀티턴 처리 내부 로직 실패: {e}")
            return False, query

    def _get_or_create_conversation_context(self, session_id: str):
        """대화 맥락 가져오기 또는 생성"""
        try:
            if not self.conversation_manager:
                return None

            sessions = getattr(self.conversation_manager, 'sessions', {})
            context = sessions.get(session_id)

            return context
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return None

    def _build_conversation_context_dict(self, context):
        """ConversationContext를 딕셔너리로 변환"""
        try:
            from core.shared.utils.query_builder import QueryBuilder
            result = QueryBuilder.build_conversation_context_dict(context)
            if result is None and context is not None:
                self.logger.error(f"Error building conversation context dict")
            return result
        except Exception as e:
            self.logger.error(f"Error building conversation context dict: {e}")
            return None

