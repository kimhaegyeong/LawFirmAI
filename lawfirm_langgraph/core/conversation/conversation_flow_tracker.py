# -*- coding: utf-8 -*-
"""
대화 흐름 추적기
대화 패턴을 학습하고 다음 질문을 예측하여 후속 질문을 제안합니다.
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import re

from .conversation_manager import ConversationContext, ConversationTurn
try:
    from core.classification.analyzers.emotion_intent_analyzer import EmotionIntentAnalyzer
except ImportError:
    # 호환성을 위한 fallback (더 이상 services에 없음)
    EmotionIntentAnalyzer = None

logger = get_logger(__name__)


@dataclass
class FlowPattern:
    """대화 흐름 패턴"""
    pattern_id: str
    pattern_type: str
    sequence: List[str]
    frequency: int
    success_rate: float
    last_seen: datetime


@dataclass
class ConversationBranch:
    """대화 분기점"""
    branch_id: str
    branch_type: str
    trigger_keywords: List[str]
    follow_up_suggestions: List[str]
    probability: float


class ConversationFlowTracker:
    """대화 흐름 추적기"""
    
    def __init__(self, max_pattern_history: int = 1000):
        """
        대화 흐름 추적기 초기화
        
        Args:
            max_pattern_history: 최대 패턴 이력 수
        """
        self.logger = get_logger(__name__)
        self.max_pattern_history = max_pattern_history
        
        # 감정/의도 분석기
        self.emotion_intent_analyzer = EmotionIntentAnalyzer()
        
        # 대화 패턴 저장소
        self.flow_patterns: Dict[str, FlowPattern] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        
        # 대화 분기점 저장소
        self.conversation_branches: Dict[str, ConversationBranch] = {}
        
        # 질문 유형 분류 캐시 추가
        self._question_type_cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(hours=24)  # 24시간 캐시 유지
        
        # LLM 기반 질문 생성 캐시
        self._llm_suggestions_cache: Dict[str, Tuple[List[str], datetime]] = {}
        
        # 통합 LLM 분석 캐시
        self._comprehensive_analysis_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        
        # ClassificationHandler 지연 로딩 (필요 시 초기화)
        self._classification_handler = None
        
        # 질문 유형별 후속 질문 템플릿
        self.follow_up_templates = {
            "law_inquiry": [
                "관련 판례는 어떤 것이 있나요?",
                "실제 적용 사례를 알려주세요",
                "다른 법령과의 관계는 어떻게 되나요?",
                "변경된 법령이 있다면 무엇인가요?"
            ],
            "contract_review": [
                "계약서에 추가해야 할 조항이 있나요?",
                "위험 요소는 무엇인가요?",
                "계약 해지 조건은 어떻게 되나요?",
                "분쟁 해결 절차는 어떻게 되나요?"
            ],
            "legal_procedure": [
                "필요한 서류는 무엇인가요?",
                "소요 기간은 얼마나 걸리나요?",
                "비용은 얼마나 드나요?",
                "주의사항이 있다면 무엇인가요?"
            ],
            "procedure_guide": [
                "필요한 서류는 무엇인가요?",
                "소요 기간은 얼마나 걸리나요?",
                "비용은 얼마나 드나요?",
                "주의사항이 있다면 무엇인가요?",
                "관련 법령을 알려주세요",
                "실제 사례를 알려주세요"
            ],
            "damage_claim": [
                "손해액 산정 방법은 어떻게 되나요?",
                "증거 수집 방법을 알려주세요",
                "시효는 언제까지인가요?",
                "화해나 조정은 가능한가요?"
            ],
            "employment_law": [
                "근로자의 권리는 무엇인가요?",
                "사용자의 의무는 무엇인가요?",
                "분쟁 발생 시 어떻게 해야 하나요?",
                "관련 기관은 어디인가요?"
            ],
            "family_law": [
                "재산 분할은 어떻게 되나요?",
                "자녀 양육권은 어떻게 결정되나요?",
                "위자료는 얼마나 받을 수 있나요?",
                "이혼 절차는 어떻게 되나요?"
            ],
            "general_question": [
                "관련 판례는 어떤 것이 있나요?",
                "실제 적용 사례를 알려주세요",
                "더 자세한 설명이 필요합니다",
                "관련 법령을 알려주세요"
            ]
        }
        
        # 대화 분기점 패턴
        self.branch_patterns = {
            "detailed_explanation": {
                "trigger_keywords": ["자세히", "구체적으로", "더", "추가로"],
                "follow_up_suggestions": [
                    "예시를 들어 설명해드릴까요?",
                    "관련 판례도 함께 알려드릴까요?",
                    "실무에서 주의할 점도 설명해드릴까요?"
                ]
            },
            "related_topics": {
                "trigger_keywords": ["관련", "연관", "비슷한", "다른"],
                "follow_up_suggestions": [
                    "관련된 다른 법령도 궁금하신가요?",
                    "비슷한 사안의 판례도 찾아드릴까요?",
                    "연관된 절차도 설명해드릴까요?"
                ]
            },
            "practical_application": {
                "trigger_keywords": ["실제", "실무", "현실", "구체적"],
                "follow_up_suggestions": [
                    "실제 사례를 들어 설명해드릴까요?",
                    "실무에서 자주 발생하는 문제점도 알려드릴까요?",
                    "주의사항도 함께 안내해드릴까요?"
                ]
            },
            "comparison": {
                "trigger_keywords": ["비교", "차이", "다른", "반대"],
                "follow_up_suggestions": [
                    "다른 법령과 비교해드릴까요?",
                    "유사한 사안과의 차이점도 설명해드릴까요?",
                    "반대되는 경우도 알려드릴까요?"
                ]
            }
        }
        
        # 대화 흐름 상태
        self.conversation_states = {
            "initial": "초기 질문",
            "clarification": "명확화 요청",
            "follow_up": "후속 질문",
            "deep_dive": "심화 탐구",
            "practical": "실무 적용",
            "comparison": "비교 분석",
            "conclusion": "결론 도출"
        }
        
        self.logger.info("ConversationFlowTracker initialized")
    
    def track_conversation_flow(self, session_id: str, turn: ConversationTurn) -> None:
        """
        대화 흐름 추적
        
        Args:
            session_id: 세션 ID
            turn: 대화 턴
        """
        try:
            # 의도 분석
            intent_result = self.emotion_intent_analyzer.analyze_intent(turn.user_query)
            intent_type = intent_result.primary_intent.value
            
            # 질문 유형 추출
            question_type = turn.question_type or "general_question"
            
            # 패턴 업데이트
            self._update_flow_pattern(session_id, intent_type, question_type)
            
            # 분기점 감지
            branch = self.detect_conversation_branch(turn.user_query)
            if branch:
                self._update_conversation_branch(branch)
            
            self.logger.debug(f"Flow tracked for session {session_id}: {intent_type} -> {question_type}")
            
        except Exception as e:
            self.logger.error(f"Error tracking conversation flow: {e}")
    
    def predict_next_intent(self, context: ConversationContext) -> List[str]:
        """
        다음 의도 예측
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 예측된 의도 목록
        """
        try:
            if not context.turns:
                return ["question"]
            
            # 최근 턴들의 의도 분석
            recent_intents = []
            for turn in context.turns[-3:]:  # 최근 3턴
                intent_result = self.emotion_intent_analyzer.analyze_intent(turn.user_query)
                recent_intents.append(intent_result.primary_intent.value)
            
            # 패턴 기반 예측
            predicted_intents = self._predict_intent_from_patterns(recent_intents)
            
            # 맥락 기반 예측
            context_intents = self._predict_intent_from_context(context)
            
            # 결과 병합 및 정렬
            all_intents = predicted_intents + context_intents
            intent_counts = defaultdict(int)
            for intent in all_intents:
                intent_counts[intent] += 1
            
            # 빈도순으로 정렬
            sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
            return [intent for intent, count in sorted_intents[:5]]  # 상위 5개
            
        except Exception as e:
            self.logger.error(f"Error predicting next intent: {e}")
            return ["question"]
    
    def suggest_follow_up_questions(self, context: ConversationContext) -> List[str]:
        """
        후속 질문 제안 (LLM 통합 분석 포함)
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 제안된 후속 질문 목록
        """
        try:
            if not context.turns:
                self.logger.debug("No turns in context, returning empty suggestions")
                return []
            
            suggestions = []
            
            # 통합 LLM 분석 (한 번에 모든 정보 추출)
            comprehensive_result = self._analyze_conversation_comprehensive_with_llm(context)
            
            if comprehensive_result:
                # LLM 분석 결과 사용
                question_type = comprehensive_result.get("question_type")
                branch_type = comprehensive_result.get("branch_type")
                entities = comprehensive_result.get("entities", {})
                llm_suggestions = comprehensive_result.get("suggestions", [])
                confidence = comprehensive_result.get("confidence", 0.8)
                
                self.logger.debug(
                    f"Comprehensive analysis result: "
                    f"question_type={question_type}, branch_type={branch_type}, "
                    f"confidence={confidence:.2f}, suggestions_count={len(llm_suggestions)}"
                )
                
                # 1. LLM이 생성한 후속 질문 추가
                if llm_suggestions:
                    suggestions.extend(llm_suggestions)
                    self.logger.debug(f"Added {len(llm_suggestions)} LLM-generated suggestions")
                
                # 2. 템플릿 기반 제안 (LLM 질문이 부족하면)
                if len(suggestions) < 3 and question_type:
                    if question_type in self.follow_up_templates:
                        template_questions = self.follow_up_templates[question_type]
                        suggestions.extend(template_questions)
                        self.logger.debug(f"Added {len(template_questions)} template suggestions")
                    elif "general_question" in self.follow_up_templates:
                        fallback_questions = self.follow_up_templates["general_question"]
                        suggestions.extend(fallback_questions)
                        self.logger.debug(f"Added {len(fallback_questions)} fallback template suggestions")
                
                # 3. 분기점 기반 제안 (LLM이 감지한 분기점 사용)
                if branch_type and branch_type in self.branch_patterns:
                    branch_suggestions = self.branch_patterns[branch_type]["follow_up_suggestions"]
                    suggestions.extend(branch_suggestions)
                    self.logger.debug(f"Added {len(branch_suggestions)} branch suggestions")
                
                # 4. 엔티티 기반 제안 (LLM이 추출한 엔티티 사용)
                entity_suggestions = self._generate_entity_suggestions_from_entities(entities)
                if entity_suggestions:
                    suggestions.extend(entity_suggestions)
                    self.logger.debug(f"Added {len(entity_suggestions)} entity-based suggestions")
            else:
                # Fallback: 기존 단계별 처리
                self.logger.debug("Comprehensive analysis failed, using fallback method")
                return self._suggest_follow_up_questions_fallback(context)
            
            # 중복 제거 및 정렬
            unique_suggestions = list(dict.fromkeys(suggestions))
            final_suggestions = unique_suggestions[:5]  # 최대 5개
            
            self.logger.info(
                f"Generated {len(final_suggestions)} follow-up questions "
                f"(comprehensive analysis, total before dedup: {len(suggestions)})"
            )
            
            return final_suggestions
            
        except Exception as e:
            self.logger.error(f"Error suggesting follow-up questions: {e}", exc_info=True)
            # Fallback: 기존 방식
            return self._suggest_follow_up_questions_fallback(context)
    
    def _suggest_follow_up_questions_fallback(self, context: ConversationContext) -> List[str]:
        """
        후속 질문 제안 (Fallback - 기존 단계별 처리)
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 제안된 후속 질문 목록
        """
        try:
            if not context.turns:
                self.logger.debug("No turns in context, returning empty suggestions")
                return []
            
            suggestions = []
            last_turn = context.turns[-1]
            
            # 1단계: LLM으로 질문 유형 분류 (캐시 우선)
            question_type = None
            recent_question_types = []
            
            for turn in context.turns[-2:]:  # 최근 2턴
                if turn.question_type:
                    recent_question_types.append(turn.question_type)
                    question_type = turn.question_type
            
            # 질문 유형이 없거나 신뢰도가 낮으면 LLM으로 재분류
            if not question_type or question_type == "general_question":
                # 캐시 확인
                cached_type = self._get_cached_question_type(last_turn.user_query)
                if cached_type:
                    question_type = cached_type
                    self.logger.debug(f"Using cached question type: {question_type}")
                else:
                    # LLM으로 분류
                    question_type = self._classify_question_type_with_llm(
                        last_turn.user_query,
                        last_turn.bot_response
                    )
                    # 캐시 저장
                    self._cache_question_type(last_turn.user_query, question_type)
                    self.logger.info(f"Classified question type with LLM: {question_type} for query: {last_turn.user_query[:50]}...")
                
                # recent_question_types 업데이트
                if question_type:
                    recent_question_types = [question_type]
            
            self.logger.debug(f"Recent question types: {recent_question_types}")
            
            # 2단계: 질문 유형별 템플릿 제안
            template_suggestions_count = 0
            for q_type in recent_question_types:
                if q_type in self.follow_up_templates:
                    template_questions = self.follow_up_templates[q_type]
                    suggestions.extend(template_questions)
                    template_suggestions_count += len(template_questions)
                    self.logger.debug(f"Added {len(template_questions)} suggestions from template '{q_type}'")
                else:
                    # 질문 유형이 템플릿에 없으면 일반 질문 템플릿 사용
                    if "general_question" in self.follow_up_templates:
                        fallback_questions = self.follow_up_templates["general_question"]
                        suggestions.extend(fallback_questions)
                        template_suggestions_count += len(fallback_questions)
                        self.logger.debug(f"Using fallback template 'general_question' for unknown type '{q_type}'")
            
            # 3단계: LLM 기반 제안 (템플릿 제안이 부족하면)
            llm_suggestions_count = 0
            if len(suggestions) < 3:
                try:
                    llm_suggestions = self._generate_llm_suggestions(context)
                    if llm_suggestions:
                        suggestions.extend(llm_suggestions)
                        llm_suggestions_count = len(llm_suggestions)
                        self.logger.debug(f"Added {llm_suggestions_count} LLM-based suggestions")
                except Exception as e:
                    self.logger.warning(f"Error generating LLM suggestions: {e}")
            
            # 4단계: 분기점 기반 제안
            branch_suggestions_count = 0
            try:
                branch = self.detect_conversation_branch(last_turn.user_query)
                if branch:
                    branch_questions = branch.follow_up_suggestions
                    suggestions.extend(branch_questions)
                    branch_suggestions_count = len(branch_questions)
                    self.logger.debug(f"Added {branch_suggestions_count} suggestions from branch '{branch.branch_type}'")
            except Exception as e:
                self.logger.warning(f"Error detecting conversation branch: {e}")
            
            # 5단계: 엔티티 기반 제안
            entity_suggestions_count = 0
            try:
                entity_suggestions = self._generate_entity_based_suggestions(context)
                if entity_suggestions:
                    suggestions.extend(entity_suggestions)
                    entity_suggestions_count = len(entity_suggestions)
                    self.logger.debug(f"Added {entity_suggestions_count} entity-based suggestions")
            except Exception as e:
                self.logger.warning(f"Error generating entity-based suggestions: {e}")
            
            # 중복 제거 및 정렬
            unique_suggestions = list(dict.fromkeys(suggestions))
            final_suggestions = unique_suggestions[:5]  # 최대 5개
            
            self.logger.info(
                f"Generated {len(final_suggestions)} follow-up questions (fallback) "
                f"(template: {template_suggestions_count}, llm: {llm_suggestions_count}, "
                f"branch: {branch_suggestions_count}, entity: {entity_suggestions_count}, "
                f"total before dedup: {len(suggestions)})"
            )
            
            return final_suggestions
            
        except Exception as e:
            self.logger.error(f"Error in fallback suggest follow-up questions: {e}", exc_info=True)
            return []
    
    def detect_conversation_branch(self, query: str) -> Optional[ConversationBranch]:
        """
        대화 분기점 감지
        
        Args:
            query: 사용자 질문
            
        Returns:
            Optional[ConversationBranch]: 감지된 분기점 (없으면 None)
        """
        try:
            query_lower = query.lower()
            
            for branch_type, pattern_info in self.branch_patterns.items():
                trigger_keywords = pattern_info["trigger_keywords"]
                follow_up_suggestions = pattern_info["follow_up_suggestions"]
                
                # 트리거 키워드 매칭
                matched_keywords = [keyword for keyword in trigger_keywords 
                                  if keyword in query_lower]
                
                if matched_keywords:
                    # 분기점 생성
                    branch = ConversationBranch(
                        branch_id=f"{branch_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        branch_type=branch_type,
                        trigger_keywords=matched_keywords,
                        follow_up_suggestions=follow_up_suggestions,
                        probability=len(matched_keywords) / len(trigger_keywords)
                    )
                    
                    return branch
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting conversation branch: {e}")
            return None
    
    def analyze_flow_patterns(self, sessions: List[ConversationContext]) -> Dict[str, Any]:
        """
        대화 흐름 패턴 분석
        
        Args:
            sessions: 대화 세션 목록
            
        Returns:
            Dict[str, Any]: 패턴 분석 결과
        """
        try:
            pattern_analysis = {
                "total_sessions": len(sessions),
                "common_patterns": [],
                "flow_transitions": {},
                "branch_frequency": {},
                "successful_patterns": [],
                "failed_patterns": []
            }
            
            # 패턴 빈도 분석
            pattern_counts = defaultdict(int)
            transition_counts = defaultdict(int)
            
            for session in sessions:
                if len(session.turns) < 2:
                    continue
                
                # 의도 시퀀스 추출
                intent_sequence = []
                for turn in session.turns:
                    intent_result = self.emotion_intent_analyzer.analyze_intent(turn.user_query)
                    intent_sequence.append(intent_result.primary_intent.value)
                
                # 패턴 카운트
                for i in range(len(intent_sequence) - 1):
                    pattern = f"{intent_sequence[i]} -> {intent_sequence[i+1]}"
                    pattern_counts[pattern] += 1
                
                # 전환 카운트
                for i in range(len(intent_sequence) - 1):
                    transition = (intent_sequence[i], intent_sequence[i+1])
                    transition_counts[transition] += 1
            
            # 상위 패턴 추출
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            pattern_analysis["common_patterns"] = sorted_patterns[:10]
            
            # 전환 분석
            pattern_analysis["flow_transitions"] = dict(sorted_patterns[:10])
            
            # 분기점 빈도 분석
            branch_counts = defaultdict(int)
            for session in sessions:
                for turn in session.turns:
                    branch = self.detect_conversation_branch(turn.user_query)
                    if branch:
                        branch_counts[branch.branch_type] += 1
            
            pattern_analysis["branch_frequency"] = dict(branch_counts)
            
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing flow patterns: {e}")
            return {}
    
    def get_conversation_state(self, context: ConversationContext) -> str:
        """
        현재 대화 상태 파악
        
        Args:
            context: 대화 맥락
            
        Returns:
            str: 대화 상태
        """
        try:
            if not context.turns:
                return "initial"
            
            # 최근 턴들의 의도 분석
            recent_intents = []
            for turn in context.turns[-3:]:
                intent_result = self.emotion_intent_analyzer.analyze_intent(turn.user_query)
                recent_intents.append(intent_result.primary_intent.value)
            
            # 상태 결정 로직
            if len(recent_intents) == 1:
                return "initial"
            elif "clarification" in recent_intents:
                return "clarification"
            elif "follow_up" in recent_intents:
                return "follow_up"
            elif len(recent_intents) >= 3:
                return "deep_dive"
            else:
                return "general"
            
        except Exception as e:
            self.logger.error(f"Error getting conversation state: {e}")
            return "initial"
    
    def _update_flow_pattern(self, session_id: str, intent_type: str, question_type: str) -> None:
        """흐름 패턴 업데이트"""
        try:
            pattern_key = f"{intent_type}_{question_type}"
            self.pattern_frequency[pattern_key] += 1
            
            # 패턴 객체 생성 또는 업데이트
            if pattern_key not in self.flow_patterns:
                self.flow_patterns[pattern_key] = FlowPattern(
                    pattern_id=pattern_key,
                    pattern_type=question_type,
                    sequence=[intent_type],
                    frequency=1,
                    success_rate=0.0,
                    last_seen=datetime.now()
                )
            else:
                pattern = self.flow_patterns[pattern_key]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                
                # 성공률 업데이트 (간단한 휴리스틱)
                if pattern.frequency > 1:
                    pattern.success_rate = min(1.0, pattern.success_rate + 0.1)
            
        except Exception as e:
            self.logger.error(f"Error updating flow pattern: {e}")
    
    def _update_conversation_branch(self, branch: ConversationBranch) -> None:
        """대화 분기점 업데이트"""
        try:
            branch_key = branch.branch_type
            if branch_key in self.conversation_branches:
                existing_branch = self.conversation_branches[branch_key]
                existing_branch.probability = (existing_branch.probability + branch.probability) / 2
            else:
                self.conversation_branches[branch_key] = branch
            
        except Exception as e:
            self.logger.error(f"Error updating conversation branch: {e}")
    
    def _predict_intent_from_patterns(self, recent_intents: List[str]) -> List[str]:
        """패턴 기반 의도 예측"""
        try:
            predictions = []
            
            # 최근 의도 패턴 분석
            if len(recent_intents) >= 2:
                last_intent = recent_intents[-1]
                
                # 패턴 기반 예측
                for pattern_key, pattern in self.flow_patterns.items():
                    if last_intent in pattern.sequence:
                        # 다음 의도 예측
                        if pattern.pattern_type == "law_inquiry":
                            predictions.extend(["clarification", "follow_up"])
                        elif pattern.pattern_type == "contract_review":
                            predictions.extend(["request", "clarification"])
                        elif pattern.pattern_type == "legal_procedure":
                            predictions.extend(["question", "clarification"])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting intent from patterns: {e}")
            return []
    
    def _predict_intent_from_context(self, context: ConversationContext) -> List[str]:
        """맥락 기반 의도 예측"""
        try:
            predictions = []
            
            # 엔티티 기반 예측
            if context.entities:
                if context.entities.get("laws"):
                    predictions.append("clarification")
                if context.entities.get("precedents"):
                    predictions.append("follow_up")
                if context.entities.get("legal_terms"):
                    predictions.append("question")
            
            # 주제 스택 기반 예측
            if context.topic_stack:
                if "계약" in context.topic_stack:
                    predictions.extend(["request", "clarification"])
                elif "손해배상" in context.topic_stack:
                    predictions.extend(["question", "follow_up"])
                elif "절차" in context.topic_stack:
                    predictions.extend(["clarification", "request"])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting intent from context: {e}")
            return []
    
    def _generate_entity_based_suggestions(self, context: ConversationContext) -> List[str]:
        """엔티티 기반 제안 생성"""
        try:
            suggestions = []
            
            # 법률 기반 제안
            if context.entities.get("laws"):
                suggestions.append("관련 조문도 함께 설명해드릴까요?")
                suggestions.append("이 법률의 적용 범위는 어떻게 되나요?")
            
            # 판례 기반 제안
            if context.entities.get("precedents"):
                suggestions.append("유사한 판례도 찾아드릴까요?")
                suggestions.append("이 판례의 핵심 쟁점은 무엇인가요?")
            
            # 법률 용어 기반 제안
            if context.entities.get("legal_terms"):
                suggestions.append("관련 용어들도 함께 설명해드릴까요?")
                suggestions.append("이 용어의 정의를 더 자세히 알려드릴까요?")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating entity-based suggestions: {e}")
            return []
    
    def _get_classification_handler(self):
        """ClassificationHandler 지연 로딩"""
        if self._classification_handler is None:
            try:
                from core.agents.handlers.classification_handler import ClassificationHandler
                # ClassificationHandler는 llm과 llm_fast가 필요하므로 None 반환
                # 실제 사용 시에는 workflow에서 초기화된 handler를 사용해야 함
                self.logger.debug("ClassificationHandler requires llm and llm_fast parameters, skipping initialization")
                return None
            except ImportError:
                try:
                    from core.classification.handlers.classification_handler import ClassificationHandler
                    self.logger.debug("ClassificationHandler requires llm and llm_fast parameters, skipping initialization")
                    return None
                except ImportError as e:
                    self.logger.debug(f"Failed to import ClassificationHandler: {e}")
                    return None
            except Exception as e:
                self.logger.debug(f"ClassificationHandler not available: {e}")
                return None
        return self._classification_handler
    
    def _classify_question_type_with_llm(self, query: str, bot_response: str = "") -> str:
        """
        LLM을 사용하여 질문 유형 분류
        
        Args:
            query: 사용자 질문
            bot_response: 봇 응답 (선택사항, 맥락 제공용)
        
        Returns:
            str: 질문 유형 (예: "law_inquiry", "procedure_guide" 등)
        """
        try:
            handler = self._get_classification_handler()
            if not handler:
                # ClassificationHandler 초기화 실패 시 상세 로깅
                self.logger.warning(
                    f"ClassificationHandler not available, using fallback. "
                    f"This may affect classification accuracy. "
                    f"Please check LLM configuration and ensure llm/llm_fast are properly initialized."
                )
                return self._classify_question_type_fallback(query)
            
            question_type, confidence = handler.classify_with_llm(query)
            
            # QuestionType enum을 문자열로 변환
            if hasattr(question_type, 'value'):
                result = question_type.value
            elif hasattr(question_type, 'name'):
                result = question_type.name.lower()
            else:
                result = str(question_type).lower()
            
            self.logger.debug(f"LLM classified question type: {result} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to classify question type with LLM: {e}, using fallback")
            return self._classify_question_type_fallback(query)
    
    def _classify_question_type_fallback(self, query: str) -> str:
        """
        키워드 기반 Fallback 분류
        
        Args:
            query: 사용자 질문
        
        Returns:
            str: 질문 유형
        """
        query_lower = query.lower()
        
        # 키워드 기반 분류
        if any(kw in query_lower for kw in ["판례", "판결", "사건", "대법원", "법원"]):
            return "precedent_search"
        elif any(kw in query_lower for kw in ["조문", "법령", "법률", "규정"]):
            return "law_inquiry"
        elif any(kw in query_lower for kw in ["절차", "방법", "소송", "청구"]):
            return "procedure_guide"
        elif any(kw in query_lower for kw in ["의미", "정의", "뜻", "용어"]):
            return "term_explanation"
        elif any(kw in query_lower for kw in ["조언", "권리", "의무", "계약서"]):
            return "legal_advice"
        else:
            return "general_question"
    
    def _get_cached_question_type(self, query: str) -> Optional[str]:
        """캐시에서 질문 유형 조회"""
        if query in self._question_type_cache:
            question_type, cached_time = self._question_type_cache[query]
            if datetime.now() - cached_time < self._cache_ttl:
                return question_type
            else:
                # 캐시 만료
                del self._question_type_cache[query]
        return None
    
    def _cache_question_type(self, query: str, question_type: str):
        """질문 유형 캐시 저장"""
        self._question_type_cache[query] = (question_type, datetime.now())
        
        # 캐시 크기 제한
        if len(self._question_type_cache) > self.max_pattern_history:
            # 가장 오래된 항목 제거
            oldest_key = min(self._question_type_cache.keys(), 
                            key=lambda k: self._question_type_cache[k][1])
            del self._question_type_cache[oldest_key]
    
    def _get_cached_llm_suggestions(self, cache_key: str) -> Optional[List[str]]:
        """LLM 기반 질문 생성 캐시 조회"""
        if cache_key in self._llm_suggestions_cache:
            suggestions, cached_time = self._llm_suggestions_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return suggestions
            else:
                # 캐시 만료
                del self._llm_suggestions_cache[cache_key]
        return None
    
    def _cache_llm_suggestions(self, cache_key: str, suggestions: List[str]):
        """LLM 기반 질문 생성 캐시 저장"""
        self._llm_suggestions_cache[cache_key] = (suggestions, datetime.now())
        
        # 캐시 크기 제한
        if len(self._llm_suggestions_cache) > self.max_pattern_history:
            # 가장 오래된 항목 제거
            oldest_key = min(self._llm_suggestions_cache.keys(), 
                            key=lambda k: self._llm_suggestions_cache[k][1])
            del self._llm_suggestions_cache[oldest_key]
    
    def _generate_llm_suggestions(self, context: ConversationContext) -> List[str]:
        """
        LLM을 사용하여 맥락 기반 후속 질문 생성
        
        Args:
            context: 대화 맥락
        
        Returns:
            List[str]: LLM이 생성한 후속 질문 목록
        """
        try:
            if not context.turns:
                return []
            
            last_turn = context.turns[-1]
            
            # 캐시 키 생성 (최근 질문과 답변 기반)
            cache_key = f"{last_turn.user_query[:100]}_{last_turn.bot_response[:100]}"
            cached_suggestions = self._get_cached_llm_suggestions(cache_key)
            if cached_suggestions:
                self.logger.debug(f"Using cached LLM suggestions for query: {last_turn.user_query[:50]}...")
                return cached_suggestions
            
            # 대화 맥락 구성
            conversation_history = []
            for turn in context.turns[-3:]:  # 최근 3턴
                conversation_history.append(f"사용자: {turn.user_query}")
                if turn.bot_response:
                    bot_response_short = turn.bot_response[:200] + "..." if len(turn.bot_response) > 200 else turn.bot_response
                    conversation_history.append(f"봇: {bot_response_short}")
            
            conversation_text = "\n".join(conversation_history)
            
            # LLM 프롬프트 생성
            prompt = f"""다음 법률 상담 대화를 분석하여 사용자가 다음에 물어볼 수 있는 후속 질문 3-5개를 생성해주세요.

대화 내용:
{conversation_text}

요구사항:
1. 대화 맥락을 고려하여 자연스러운 후속 질문 생성
2. 법률 관련 질문이어야 함
3. 구체적이고 실용적인 질문 생성
4. 각 질문은 한 문장으로 작성
5. JSON 형식으로 응답

응답 형식:
{{
    "suggestions": [
        "질문 1",
        "질문 2",
        "질문 3"
    ]
}}"""
            
            # LLM 호출
            handler = self._get_classification_handler()
            if not handler or not hasattr(handler, 'llm'):
                self.logger.debug("LLM not available for suggestions generation")
                return []
            
            response = handler.llm.invoke(prompt)
            response_content = self._extract_response_content(response)
            
            # JSON 파싱
            try:
                result = json.loads(response_content)
                suggestions = result.get("suggestions", [])
                if isinstance(suggestions, list) and len(suggestions) > 0:
                    filtered_suggestions = [str(s).strip() for s in suggestions if s and str(s).strip()]
                    if filtered_suggestions:
                        # 캐시 저장
                        self._cache_llm_suggestions(cache_key, filtered_suggestions)
                        self.logger.info(f"Generated {len(filtered_suggestions)} LLM-based suggestions")
                        return filtered_suggestions
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 추출
                suggestions = self._extract_suggestions_from_text(response_content)
                if suggestions:
                    self._cache_llm_suggestions(cache_key, suggestions)
                    return suggestions
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Failed to generate LLM suggestions: {e}")
            return []
    
    def _extract_response_content(self, response: Any) -> str:
        """LLM 응답에서 텍스트 추출"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _analyze_conversation_comprehensive_with_llm(self, context: ConversationContext) -> Dict[str, Any]:
        """
        LLM을 한 번 호출하여 대화 맥락에서 모든 정보를 통합 추출
        
        한 번의 질의로 추출하는 정보:
        - 질문 유형 (question_type)
        - 분기점 유형 (branch_type)
        - 엔티티 (entities: laws, precedents, legal_terms)
        - 후속 질문 제안 (suggestions)
        - 신뢰도 (confidence)
        
        Args:
            context: 대화 맥락
        
        Returns:
            Dict[str, Any]: 통합 분석 결과
        """
        try:
            if not context.turns:
                return {}
            
            last_turn = context.turns[-1]
            
            # 캐시 키 생성 (문자열로 변환 후 슬라이스)
            user_query_str = str(last_turn.user_query) if last_turn.user_query else ""
            bot_response_str = str(last_turn.bot_response) if last_turn.bot_response else ""
            cache_key = f"comprehensive:{user_query_str[:100]}_{bot_response_str[:100]}"
            
            # 캐시 확인
            if cache_key in self._comprehensive_analysis_cache:
                cached_result, cached_time = self._comprehensive_analysis_cache[cache_key]
                if datetime.now() - cached_time < self._cache_ttl:
                    self.logger.debug("Using cached comprehensive analysis")
                    return cached_result
                else:
                    del self._comprehensive_analysis_cache[cache_key]
            
            # 대화 맥락 구성
            conversation_history = []
            for turn in context.turns[-3:]:  # 최근 3턴
                conversation_history.append(f"사용자: {turn.user_query}")
                if turn.bot_response:
                    bot_response_short = turn.bot_response[:200] + "..." if len(turn.bot_response) > 200 else turn.bot_response
                    conversation_history.append(f"봇: {bot_response_short}")
            
            conversation_text = "\n".join(conversation_history)
            
            # 통합 프롬프트 생성
            prompt = f"""다음 법률 상담 대화를 종합적으로 분석하여 모든 정보를 추출해주세요.

대화 내용:
{conversation_text}

## 분석 항목

### 1. 질문 유형 분류
다음 유형 중 가장 적합한 하나를 선택하세요:
- precedent_search: 판례, 사건, 법원 판결, 판시사항 관련
- law_inquiry: 법률 조문, 법령, 규정의 내용을 묻는 질문
- legal_advice: 법률 조언, 해석, 권리 구제 방법을 묻는 질문
- procedure_guide: 법적 절차, 소송 방법, 대응 방법을 묻는 질문
- term_explanation: 법률 용어의 정의나 의미를 묻는 질문
- general_question: 범용적인 법률 질문

### 2. 분기점 감지
다음 분기점 유형 중 해당하는 것이 있으면 선택하세요 (없으면 null):
- detailed_explanation: "자세히", "구체적으로", "더", "추가로" 등의 키워드
- related_topics: "관련", "연관", "비슷한", "다른" 등의 키워드
- practical_application: "실제", "실무", "현실", "구체적" 등의 키워드
- comparison: "비교", "차이", "다른", "반대" 등의 키워드

### 3. 엔티티 추출
대화에서 언급된 법률 관련 엔티티를 추출하세요:
- laws: 법률명 (예: "민법", "형법", "상법")
- precedents: 판례 관련 키워드 (예: "대법원", "판결", "사건")
- legal_terms: 법률 용어 (예: "손해배상", "계약", "소송")

### 4. 후속 질문 제안
대화 맥락을 고려하여 사용자가 다음에 물어볼 수 있는 후속 질문 5개를 생성하세요.
- 대화 맥락을 고려하여 자연스러운 질문 생성
- 법률 관련 질문이어야 함
- 구체적이고 실용적인 질문 생성
- 각 질문은 한 문장으로 작성

## 응답 형식 (JSON)
{{
    "question_type": "law_inquiry" | "procedure_guide" | "legal_advice" | "precedent_search" | "term_explanation" | "general_question",
    "branch_type": "detailed_explanation" | "related_topics" | "practical_application" | "comparison" | null,
    "entities": {{
        "laws": ["법률명1", "법률명2"],
        "precedents": ["판례 키워드1", "판례 키워드2"],
        "legal_terms": ["용어1", "용어2"]
    }},
    "suggestions": [
        "후속 질문 1",
        "후속 질문 2",
        "후속 질문 3",
        "후속 질문 4",
        "후속 질문 5"
    ],
    "confidence": 0.0-1.0,
    "reasoning": "분석 근거 (한국어)"
}}"""
            
            # LLM 호출
            handler = self._get_classification_handler()
            if not handler or not hasattr(handler, 'llm'):
                self.logger.debug("LLM not available for comprehensive analysis")
                return {}
            
            response = handler.llm.invoke(prompt)
            response_content = self._extract_response_content(response)
            
            # JSON 파싱
            try:
                result = json.loads(response_content)
                
                # 결과 검증 및 정규화
                validated_result = {
                    "question_type": result.get("question_type", "general_question"),
                    "branch_type": result.get("branch_type"),
                    "entities": {
                        "laws": result.get("entities", {}).get("laws", []),
                        "precedents": result.get("entities", {}).get("precedents", []),
                        "legal_terms": result.get("entities", {}).get("legal_terms", [])
                    },
                    "suggestions": result.get("suggestions", []),
                    "confidence": result.get("confidence", 0.8),
                    "reasoning": result.get("reasoning", "")
                }
                
                # 캐시 저장
                self._comprehensive_analysis_cache[cache_key] = (validated_result, datetime.now())
                
                # 캐시 크기 제한
                if len(self._comprehensive_analysis_cache) > self.max_pattern_history:
                    oldest_key = min(self._comprehensive_analysis_cache.keys(), 
                                    key=lambda k: self._comprehensive_analysis_cache[k][1])
                    del self._comprehensive_analysis_cache[oldest_key]
                
                self.logger.info(
                    f"Comprehensive analysis completed: "
                    f"question_type={validated_result['question_type']}, "
                    f"branch_type={validated_result['branch_type']}, "
                    f"suggestions_count={len(validated_result['suggestions'])}"
                )
                
                return validated_result
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse comprehensive analysis JSON: {e}")
                # Fallback: 텍스트에서 추출 시도
                return self._extract_comprehensive_analysis_from_text(response_content)
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}", exc_info=True)
            return {}
    
    def _extract_comprehensive_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 통합 분석 결과 추출 (JSON 파싱 실패 시 Fallback)
        
        Args:
            text: LLM 응답 텍스트
        
        Returns:
            Dict[str, Any]: 추출된 분석 결과
        """
        try:
            result = {
                "question_type": "general_question",
                "branch_type": None,
                "entities": {
                    "laws": [],
                    "precedents": [],
                    "legal_terms": []
                },
                "suggestions": [],
                "confidence": 0.5,
                "reasoning": ""
            }
            
            # 질문 유형 추출
            question_type_patterns = {
                "precedent_search": r"precedent_search|판례|사건|판결",
                "law_inquiry": r"law_inquiry|법률|법령|조문",
                "legal_advice": r"legal_advice|조언|권리",
                "procedure_guide": r"procedure_guide|절차|소송",
                "term_explanation": r"term_explanation|용어|정의"
            }
            
            for q_type, pattern in question_type_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    result["question_type"] = q_type
                    break
            
            # 분기점 추출
            branch_patterns = {
                "detailed_explanation": r"detailed_explanation|자세히|구체적으로",
                "related_topics": r"related_topics|관련|연관",
                "practical_application": r"practical_application|실제|실무",
                "comparison": r"comparison|비교|차이"
            }
            
            for branch_type, pattern in branch_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    result["branch_type"] = branch_type
                    break
            
            # 후속 질문 추출
            suggestions = self._extract_suggestions_from_text(text)
            result["suggestions"] = suggestions
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error extracting comprehensive analysis from text: {e}")
            return {}
    
    def _generate_entity_suggestions_from_entities(self, entities: Dict[str, List[str]]) -> List[str]:
        """
        엔티티 딕셔너리에서 제안 생성
        
        Args:
            entities: 엔티티 딕셔너리 (laws, precedents, legal_terms)
        
        Returns:
            List[str]: 엔티티 기반 제안 목록
        """
        try:
            suggestions = []
            
            # 법률 기반 제안
            if entities.get("laws"):
                suggestions.append("관련 조문도 함께 설명해드릴까요?")
                suggestions.append("이 법률의 적용 범위는 어떻게 되나요?")
            
            # 판례 기반 제안
            if entities.get("precedents"):
                suggestions.append("유사한 판례도 찾아드릴까요?")
                suggestions.append("이 판례의 핵심 쟁점은 무엇인가요?")
            
            # 법률 용어 기반 제안
            if entities.get("legal_terms"):
                suggestions.append("관련 용어들도 함께 설명해드릴까요?")
                suggestions.append("이 용어의 정의를 더 자세히 알려드릴까요?")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating entity suggestions from entities: {e}")
            return []
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """텍스트에서 질문 추출 (JSON 파싱 실패 시)"""
        suggestions = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or '질문' in line or len(line) > 10):
                # 번호나 불필요한 문자 제거
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                line = re.sub(r'^["\']|["\']$', '', line)  # 따옴표 제거
                if line and line not in suggestions and len(line) > 5:
                    suggestions.append(line)
        return suggestions[:5]


# 테스트 함수
def test_conversation_flow_tracker():
    """대화 흐름 추적기 테스트"""
    tracker = ConversationFlowTracker()
    
    print("=== 대화 흐름 추적기 테스트 ===")
    
    # 테스트 세션 생성
    from .conversation_manager import ConversationContext, ConversationTurn
    
    context = ConversationContext(
        session_id="test_session",
        turns=[],
        entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
        topic_stack=[],
        created_at=datetime.now(),
        last_updated=datetime.now()
    )
    
    # 테스트 턴들
    test_turns = [
        ConversationTurn(
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따라 손해배상 청구는...",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        ),
        ConversationTurn(
            user_query="더 자세히 설명해주세요",
            bot_response="구체적으로 말씀드리면...",
            timestamp=datetime.now(),
            question_type="clarification"
        ),
        ConversationTurn(
            user_query="관련 판례도 알려주세요",
            bot_response="관련 판례는 다음과 같습니다...",
            timestamp=datetime.now(),
            question_type="follow_up"
        )
    ]
    
    # 흐름 추적 테스트
    print("\n1. 흐름 추적 테스트")
    for turn in test_turns:
        context.turns.append(turn)
        tracker.track_conversation_flow(context.session_id, turn)
        print(f"   턴 추가: {turn.user_query}")
    
    # 다음 의도 예측 테스트
    print("\n2. 다음 의도 예측 테스트")
    predicted_intents = tracker.predict_next_intent(context)
    print(f"   예측된 의도: {predicted_intents}")
    
    # 후속 질문 제안 테스트
    print("\n3. 후속 질문 제안 테스트")
    suggestions = tracker.suggest_follow_up_questions(context)
    print(f"   제안된 질문: {suggestions}")
    
    # 분기점 감지 테스트
    print("\n4. 분기점 감지 테스트")
    branch = tracker.detect_conversation_branch("더 자세히 설명해주세요")
    if branch:
        print(f"   감지된 분기점: {branch.branch_type}")
        print(f"   제안: {branch.follow_up_suggestions}")
    
    # 대화 상태 파악 테스트
    print("\n5. 대화 상태 파악 테스트")
    state = tracker.get_conversation_state(context)
    print(f"   현재 상태: {state}")
    
    # 패턴 분석 테스트
    print("\n6. 패턴 분석 테스트")
    pattern_analysis = tracker.analyze_flow_patterns([context])
    print(f"   패턴 분석 결과: {pattern_analysis}")
    
    print("\n테스트 완료")


if __name__ == "__main__":
    test_conversation_flow_tracker()
