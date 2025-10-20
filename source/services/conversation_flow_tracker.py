# -*- coding: utf-8 -*-
"""
대화 흐름 추적기
대화 패턴을 학습하고 다음 질문을 예측하여 후속 질문을 제안합니다.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re

from .conversation_manager import ConversationContext, ConversationTurn
from .emotion_intent_analyzer import EmotionIntentAnalyzer, IntentType, EmotionType

logger = logging.getLogger(__name__)


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


@dataclass
class FlowPrediction:
    """흐름 예측 결과"""
    predicted_intents: List[str]
    suggested_questions: List[str]
    confidence: float
    reasoning: str


class ConversationFlowTracker:
    """대화 흐름 추적기"""
    
    def __init__(self, max_pattern_history: int = 1000):
        """
        대화 흐름 추적기 초기화
        
        Args:
            max_pattern_history: 최대 패턴 이력 수
        """
        self.logger = logging.getLogger(__name__)
        self.max_pattern_history = max_pattern_history
        
        # 감정/의도 분석기
        self.emotion_intent_analyzer = EmotionIntentAnalyzer()
        
        # 대화 패턴 저장소
        self.flow_patterns: Dict[str, FlowPattern] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        
        # 대화 분기점 저장소
        self.conversation_branches: Dict[str, ConversationBranch] = {}
        
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
        후속 질문 제안
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 제안된 후속 질문 목록
        """
        try:
            if not context.turns:
                return []
            
            suggestions = []
            
            # 최근 질문 유형 기반 제안
            recent_question_types = []
            for turn in context.turns[-2:]:  # 최근 2턴
                if turn.question_type:
                    recent_question_types.append(turn.question_type)
            
            # 질문 유형별 템플릿 제안
            for question_type in recent_question_types:
                if question_type in self.follow_up_templates:
                    suggestions.extend(self.follow_up_templates[question_type])
            
            # 분기점 기반 제안
            last_turn = context.turns[-1]
            branch = self.detect_conversation_branch(last_turn.user_query)
            if branch:
                suggestions.extend(branch.follow_up_suggestions)
            
            # 엔티티 기반 제안
            entity_suggestions = self._generate_entity_based_suggestions(context)
            suggestions.extend(entity_suggestions)
            
            # 중복 제거 및 정렬
            unique_suggestions = list(dict.fromkeys(suggestions))
            return unique_suggestions[:5]  # 최대 5개
            
        except Exception as e:
            self.logger.error(f"Error suggesting follow-up questions: {e}")
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
