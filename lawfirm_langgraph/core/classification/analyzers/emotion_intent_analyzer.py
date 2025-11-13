# -*- coding: utf-8 -*-
"""
감정 및 의도 분석기
사용자 감정과 의도를 파악하여 적절한 응답 톤 결정
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.conversation.conversation_manager import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """감정 유형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    URGENT = "urgent"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


class IntentType(Enum):
    """의도 유형"""
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    THANKS = "thanks"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    GENERAL = "general"


class UrgencyLevel(Enum):
    """긴급도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmotionAnalysis:
    """감정 분석 결과"""
    primary_emotion: EmotionType
    emotion_scores: Dict[str, float]
    confidence: float
    intensity: float
    reasoning: str


@dataclass
class IntentAnalysis:
    """의도 분석 결과"""
    primary_intent: IntentType
    intent_scores: Dict[str, float]
    confidence: float
    urgency_level: UrgencyLevel
    reasoning: str


@dataclass
class ResponseTone:
    """응답 톤"""
    tone_type: str
    empathy_level: float
    formality_level: float
    urgency_response: bool
    explanation_depth: str


class EmotionIntentAnalyzer:
    """감정 및 의도 분석기"""
    
    def __init__(self):
        """감정 및 의도 분석기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 감정 키워드 패턴
        self.emotion_patterns = {
            EmotionType.POSITIVE: [
                r"감사|고마워|좋아|훌륭|완벽|만족|기쁘|행복|좋은",
                r"도움|도움이|해결|해결됨|이해|이해됨|명확|명확해"
            ],
            EmotionType.NEGATIVE: [
                r"안좋|나쁘|문제|문제가|어려워|힘들|복잡|복잡해",
                r"이해안|모르겠|모르겠어|알겠|알겠어|어떻게|어떡해"
            ],
            EmotionType.URGENT: [
                r"급해|급함|빨리|빨리빨리|즉시|당장|지금|지금당장",
                r"긴급|긴급해|시급|시급해|마감|마감이|데드라인",
                r"오늘|내일|내일까지|오늘까지|시간|시간이|시간없"
            ],
            EmotionType.ANXIOUS: [
                r"걱정|걱정돼|불안|불안해|무서워|두려워|걱정스러워",
                r"어떻게|어떡해|어떻게해|어떻게해야|어떻게하죠",
                r"괜찮|괜찮아|괜찮을까|문제없|문제없을까"
            ],
            EmotionType.ANGRY: [
                r"화나|화났|짜증|짜증나|답답|답답해|속상|속상해",
                r"이상해|이상하|말이안|말이안돼|말이안되|말이안됨",
                r"왜|왜그래|왜이래|왜이런|왜이런거야"
            ],
            EmotionType.SATISFIED: [
                r"만족|만족해|좋아|좋네|훌륭|완벽|완벽해|완료|완료됨",
                r"해결|해결됨|해결됐|이해|이해됨|이해됐|명확|명확해"
            ],
            EmotionType.CONFUSED: [
                r"헷갈|헷갈려|혼란|혼란스러워|모르겠|모르겠어|알겠",
                r"이해안|이해안돼|이해안되|이해안됨|복잡|복잡해",
                r"어떻게|어떡해|어떻게해|어떻게해야|어떻게하죠"
            ]
        }
        
        # 의도 키워드 패턴
        self.intent_patterns = {
            IntentType.QUESTION: [
                r"무엇|뭐|어떤|어떻게|언제|어디서|왜|어떤|어떤것",
                r"알려주|가르쳐|설명해|말해|알고싶|궁금|궁금해"
            ],
            IntentType.REQUEST: [
                r"해주|해줘|도와|도와줘|부탁|부탁해|요청|요청해",
                r"작성해|만들어|준비해|처리해|해결해|진행해"
            ],
            IntentType.COMPLAINT: [
                r"문제|문제가|이상해|이상하|말이안|말이안돼|말이안되",
                r"왜|왜그래|왜이래|왜이런|왜이런거야|이상|이상해"
            ],
            IntentType.THANKS: [
                r"감사|고마워|고맙|고맙습니다|감사합니다|고마워요",
                r"도움|도움이|도움이됐|도움이되었|도움이되었어"
            ],
            IntentType.CLARIFICATION: [
                r"정확히|정확한|구체적|구체적으로|자세히|자세한",
                r"예시|예를|예를들어|즉|다시|다시한번|다시한번더"
            ],
            IntentType.FOLLOW_UP: [
                r"추가|추가로|또|또한|그리고|그리고|더|더자세히",
                r"이어서|계속|계속해서|다음|다음으로|그다음"
            ],
            IntentType.EMERGENCY: [
                r"긴급|긴급해|급해|급함|즉시|당장|지금|지금당장",
                r"마감|마감이|데드라인|시간|시간이|시간없|오늘|내일"
            ]
        }
        
        # 긴급도 평가 패턴
        self.urgency_patterns = {
            UrgencyLevel.CRITICAL: [
                r"지금당장|즉시|당장|긴급|긴급해|마감|마감이|데드라인",
                r"오늘까지|내일까지|시간없|시간이|시간이없"
            ],
            UrgencyLevel.HIGH: [
                r"빨리|빨리빨리|급해|급함|시급|시급해|오늘|내일",
                r"가능한빨리|최대한빨리|빨리빨리|어서|어서빨리"
            ],
            UrgencyLevel.MEDIUM: [
                r"가능하면|가능한|시간되면|시간날때|여유있을때",
                r"나중에|나중에라도|언젠가|언젠가는|기회되면"
            ],
            UrgencyLevel.LOW: [
                r"천천히|여유있게|시간있을때|시간날때|나중에",
                r"급하지|급하지않|급하지않아|급하지않음"
            ]
        }
        
        # 응답 톤 패턴
        self.response_tone_patterns = {
            "empathetic": {
                "keywords": ["걱정", "불안", "어려워", "힘들", "문제"],
                "empathy_level": 0.8,
                "formality_level": 0.6
            },
            "professional": {
                "keywords": ["법률", "법령", "조문", "판례", "법원"],
                "empathy_level": 0.3,
                "formality_level": 0.9
            },
            "supportive": {
                "keywords": ["도움", "해결", "이해", "명확", "설명"],
                "empathy_level": 0.7,
                "formality_level": 0.5
            },
            "urgent": {
                "keywords": ["급해", "긴급", "빨리", "즉시", "당장"],
                "empathy_level": 0.4,
                "formality_level": 0.7
            },
            "casual": {
                "keywords": ["궁금", "알고싶", "좋아", "만족", "감사"],
                "empathy_level": 0.6,
                "formality_level": 0.3
            }
        }
        
        self.logger.info("EmotionIntentAnalyzer initialized")
    
    def analyze_emotion(self, text: str) -> EmotionAnalysis:
        """
        감정 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            EmotionAnalysis: 감정 분석 결과
        """
        try:
            text_lower = text.lower()
            emotion_scores = {}
            
            # 각 감정 유형별 점수 계산
            for emotion_type, patterns in self.emotion_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    score += len(matches) * 0.1
                
                # 패턴 길이에 따른 가중치
                if score > 0:
                    score += 0.1
                
                emotion_scores[emotion_type.value] = min(1.0, score)
            
            # 기본 감정 설정 (점수가 모두 낮으면 중립)
            if not any(score > 0.1 for score in emotion_scores.values()):
                emotion_scores[EmotionType.NEUTRAL.value] = 0.5
            
            # 주요 감정 결정
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            primary_emotion_type = EmotionType(primary_emotion[0])
            
            # 신뢰도 계산
            confidence = primary_emotion[1]
            
            # 강도 계산
            intensity = self._calculate_emotion_intensity(text_lower)
            
            # 추론 과정 생성
            reasoning = self._generate_emotion_reasoning(text, primary_emotion_type, emotion_scores)
            
            return EmotionAnalysis(
                primary_emotion=primary_emotion_type,
                emotion_scores=emotion_scores,
                confidence=confidence,
                intensity=intensity,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing emotion: {e}")
            return EmotionAnalysis(
                primary_emotion=EmotionType.NEUTRAL,
                emotion_scores={EmotionType.NEUTRAL.value: 1.0},
                confidence=0.0,
                intensity=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def analyze_intent(self, text: str, context: Optional[ConversationContext] = None) -> IntentAnalysis:
        """
        의도 분석
        
        Args:
            text: 분석할 텍스트
            context: 대화 맥락 (선택사항)
            
        Returns:
            IntentAnalysis: 의도 분석 결과
        """
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            # 각 의도 유형별 점수 계산
            for intent_type, patterns in self.intent_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    score += len(matches) * 0.1
                
                # 패턴 길이에 따른 가중치
                if score > 0:
                    score += 0.1
                
                intent_scores[intent_type.value] = min(1.0, score)
            
            # 맥락 기반 의도 조정
            if context:
                intent_scores = self._adjust_intent_with_context(intent_scores, context)
            
            # 기본 의도 설정 (점수가 모두 낮으면 일반)
            if not any(score > 0.1 for score in intent_scores.values()):
                intent_scores[IntentType.GENERAL.value] = 0.5
            
            # 주요 의도 결정
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            primary_intent_type = IntentType(primary_intent[0])
            
            # 신뢰도 계산
            confidence = primary_intent[1]
            
            # 긴급도 평가
            urgency_level = self.assess_urgency(text, {})
            
            # 추론 과정 생성
            reasoning = self._generate_intent_reasoning(text, primary_intent_type, intent_scores)
            
            return IntentAnalysis(
                primary_intent=primary_intent_type,
                intent_scores=intent_scores,
                confidence=confidence,
                urgency_level=urgency_level,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            return IntentAnalysis(
                primary_intent=IntentType.GENERAL,
                intent_scores={IntentType.GENERAL.value: 1.0},
                confidence=0.0,
                urgency_level=UrgencyLevel.LOW,
                reasoning=f"Error: {str(e)}"
            )
    
    def get_contextual_response_tone(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                     user_profile: Optional[Dict] = None) -> ResponseTone:
        """
        맥락적 응답 톤 결정
        
        Args:
            emotion: 감정 분석 결과
            intent: 의도 분석 결과
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            ResponseTone: 응답 톤
        """
        try:
            # 기본 톤 결정
            tone_type = self._determine_base_tone(emotion, intent)
            
            # 공감 수준 계산
            empathy_level = self._calculate_empathy_level(emotion, intent)
            
            # 격식 수준 계산
            formality_level = self._calculate_formality_level(emotion, intent, user_profile)
            
            # 긴급 응답 여부
            urgency_response = intent.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
            
            # 설명 깊이 결정
            explanation_depth = self._determine_explanation_depth(emotion, intent, user_profile)
            
            return ResponseTone(
                tone_type=tone_type,
                empathy_level=empathy_level,
                formality_level=formality_level,
                urgency_response=urgency_response,
                explanation_depth=explanation_depth
            )
            
        except Exception as e:
            self.logger.error(f"Error determining response tone: {e}")
            return ResponseTone(
                tone_type="professional",
                empathy_level=0.5,
                formality_level=0.7,
                urgency_response=False,
                explanation_depth="medium"
            )
    
    def assess_urgency(self, text: str, emotion: Dict[str, float]) -> UrgencyLevel:
        """
        긴급도 평가
        
        Args:
            text: 분석할 텍스트
            emotion: 감정 점수
            
        Returns:
            UrgencyLevel: 긴급도 수준
        """
        try:
            text_lower = text.lower()
            
            # 긴급도 패턴 매칭
            for urgency_level, patterns in self.urgency_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return urgency_level
            
            # 감정 기반 긴급도 조정
            if emotion.get(EmotionType.URGENT.value, 0) > 0.5:
                return UrgencyLevel.HIGH
            elif emotion.get(EmotionType.ANXIOUS.value, 0) > 0.7:
                return UrgencyLevel.MEDIUM
            
            return UrgencyLevel.LOW
            
        except Exception as e:
            self.logger.error(f"Error assessing urgency: {e}")
            return UrgencyLevel.LOW
    
    def _calculate_emotion_intensity(self, text_lower: str) -> float:
        """감정 강도 계산"""
        try:
            intensity_indicators = [
                r"정말|정말로|진짜|진짜로|완전|완전히|너무|너무나",
                r"매우|엄청|엄청나|대단|대단히|극도로|극도로",
                r"!!+|!!!+|!!!+",  # 느낌표 반복
                r"ㅠㅠ|ㅜㅜ|ㅠㅠㅠ|ㅜㅜㅜ",  # 울음 표시
                r"ㅋㅋ|ㅎㅎ|ㅋㅋㅋ|ㅎㅎㅎ"  # 웃음 표시
            ]
            
            intensity = 0.0
            for pattern in intensity_indicators:
                matches = re.findall(pattern, text_lower)
                intensity += len(matches) * 0.1
            
            return min(1.0, intensity)
            
        except Exception as e:
            self.logger.error(f"Error calculating emotion intensity: {e}")
            return 0.0
    
    def _adjust_intent_with_context(self, intent_scores: Dict[str, float], 
                                   context: ConversationContext) -> Dict[str, float]:
        """맥락을 고려한 의도 점수 조정"""
        try:
            # 최근 대화에서 의도 패턴 분석
            if context.turns:
                recent_turns = context.turns[-3:]  # 최근 3턴
                
                # 후속 질문 패턴 감지
                if len(recent_turns) > 1:
                    last_turn = recent_turns[-1]
                    if any(keyword in last_turn.user_query.lower() 
                          for keyword in ["추가", "더", "또한", "그리고"]):
                        intent_scores[IntentType.FOLLOW_UP.value] += 0.2
                
                # 명확화 요청 패턴 감지
                if any(keyword in context.turns[-1].user_query.lower() 
                      for keyword in ["정확히", "구체적으로", "자세히", "예시"]):
                    intent_scores[IntentType.CLARIFICATION.value] += 0.2
            
            return intent_scores
            
        except Exception as e:
            self.logger.error(f"Error adjusting intent with context: {e}")
            return intent_scores
    
    def _determine_base_tone(self, emotion: EmotionAnalysis, intent: IntentAnalysis) -> str:
        """기본 톤 결정"""
        try:
            # 긴급한 경우
            if intent.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
                return "urgent"
            
            # 감사 표현인 경우
            if intent.primary_intent == IntentType.THANKS:
                return "supportive"
            
            # 불만/화남인 경우
            if emotion.primary_emotion in [EmotionType.ANGRY, EmotionType.NEGATIVE]:
                return "empathetic"
            
            # 불안/걱정인 경우
            if emotion.primary_emotion in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                return "empathetic"
            
            # 법률 관련 질문인 경우
            if intent.primary_intent == IntentType.QUESTION:
                return "professional"
            
            # 기본값
            return "professional"
            
        except Exception as e:
            self.logger.error(f"Error determining base tone: {e}")
            return "professional"
    
    def _calculate_empathy_level(self, emotion: EmotionAnalysis, intent: IntentAnalysis) -> float:
        """공감 수준 계산"""
        try:
            empathy_level = 0.5  # 기본값
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                empathy_level += 0.3
            elif emotion.primary_emotion in [EmotionType.ANGRY, EmotionType.NEGATIVE]:
                empathy_level += 0.2
            elif emotion.primary_emotion == EmotionType.POSITIVE:
                empathy_level += 0.1
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.COMPLAINT:
                empathy_level += 0.2
            elif intent.primary_intent == IntentType.THANKS:
                empathy_level += 0.1
            
            return min(1.0, empathy_level)
            
        except Exception as e:
            self.logger.error(f"Error calculating empathy level: {e}")
            return 0.5
    
    def _calculate_formality_level(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                  user_profile: Optional[Dict]) -> float:
        """격식 수준 계산"""
        try:
            formality_level = 0.7  # 기본값 (법률 상담이므로 격식적)
            
            # 사용자 프로필에 따른 조정
            if user_profile:
                expertise_level = user_profile.get("expertise_level", "beginner")
                if expertise_level == "expert":
                    formality_level += 0.2
                elif expertise_level == "beginner":
                    formality_level -= 0.1
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.POSITIVE, EmotionType.SATISFIED]:
                formality_level -= 0.1
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.THANKS:
                formality_level -= 0.1
            
            return max(0.0, min(1.0, formality_level))
            
        except Exception as e:
            self.logger.error(f"Error calculating formality level: {e}")
            return 0.7
    
    def _determine_explanation_depth(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                    user_profile: Optional[Dict]) -> str:
        """설명 깊이 결정"""
        try:
            # 사용자 프로필에 따른 조정
            if user_profile:
                detail_level = user_profile.get("preferred_detail_level", "medium")
                if detail_level == "detailed":
                    return "detailed"
                elif detail_level == "simple":
                    return "simple"
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.CLARIFICATION:
                return "detailed"
            elif intent.primary_intent == IntentType.EMERGENCY:
                return "simple"
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.CONFUSED, EmotionType.ANXIOUS]:
                return "detailed"
            elif emotion.primary_emotion == EmotionType.URGENT:
                return "simple"
            
            return "medium"
            
        except Exception as e:
            self.logger.error(f"Error determining explanation depth: {e}")
            return "medium"
    
    def _generate_emotion_reasoning(self, text: str, primary_emotion: EmotionType, 
                                   emotion_scores: Dict[str, float]) -> str:
        """감정 추론 과정 생성"""
        try:
            reasoning_parts = []
            
            # 주요 감정 설명
            emotion_descriptions = {
                EmotionType.POSITIVE: "긍정적 감정",
                EmotionType.NEGATIVE: "부정적 감정",
                EmotionType.NEUTRAL: "중립적 감정",
                EmotionType.URGENT: "긴급한 감정",
                EmotionType.ANXIOUS: "불안한 감정",
                EmotionType.ANGRY: "화난 감정",
                EmotionType.SATISFIED: "만족한 감정",
                EmotionType.CONFUSED: "혼란스러운 감정"
            }
            
            reasoning_parts.append(f"주요 감정: {emotion_descriptions.get(primary_emotion, '알 수 없음')}")
            
            # 높은 점수의 감정들
            high_score_emotions = [emotion for emotion, score in emotion_scores.items() 
                                  if score > 0.3 and emotion != primary_emotion.value]
            if high_score_emotions:
                reasoning_parts.append(f"기타 감정: {', '.join(high_score_emotions)}")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating emotion reasoning: {e}")
            return f"Error: {str(e)}"
    
    def _generate_intent_reasoning(self, text: str, primary_intent: IntentType, 
                                  intent_scores: Dict[str, float]) -> str:
        """의도 추론 과정 생성"""
        try:
            reasoning_parts = []
            
            # 주요 의도 설명
            intent_descriptions = {
                IntentType.QUESTION: "질문 의도",
                IntentType.REQUEST: "요청 의도",
                IntentType.COMPLAINT: "불만 의도",
                IntentType.THANKS: "감사 의도",
                IntentType.CLARIFICATION: "명확화 의도",
                IntentType.FOLLOW_UP: "후속 질문 의도",
                IntentType.EMERGENCY: "긴급 의도",
                IntentType.GENERAL: "일반 의도"
            }
            
            reasoning_parts.append(f"주요 의도: {intent_descriptions.get(primary_intent, '알 수 없음')}")
            
            # 높은 점수의 의도들
            high_score_intents = [intent for intent, score in intent_scores.items() 
                                 if score > 0.3 and intent != primary_intent.value]
            if high_score_intents:
                reasoning_parts.append(f"기타 의도: {', '.join(high_score_intents)}")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating intent reasoning: {e}")
            return f"Error: {str(e)}"


# 테스트 함수
def test_emotion_intent_analyzer():
    """감정 및 의도 분석기 테스트"""
    analyzer = EmotionIntentAnalyzer()
    
    print("=== 감정 및 의도 분석기 테스트 ===")
    
    # 테스트 텍스트들
    test_texts = [
        "손해배상 청구 방법을 알려주세요",
        "급해요! 오늘까지 답변해주세요!",
        "감사합니다. 정말 도움이 되었어요",
        "이해가 안 돼요. 더 자세히 설명해주세요",
        "왜 이런 문제가 생겼나요? 정말 화나네요",
        "추가로 궁금한 것이 있어요",
        "정확히 말하면 어떤 절차인가요?"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 텍스트: {text}")
        
        # 감정 분석
        emotion_result = analyzer.analyze_emotion(text)
        print(f"   감정: {emotion_result.primary_emotion.value} (신뢰도: {emotion_result.confidence:.2f})")
        print(f"   강도: {emotion_result.intensity:.2f}")
        print(f"   추론: {emotion_result.reasoning}")
        
        # 의도 분석
        intent_result = analyzer.analyze_intent(text)
        print(f"   의도: {intent_result.primary_intent.value} (신뢰도: {intent_result.confidence:.2f})")
        print(f"   긴급도: {intent_result.urgency_level.value}")
        print(f"   추론: {intent_result.reasoning}")
        
        # 응답 톤 결정
        response_tone = analyzer.get_contextual_response_tone(emotion_result, intent_result)
        print(f"   응답 톤: {response_tone.tone_type}")
        print(f"   공감 수준: {response_tone.empathy_level:.2f}")
        print(f"   격식 수준: {response_tone.formality_level:.2f}")
        print(f"   긴급 응답: {response_tone.urgency_response}")
        print(f"   설명 깊이: {response_tone.explanation_depth}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_emotion_intent_analyzer()
