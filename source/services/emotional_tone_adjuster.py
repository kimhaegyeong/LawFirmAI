# -*- coding: utf-8 -*-
"""
감정 톤 조절기
사용자 질문의 감정 톤에 맞는 답변 톤 조절 시스템
"""

import random
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

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


class ToneIntensity(Enum):
    """톤 강도"""
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"


class EmotionalToneAdjuster:
    """사용자 질문의 감정 톤에 맞는 답변 톤 조절"""
    
    def __init__(self):
        """감정 톤 조절기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 감정별 톤 패턴 정의
        self.tone_patterns = {
            EmotionType.URGENT: {
                "opening": [
                    "바로 답변드릴게요", "급하신 것 같아서", "시급한 문제군요", 
                    "빨리 해결해드릴게요", "지금 당장 도움드릴게요"
                ],
                "closing": [
                    "빨리 해결하시길 바라요", "서둘러 처리하세요", "시간이 중요해요",
                    "즉시 조치하시면 좋겠어요", "급하게 서두르세요"
                ],
                "style": "직접적이고 빠른 조언",
                "keywords": ["바로", "즉시", "빨리", "급하게", "당장"]
            },
            EmotionType.ANXIOUS: {
                "opening": [
                    "걱정하지 마세요", "차근차근 설명드릴게요", "이해하기 쉽게 말씀드릴게요",
                    "단계별로 알아보죠", "천천히 설명해드릴게요"
                ],
                "closing": [
                    "괜찮을 거예요", "걱정하지 마세요", "차근차근 해결해보세요",
                    "천천히 진행하시면 됩니다", "안심하세요"
                ],
                "style": "안심시키고 단계별 설명",
                "keywords": ["차근차근", "천천히", "단계별", "안심", "걱정하지"]
            },
            EmotionType.ANGRY: {
                "opening": [
                    "이해하시는 마음이에요", "답답하시겠어요", "속상하시겠어요",
                    "화나실 만해요", "이해할 수 있어요"
                ],
                "closing": [
                    "차분히 해결해보세요", "급하게 서두르지 마세요", "신중하게 접근하세요",
                    "냉정하게 판단하세요", "여유를 갖고 해결하세요"
                ],
                "style": "공감하고 차분한 조언",
                "keywords": ["이해", "공감", "차분히", "신중하게", "냉정하게"]
            },
            EmotionType.CONFUSED: {
                "opening": [
                    "차근차근 설명드릴게요", "단계별로 알아보죠", "쉽게 설명해드릴게요",
                    "명확하게 정리해드릴게요", "이해하기 쉽게 말씀드릴게요"
                ],
                "closing": [
                    "이해가 되셨나요?", "더 궁금한 점 있으시면 물어보세요", 
                    "명확해지셨길 바라요", "혼란스러우시면 다시 설명드릴게요"
                ],
                "style": "친절하고 상세한 설명",
                "keywords": ["차근차근", "단계별", "명확하게", "이해하기 쉽게"]
            },
            EmotionType.POSITIVE: {
                "opening": [
                    "좋은 질문이네요", "잘 물어보셨어요", "정확한 포인트입니다",
                    "핵심을 짚으셨네요", "훌륭한 질문이에요"
                ],
                "closing": [
                    "도움이 되었길 바라요", "좋은 결과 있으시길 바라요",
                    "성공적으로 해결하시길 바라요", "만족스러운 결과 있으시길 바라요"
                ],
                "style": "긍정적이고 격려하는 톤",
                "keywords": ["좋은", "훌륭한", "정확한", "핵심", "만족스러운"]
            },
            EmotionType.NEUTRAL: {
                "opening": [
                    "질문하신 내용에 대해", "말씀하신 사안에 대해", "문의하신 내용에 대해",
                    "관련해서 말씀드리면", "해당 사안에 대해"
                ],
                "closing": [
                    "도움이 되었길 바라요", "추가 질문 있으시면 언제든지",
                    "필요하시면 더 자세히 설명드릴게요", "궁금한 점 있으시면 물어보세요"
                ],
                "style": "전문적이고 친절한 설명",
                "keywords": ["질문하신", "말씀하신", "문의하신", "관련해서"]
            }
        }
        
        # 감정 강도별 조절 패턴
        self.intensity_patterns = {
            ToneIntensity.MILD: {
                "modifier": "조금", "softener": "~것 같아요", "tone": "부드러운"
            },
            ToneIntensity.MODERATE: {
                "modifier": "", "softener": "", "tone": "표준"
            },
            ToneIntensity.STRONG: {
                "modifier": "정말", "softener": "~네요", "tone": "강한"
            }
        }
        
        # 사용 통계
        self.usage_stats = {}
        
        self.logger.info("EmotionalToneAdjuster initialized")
    
    def adjust_tone(self, answer: str, user_emotion: str, emotion_intensity: float = 0.5) -> str:
        """
        감정에 맞는 톤으로 답변 조절
        
        Args:
            answer: 원본 답변
            user_emotion: 사용자 감정
            emotion_intensity: 감정 강도 (0.0 ~ 1.0)
            
        Returns:
            str: 톤이 조절된 답변
        """
        try:
            if not answer or not isinstance(answer, str):
                return answer
            
            # 감정 유형 변환
            emotion_type = self._parse_emotion_type(user_emotion)
            
            # 강도 결정
            intensity = self._determine_intensity(emotion_intensity)
            
            # 톤 패턴 가져오기
            tone_config = self.tone_patterns.get(emotion_type, self.tone_patterns[EmotionType.NEUTRAL])
            
            # 강도에 따른 조절
            if intensity == ToneIntensity.STRONG:
                answer = self._apply_strong_tone(answer, tone_config)
            elif intensity == ToneIntensity.MILD:
                answer = self._apply_mild_tone(answer, tone_config)
            else:
                answer = self._apply_standard_tone(answer, tone_config)
            
            # 사용 통계 업데이트
            self._track_usage(emotion_type, intensity)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error adjusting tone: {e}")
            return answer
    
    def _parse_emotion_type(self, emotion_str: str) -> EmotionType:
        """문자열을 감정 유형으로 변환"""
        try:
            emotion_mapping = {
                "urgent": EmotionType.URGENT,
                "anxious": EmotionType.ANXIOUS,
                "angry": EmotionType.ANGRY,
                "confused": EmotionType.CONFUSED,
                "positive": EmotionType.POSITIVE,
                "negative": EmotionType.NEGATIVE,
                "satisfied": EmotionType.SATISFIED,
                "neutral": EmotionType.NEUTRAL
            }
            return emotion_mapping.get(emotion_str.lower(), EmotionType.NEUTRAL)
        except Exception:
            return EmotionType.NEUTRAL
    
    def _determine_intensity(self, intensity_score: float) -> ToneIntensity:
        """감정 강도 점수를 톤 강도로 변환"""
        try:
            if intensity_score >= 0.7:
                return ToneIntensity.STRONG
            elif intensity_score <= 0.3:
                return ToneIntensity.MILD
            else:
                return ToneIntensity.MODERATE
        except Exception:
            return ToneIntensity.MODERATE
    
    def _apply_strong_tone(self, answer: str, tone_config: Dict) -> str:
        """강한 감정에 맞는 톤 적용"""
        try:
            opening = random.choice(tone_config["opening"])
            closing = random.choice(tone_config["closing"])
            
            # 강한 톤의 키워드 추가
            keywords = tone_config.get("keywords", [])
            if keywords:
                keyword = random.choice(keywords)
                answer = f"{keyword} {answer}"
            
            return f"{opening}, {answer} {closing}."
            
        except Exception as e:
            self.logger.error(f"Error applying strong tone: {e}")
            return answer
    
    def _apply_standard_tone(self, answer: str, tone_config: Dict) -> str:
        """표준 톤 적용"""
        try:
            opening = random.choice(tone_config["opening"])
            return f"{opening} {answer}"
            
        except Exception as e:
            self.logger.error(f"Error applying standard tone: {e}")
            return answer
    
    def _apply_mild_tone(self, answer: str, tone_config: Dict) -> str:
        """약한 감정에 맞는 톤 적용"""
        try:
            # 부드러운 표현 사용
            soft_openings = [
                "조금", "약간", "살짝", "가볍게"
            ]
            
            opening = random.choice(soft_openings)
            return f"{opening} {answer}"
            
        except Exception as e:
            self.logger.error(f"Error applying mild tone: {e}")
            return answer
    
    def _track_usage(self, emotion_type: EmotionType, intensity: ToneIntensity) -> None:
        """톤 사용 통계 추적"""
        try:
            key = f"{emotion_type.value}_{intensity.value}"
            if key not in self.usage_stats:
                self.usage_stats[key] = 0
            self.usage_stats[key] += 1
        except Exception as e:
            self.logger.error(f"Error tracking usage: {e}")
    
    def get_usage_stats(self) -> Dict[str, int]:
        """톤 사용 통계 반환"""
        return self.usage_stats.copy()
    
    def reset_stats(self) -> None:
        """사용 통계 초기화"""
        self.usage_stats = {}
    
    def analyze_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 감정 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict[str, Any]: 감정 분석 결과
        """
        try:
            text_lower = text.lower()
            
            # 감정 키워드 매칭
            emotion_scores = {}
            for emotion_type, config in self.tone_patterns.items():
                score = 0.0
                keywords = config.get("keywords", [])
                
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 0.2
                
                # 감정별 특수 패턴 검사
                if emotion_type == EmotionType.URGENT:
                    urgent_patterns = [r"급해", r"빨리", r"즉시", r"당장", r"시급"]
                    for pattern in urgent_patterns:
                        if re.search(pattern, text_lower):
                            score += 0.3
                
                elif emotion_type == EmotionType.ANXIOUS:
                    anxious_patterns = [r"걱정", r"불안", r"어떻게", r"모르겠", r"괜찮"]
                    for pattern in anxious_patterns:
                        if re.search(pattern, text_lower):
                            score += 0.3
                
                elif emotion_type == EmotionType.ANGRY:
                    angry_patterns = [r"화나", r"짜증", r"답답", r"속상", r"이상해"]
                    for pattern in angry_patterns:
                        if re.search(pattern, text_lower):
                            score += 0.3
                
                emotion_scores[emotion_type.value] = min(1.0, score)
            
            # 주요 감정 결정
            if emotion_scores:
                primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                primary_emotion_type = EmotionType(primary_emotion[0])
                confidence = primary_emotion[1]
            else:
                primary_emotion_type = EmotionType.NEUTRAL
                confidence = 0.5
            
            return {
                "primary_emotion": primary_emotion_type.value,
                "emotion_scores": emotion_scores,
                "confidence": confidence,
                "intensity": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing emotion: {e}")
            return {
                "primary_emotion": "neutral",
                "emotion_scores": {},
                "confidence": 0.5,
                "intensity": 0.5
            }
    
    def get_tone_suggestions(self, emotion: str, intensity: float = 0.5) -> List[str]:
        """
        감정과 강도에 맞는 톤 제안
        
        Args:
            emotion: 감정 유형
            intensity: 감정 강도
            
        Returns:
            List[str]: 제안된 톤 표현들
        """
        try:
            emotion_type = self._parse_emotion_type(emotion)
            tone_config = self.tone_patterns.get(emotion_type, self.tone_patterns[EmotionType.NEUTRAL])
            
            suggestions = []
            suggestions.extend(tone_config["opening"])
            suggestions.extend(tone_config["closing"])
            
            return suggestions[:10]  # 최대 10개 제안
            
        except Exception as e:
            self.logger.error(f"Error getting tone suggestions: {e}")
            return []
