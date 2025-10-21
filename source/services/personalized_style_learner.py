# -*- coding: utf-8 -*-
"""
개인화된 대화 스타일 학습 시스템
사용자별 선호하는 대화 스타일 학습 및 적용
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class FormalityLevel(Enum):
    """형식성 레벨"""
    FORMAL = "formal"
    MEDIUM = "medium"
    CASUAL = "casual"


class DetailPreference(Enum):
    """상세도 선호도"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExplanationStyle(Enum):
    """설명 스타일"""
    STEP_BY_STEP = "step_by_step"
    COMPREHENSIVE = "comprehensive"
    BRIEF = "brief"


class TonePreference(Enum):
    """톤 선호도"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"


@dataclass
class StylePreferences:
    """스타일 선호도 데이터 클래스"""
    formality_level: FormalityLevel = FormalityLevel.MEDIUM
    detail_preference: DetailPreference = DetailPreference.MEDIUM
    explanation_style: ExplanationStyle = ExplanationStyle.STEP_BY_STEP
    tone_preference: TonePreference = TonePreference.FRIENDLY
    response_length: str = "medium"  # long, medium, short
    preferred_keywords: List[str] = None
    avoided_keywords: List[str] = None
    interaction_frequency: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if self.preferred_keywords is None:
            self.preferred_keywords = []
        if self.avoided_keywords is None:
            self.avoided_keywords = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class InteractionPattern:
    """상호작용 패턴 데이터 클래스"""
    question_length: int
    response_satisfaction: float
    preferred_response_length: int
    interaction_time: float
    question_complexity: str
    topic_category: str
    timestamp: str


class PersonalizedStyleLearner:
    """사용자별 선호하는 대화 스타일 학습"""
    
    def __init__(self, user_profile_manager=None):
        """개인화된 스타일 학습기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.user_profile_manager = user_profile_manager
        
        # 스타일 학습 설정
        self.learning_config = {
            "min_interactions_for_learning": 5,
            "learning_decay_factor": 0.9,
            "preference_update_weight": 0.1,
            "pattern_analysis_window": 30  # days
        }
        
        # 기본 스타일 선호도
        self.default_preferences = StylePreferences()
        
        # 스타일 변환 규칙
        self.style_conversion_rules = {
            "formality_level": {
                "formal": {
                    "opening": "~에 대해 말씀드리면",
                    "closing": "~하시기 바랍니다",
                    "connectors": ["또한", "그리고", "또한"]
                },
                "medium": {
                    "opening": "~에 대해 설명드릴게요",
                    "closing": "~하시면 됩니다",
                    "connectors": ["그리고", "또한", "추가로"]
                },
                "casual": {
                    "opening": "~에 대해서는",
                    "closing": "~하시면 돼요",
                    "connectors": ["그리고", "그럼", "그래서"]
                }
            },
            "detail_preference": {
                "high": {
                    "modifier": "자세히",
                    "examples": True,
                    "step_by_step": True
                },
                "medium": {
                    "modifier": "",
                    "examples": True,
                    "step_by_step": False
                },
                "low": {
                    "modifier": "간단히",
                    "examples": False,
                    "step_by_step": False
                }
            }
        }
        
        # 사용자별 스타일 캐시
        self.style_cache = {}
        
        self.logger.info("PersonalizedStyleLearner initialized")
    
    def learn_user_preferences(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 상호작용 데이터로부터 선호도 학습
        
        Args:
            user_id: 사용자 ID
            interaction_data: 상호작용 데이터
            
        Returns:
            Dict[str, Any]: 업데이트된 선호도
        """
        try:
            # 1. 기존 프로필 가져오기
            profile = self._get_user_profile(user_id)
            current_preferences = profile.get("style_preferences", asdict(self.default_preferences))
            
            # 2. 상호작용 패턴 분석
            interaction_pattern = self._analyze_interaction_patterns(interaction_data)
            
            # 3. 선호도 업데이트
            updated_preferences = self._update_preferences(current_preferences, interaction_pattern)
            
            # 4. 프로필 업데이트
            self._update_user_profile(user_id, updated_preferences)
            
            # 5. 캐시 업데이트
            self.style_cache[user_id] = updated_preferences
            
            return updated_preferences
            
        except Exception as e:
            self.logger.error(f"Error learning user preferences: {e}")
            return asdict(self.default_preferences)
    
    def apply_personalized_style(self, answer: str, user_id: str) -> str:
        """
        개인화된 스타일 적용
        
        Args:
            answer: 원본 답변
            user_id: 사용자 ID
            
        Returns:
            str: 개인화된 스타일이 적용된 답변
        """
        try:
            if not answer or not isinstance(answer, str):
                return answer
            
            # 사용자 스타일 선호도 가져오기
            preferences = self._get_user_style_preferences(user_id)
            
            # 형식성 레벨 적용
            answer = self._apply_formality_level(answer, preferences["formality_level"])
            
            # 상세도 선호도 적용
            answer = self._apply_detail_preference(answer, preferences["detail_preference"])
            
            # 설명 스타일 적용
            answer = self._apply_explanation_style(answer, preferences["explanation_style"])
            
            # 톤 선호도 적용
            answer = self._apply_tone_preference(answer, preferences["tone_preference"])
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying personalized style: {e}")
            return answer
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """사용자 프로필 가져오기"""
        try:
            if self.user_profile_manager:
                return self.user_profile_manager.get_profile(user_id) or {}
            return {}
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return {}
    
    def _update_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """사용자 프로필 업데이트"""
        try:
            if self.user_profile_manager:
                self.user_profile_manager.update_preferences(user_id, {
                    "style_preferences": preferences,
                    "last_updated": datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
    
    def _analyze_interaction_patterns(self, interaction_data: Dict[str, Any]) -> InteractionPattern:
        """상호작용 패턴 분석"""
        try:
            question = interaction_data.get("question", "")
            response_length = len(interaction_data.get("answer", ""))
            satisfaction = interaction_data.get("satisfaction_score", 0.5)
            interaction_time = interaction_data.get("interaction_time", 0.0)
            
            # 질문 복잡도 분석
            question_complexity = self._analyze_question_complexity(question)
            
            # 토픽 카테고리 분석
            topic_category = self._analyze_topic_category(question)
            
            return InteractionPattern(
                question_length=len(question),
                response_satisfaction=satisfaction,
                preferred_response_length=response_length,
                interaction_time=interaction_time,
                question_complexity=question_complexity,
                topic_category=topic_category,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing interaction patterns: {e}")
            return InteractionPattern(
                question_length=0,
                response_satisfaction=0.5,
                preferred_response_length=0,
                interaction_time=0.0,
                question_complexity="simple",
                topic_category="general",
                timestamp=datetime.now().isoformat()
            )
    
    def _analyze_question_complexity(self, question: str) -> str:
        """질문 복잡도 분석"""
        try:
            if not question:
                return "simple"
            
            # 복잡도 지표
            complexity_indicators = {
                "simple": ["뭐", "어떻게", "언제", "어디"],
                "medium": ["왜", "어떤", "어느", "몇"],
                "complex": ["만약", "경우", "상황", "조건", "절차", "과정"]
            }
            
            question_lower = question.lower()
            
            for complexity, indicators in complexity_indicators.items():
                for indicator in indicators:
                    if indicator in question_lower:
                        return complexity
            
            # 질문 길이 기반 복잡도
            if len(question) > 100:
                return "complex"
            elif len(question) > 50:
                return "medium"
            else:
                return "simple"
                
        except Exception as e:
            self.logger.error(f"Error analyzing question complexity: {e}")
            return "simple"
    
    def _analyze_topic_category(self, question: str) -> str:
        """토픽 카테고리 분석"""
        try:
            if not question:
                return "general"
            
            topic_keywords = {
                "contract": ["계약", "계약서", "약정", "합의"],
                "family": ["가족", "이혼", "상속", "양육"],
                "criminal": ["형사", "범죄", "처벌", "벌금"],
                "civil": ["민사", "손해", "배상", "소송"],
                "labor": ["근로", "임금", "해고", "근로자"],
                "property": ["부동산", "매매", "임대", "등기"]
            }
            
            question_lower = question.lower()
            
            for category, keywords in topic_keywords.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        return category
            
            return "general"
            
        except Exception as e:
            self.logger.error(f"Error analyzing topic category: {e}")
            return "general"
    
    def _update_preferences(self, current_preferences: Dict[str, Any], pattern: InteractionPattern) -> Dict[str, Any]:
        """선호도 업데이트"""
        try:
            # 형식성 레벨 업데이트
            formality_level = self._update_formality_level(current_preferences["formality_level"], pattern)
            
            # 상세도 선호도 업데이트
            detail_preference = self._update_detail_preference(current_preferences["detail_preference"], pattern)
            
            # 설명 스타일 업데이트
            explanation_style = self._update_explanation_style(current_preferences["explanation_style"], pattern)
            
            # 톤 선호도 업데이트
            tone_preference = self._update_tone_preference(current_preferences["tone_preference"], pattern)
            
            # 응답 길이 선호도 업데이트
            response_length = self._update_response_length_preference(current_preferences["response_length"], pattern)
            
            return {
                "formality_level": formality_level,
                "detail_preference": detail_preference,
                "explanation_style": explanation_style,
                "tone_preference": tone_preference,
                "response_length": response_length,
                "preferred_keywords": current_preferences.get("preferred_keywords", []),
                "avoided_keywords": current_preferences.get("avoided_keywords", []),
                "interaction_frequency": current_preferences.get("interaction_frequency", 0) + 1,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating preferences: {e}")
            return current_preferences
    
    def _update_formality_level(self, current_level: str, pattern: InteractionPattern) -> str:
        """형식성 레벨 업데이트"""
        try:
            # 질문 길이와 복잡도에 따른 형식성 결정
            if pattern.question_length > 100 or pattern.question_complexity == "complex":
                return "formal"
            elif pattern.question_length < 30 or pattern.question_complexity == "simple":
                return "casual"
            else:
                return "medium"
        except Exception:
            return current_level
    
    def _update_detail_preference(self, current_preference: str, pattern: InteractionPattern) -> str:
        """상세도 선호도 업데이트"""
        try:
            # 만족도와 질문 복잡도에 따른 상세도 결정
            if pattern.response_satisfaction > 0.8 and pattern.question_complexity == "complex":
                return "high"
            elif pattern.response_satisfaction < 0.3 or pattern.question_complexity == "simple":
                return "low"
            else:
                return "medium"
        except Exception:
            return current_preference
    
    def _update_explanation_style(self, current_style: str, pattern: InteractionPattern) -> str:
        """설명 스타일 업데이트"""
        try:
            # 질문 복잡도에 따른 설명 스타일 결정
            if pattern.question_complexity == "complex":
                return "step_by_step"
            elif pattern.question_complexity == "simple":
                return "brief"
            else:
                return "comprehensive"
        except Exception:
            return current_style
    
    def _update_tone_preference(self, current_tone: str, pattern: InteractionPattern) -> str:
        """톤 선호도 업데이트"""
        try:
            # 만족도에 따른 톤 선호도 결정
            if pattern.response_satisfaction > 0.8:
                return "friendly"
            elif pattern.response_satisfaction < 0.3:
                return "professional"
            else:
                return current_tone
        except Exception:
            return current_tone
    
    def _update_response_length_preference(self, current_length: str, pattern: InteractionPattern) -> str:
        """응답 길이 선호도 업데이트"""
        try:
            # 선호하는 응답 길이 분석
            if pattern.preferred_response_length > 500:
                return "long"
            elif pattern.preferred_response_length < 200:
                return "short"
            else:
                return "medium"
        except Exception:
            return current_length
    
    def _get_user_style_preferences(self, user_id: str) -> Dict[str, Any]:
        """사용자 스타일 선호도 가져오기"""
        try:
            # 캐시에서 확인
            if user_id in self.style_cache:
                return self.style_cache[user_id]
            
            # 프로필에서 가져오기
            profile = self._get_user_profile(user_id)
            preferences = profile.get("style_preferences", asdict(self.default_preferences))
            
            # 캐시에 저장
            self.style_cache[user_id] = preferences
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error getting user style preferences: {e}")
            return asdict(self.default_preferences)
    
    def _apply_formality_level(self, answer: str, formality_level: str) -> str:
        """형식성 레벨 적용"""
        try:
            rules = self.style_conversion_rules["formality_level"].get(formality_level, 
                                                                     self.style_conversion_rules["formality_level"]["medium"])
            
            # 연결어 변경
            for old_connector, new_connector in [("그리고", rules["connectors"][0]), ("또한", rules["connectors"][1])]:
                answer = answer.replace(old_connector, new_connector)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying formality level: {e}")
            return answer
    
    def _apply_detail_preference(self, answer: str, detail_preference: str) -> str:
        """상세도 선호도 적용"""
        try:
            rules = self.style_conversion_rules["detail_preference"].get(detail_preference,
                                                                       self.style_conversion_rules["detail_preference"]["medium"])
            
            if detail_preference == "low":
                # 간단하게 만들기
                sentences = answer.split(".")
                if len(sentences) > 3:
                    answer = ". ".join(sentences[:3]) + "."
            
            elif detail_preference == "high":
                # 자세히 만들기
                if "예를 들어" not in answer and rules["examples"]:
                    answer += " 예를 들어, 실제 사례를 통해 설명드리면 더 명확할 것 같아요."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying detail preference: {e}")
            return answer
    
    def _apply_explanation_style(self, answer: str, explanation_style: str) -> str:
        """설명 스타일 적용"""
        try:
            if explanation_style == "step_by_step":
                # 단계별 설명 추가
                if "단계" not in answer and "절차" not in answer:
                    answer = f"단계별로 설명드리면, {answer}"
            
            elif explanation_style == "brief":
                # 간결하게 만들기
                if len(answer) > 300:
                    sentences = answer.split(".")
                    answer = ". ".join(sentences[:2]) + "."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying explanation style: {e}")
            return answer
    
    def _apply_tone_preference(self, answer: str, tone_preference: str) -> str:
        """톤 선호도 적용"""
        try:
            if tone_preference == "casual":
                # 캐주얼한 표현으로 변경
                replacements = {
                    "~입니다": "~예요",
                    "~합니다": "~해요",
                    "~하시기 바랍니다": "~하시면 돼요"
                }
                
                for formal, casual in replacements.items():
                    answer = answer.replace(formal, casual)
            
            elif tone_preference == "professional":
                # 전문적인 표현으로 변경
                replacements = {
                    "~예요": "~입니다",
                    "~해요": "~합니다",
                    "~하시면 돼요": "~하시기 바랍니다"
                }
                
                for casual, formal in replacements.items():
                    answer = answer.replace(casual, formal)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error applying tone preference: {e}")
            return answer
    
    def get_style_recommendations(self, user_id: str) -> Dict[str, Any]:
        """사용자별 스타일 추천"""
        try:
            preferences = self._get_user_style_preferences(user_id)
            
            recommendations = {
                "current_style": preferences,
                "suggestions": [],
                "improvement_areas": []
            }
            
            # 상호작용 빈도가 낮으면 기본 스타일 유지 권장
            if preferences.get("interaction_frequency", 0) < self.learning_config["min_interactions_for_learning"]:
                recommendations["suggestions"].append("더 많은 상호작용을 통해 개인화된 스타일을 학습할 수 있어요")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting style recommendations: {e}")
            return {"current_style": asdict(self.default_preferences), "suggestions": [], "improvement_areas": []}
