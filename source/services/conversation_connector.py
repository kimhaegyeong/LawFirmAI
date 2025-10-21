# -*- coding: utf-8 -*-
"""
대화 연결어 시스템
이전 대화와의 자연스러운 연결을 위한 연결어 추가 시스템
"""

import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationConnector:
    """대화 맥락을 고려한 자연스러운 연결어 추가"""
    
    def __init__(self):
        """대화 연결어 시스템 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 연결어 패턴 정의
        self.connector_patterns = {
            "follow_up": {
                "이어서": ["그럼", "그렇다면", "그러면", "그래서"],
                "추가질문": ["또한", "그리고", "추가로", "한 가지 더"],
                "이전관련": ["앞서 말씀하신", "이전에 언급한", "방금 말한", "앞서 질문하신"]
            },
            "emotion_based": {
                "긍정": ["좋은 질문이네요", "잘 물어보셨어요", "정확한 포인트입니다", "핵심을 짚으셨네요"],
                "부정": ["이해가 안 되시는군요", "궁금하시겠어요", "혼란스러우실 것 같아요", "어려우실 것 같아요"],
                "긴급": ["급하신 것 같네요", "바로 답변드릴게요", "시급한 문제군요", "빨리 해결해드릴게요"],
                "걱정": ["걱정하지 마세요", "차근차근 설명드릴게요", "이해하기 쉽게 말씀드릴게요", "단계별로 알아보죠"]
            },
            "context_based": {
                "법령": ["관련 법령을 보면", "법적으로는", "법령에 따르면", "법률상으로는"],
                "판례": ["유사한 판례가 있어요", "법원에서는", "판례를 보면", "실제 사례를 보면"],
                "실무": ["실무적으로는", "현실적으로는", "실제로는", "일반적으로는"],
                "계약": ["계약서에서는", "계약상으로는", "계약 조건을 보면", "계약 내용에 따르면"]
            },
            "transition": {
                "새로운주제": ["이제 다른 관점에서", "다른 측면을 보면", "추가로 고려할 점은", "또 다른 중요한 점은"],
                "요약": ["정리하면", "요약하면", "핵심은", "중요한 포인트는"],
                "조언": ["권장드리는 방법은", "추천드리는 것은", "이렇게 하시면", "이런 방법이 있어요"]
            }
        }
        
        # 연결어 사용 빈도 추적
        self.usage_stats = {}
        
        self.logger.info("ConversationConnector initialized")
    
    def add_natural_connectors(self, answer: str, context: Dict[str, Any]) -> str:
        """
        자연스러운 연결어와 표현 추가
        
        Args:
            answer: 원본 답변
            context: 대화 맥락 정보
            
        Returns:
            str: 연결어가 추가된 답변
        """
        try:
            if not answer or not isinstance(answer, str):
                return answer
            
            # 1. 이전 대화 분석
            previous_topic = context.get("previous_topic", "")
            conversation_flow = context.get("conversation_flow", "new")
            question_type = context.get("question_type", "general")
            user_emotion = context.get("user_emotion", "neutral")
            
            # 2. 적절한 연결어 선택
            connector = self._select_connector(
                previous_topic, conversation_flow, user_emotion, question_type
            )
            
            # 3. 답변에 연결어 추가
            if connector:
                answer = f"{connector} {answer}"
                self._track_usage(connector)
            
            # 4. 자연스러운 마무리 추가
            answer = self._add_natural_ending(answer, context)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error adding connectors: {e}")
            return answer
    
    def _select_connector(self, previous_topic: str, flow: str, emotion: str, question_type: str) -> str:
        """
        상황에 맞는 연결어 선택
        
        Args:
            previous_topic: 이전 주제
            flow: 대화 흐름
            emotion: 사용자 감정
            question_type: 질문 유형
            
        Returns:
            str: 선택된 연결어
        """
        try:
            # 1. 대화 흐름 기반 선택
            if flow == "follow_up" and previous_topic:
                return random.choice(self.connector_patterns["follow_up"]["이전관련"])
            
            # 2. 감정 기반 선택
            if emotion in ["urgent", "anxious"]:
                return random.choice(self.connector_patterns["emotion_based"]["긴급"])
            elif emotion in ["confused", "negative"]:
                return random.choice(self.connector_patterns["emotion_based"]["부정"])
            elif emotion in ["positive", "satisfied"]:
                return random.choice(self.connector_patterns["emotion_based"]["긍정"])
            
            # 3. 질문 유형 기반 선택
            if question_type in ["law_article", "legal_provision"]:
                return random.choice(self.connector_patterns["context_based"]["법령"])
            elif question_type in ["precedent", "case_law"]:
                return random.choice(self.connector_patterns["context_based"]["판례"])
            elif question_type in ["contract", "agreement"]:
                return random.choice(self.connector_patterns["context_based"]["계약"])
            elif question_type in ["practical", "procedure"]:
                return random.choice(self.connector_patterns["context_based"]["실무"])
            
            # 4. 기본값 (연결어 없음)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error selecting connector: {e}")
            return ""
    
    def _add_natural_ending(self, answer: str, context: Dict[str, Any]) -> str:
        """
        자연스러운 마무리 추가
        
        Args:
            answer: 답변
            context: 맥락 정보
            
        Returns:
            str: 마무리가 추가된 답변
        """
        try:
            # 이미 마무리가 있는지 확인
            if any(ending in answer for ending in ["있으시면", "언제든지", "바라요", "도움이"]):
                return answer
            
            # 질문 유형에 따른 마무리
            question_type = context.get("question_type", "general")
            user_emotion = context.get("user_emotion", "neutral")
            
            endings = {
                "general": ["추가 질문 있으시면 언제든지 물어보세요", "더 궁금한 점 있으시면 말씀해주세요"],
                "urgent": ["빨리 해결하시길 바라요", "서둘러 처리하시면 좋겠어요"],
                "confused": ["이해가 되셨나요?", "더 자세히 설명드릴까요?"],
                "contract": ["계약서 검토 시 주의하세요", "계약 조건을 꼼꼼히 확인하세요"]
            }
            
            # 적절한 마무리 선택
            if user_emotion == "urgent":
                ending = random.choice(endings["urgent"])
            elif user_emotion == "confused":
                ending = random.choice(endings["confused"])
            elif question_type in ["contract", "agreement"]:
                ending = random.choice(endings["contract"])
            else:
                ending = random.choice(endings["general"])
            
            # 마무리 추가
            if not answer.endswith(("요", "니다", "다", "어요", "예요")):
                answer += "."
            
            answer += f" {ending}."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error adding natural ending: {e}")
            return answer
    
    def _track_usage(self, connector: str) -> None:
        """연결어 사용 통계 추적"""
        try:
            if connector not in self.usage_stats:
                self.usage_stats[connector] = 0
            self.usage_stats[connector] += 1
        except Exception as e:
            self.logger.error(f"Error tracking usage: {e}")
    
    def get_usage_stats(self) -> Dict[str, int]:
        """연결어 사용 통계 반환"""
        return self.usage_stats.copy()
    
    def reset_stats(self) -> None:
        """사용 통계 초기화"""
        self.usage_stats = {}
    
    def get_connector_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """
        맥락에 맞는 연결어 제안
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 제안된 연결어 목록
        """
        try:
            suggestions = []
            
            emotion = context.get("user_emotion", "neutral")
            question_type = context.get("question_type", "general")
            flow = context.get("conversation_flow", "new")
            
            # 감정 기반 제안
            if emotion in self.connector_patterns["emotion_based"]:
                suggestions.extend(self.connector_patterns["emotion_based"][emotion])
            
            # 질문 유형 기반 제안
            if question_type in self.connector_patterns["context_based"]:
                suggestions.extend(self.connector_patterns["context_based"][question_type])
            
            # 대화 흐름 기반 제안
            if flow in self.connector_patterns["follow_up"]:
                suggestions.extend(self.connector_patterns["follow_up"][flow])
            
            return suggestions[:5]  # 최대 5개 제안
            
        except Exception as e:
            self.logger.error(f"Error getting connector suggestions: {e}")
            return []
