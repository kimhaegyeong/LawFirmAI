#!/usr/bin/env python3
"""
사용자 설정 관리자
면책 조항 표시 방식, 스타일 등을 관리합니다.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DisclaimerStyle(Enum):
    """면책 조항 스타일"""
    NATURAL = "natural"      # 자연스러운 통합
    FORMAL = "formal"        # 공식적인 면책 조항
    NONE = "none"            # 면책 조항 없음

class DisclaimerPosition(Enum):
    """면책 조항 위치"""
    END = "end"              # 답변 끝에 추가
    INTEGRATED = "integrated" # 답변에 자연스럽게 통합
    NONE = "none"            # 표시하지 않음

@dataclass
class UserPreferences:
    """사용자 설정 데이터 클래스"""
    show_disclaimer: bool = True
    disclaimer_style: DisclaimerStyle = DisclaimerStyle.NATURAL
    disclaimer_position: DisclaimerPosition = DisclaimerPosition.END
    preferred_tone: str = "friendly"  # friendly, professional, casual
    example_preference: bool = True    # 예시 포함 여부

class UserPreferenceManager:
    """사용자 설정 관리자"""
    
    def __init__(self, config_file: str = "data/user_preferences.json"):
        self.config_file = config_file
        self.default_preferences = UserPreferences()
        self.preferences = self._load_preferences()
        
        # 자연스러운 면책 조항 템플릿
        self.natural_disclaimers = [
            "이 정보는 일반적인 안내이며, 구체적인 사안은 전문가와 상담하시기 바랍니다.",
            "위 내용은 참고용이며, 실제 법적 문제는 변호사와 상담하시는 것이 좋습니다.",
            "개별 상황에 따라 다를 수 있으니, 필요시 법률 전문가의 조언을 구하시기 바랍니다.",
            "이 설명은 일반적인 경우를 기준으로 하며, 특수한 상황은 전문가와 상담하세요.",
            "실제 사안에 적용하실 때는 변호사와 상담하여 정확한 조언을 받으시기 바랍니다."
        ]
        
        # 공식적인 면책 조항
        self.formal_disclaimers = [
            "본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.",
            "구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.",
            "이 정보는 참고용이며, 실제 법적 문제 해결을 위해서는 전문가의 조언이 필요합니다."
        ]
    
    def _load_preferences(self) -> UserPreferences:
        """사용자 설정 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return UserPreferences(
                        show_disclaimer=data.get('show_disclaimer', True),
                        disclaimer_style=DisclaimerStyle(data.get('disclaimer_style', 'natural')),
                        disclaimer_position=DisclaimerPosition(data.get('disclaimer_position', 'end')),
                        preferred_tone=data.get('preferred_tone', 'friendly'),
                        example_preference=data.get('example_preference', True)
                    )
        except Exception as e:
            print(f"설정 로드 실패: {e}")
        
        return self.default_preferences
    
    def save_preferences(self) -> bool:
        """사용자 설정 저장"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'show_disclaimer': self.preferences.show_disclaimer,
                    'disclaimer_style': self.preferences.disclaimer_style.value,
                    'disclaimer_position': self.preferences.disclaimer_position.value,
                    'preferred_tone': self.preferences.preferred_tone,
                    'example_preference': self.preferences.example_preference
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"설정 저장 실패: {e}")
            return False
    
    def get_disclaimer_text(self, style: Optional[DisclaimerStyle] = None) -> str:
        """면책 조항 텍스트 반환"""
        if not self.preferences.show_disclaimer:
            return ""
        
        style = style or self.preferences.disclaimer_style
        
        if style == DisclaimerStyle.NONE:
            return ""
        elif style == DisclaimerStyle.NATURAL:
            import random
            return random.choice(self.natural_disclaimers)
        elif style == DisclaimerStyle.FORMAL:
            import random
            return random.choice(self.formal_disclaimers)
        
        return ""
    
    def add_disclaimer_to_response(self, response: str, question: str = "") -> str:
        """답변에 면책 조항 추가"""
        if not self.preferences.show_disclaimer:
            return response
        
        disclaimer_text = self.get_disclaimer_text()
        if not disclaimer_text:
            return response
        
        if self.preferences.disclaimer_position == DisclaimerPosition.NONE:
            return response
        elif self.preferences.disclaimer_position == DisclaimerPosition.END:
            return f"{response}\n\n{disclaimer_text}"
        elif self.preferences.disclaimer_position == DisclaimerPosition.INTEGRATED:
            return self._integrate_disclaimer_naturally(response, disclaimer_text)
        
        return response
    
    def _integrate_disclaimer_naturally(self, response: str, disclaimer: str) -> str:
        """면책 조항을 답변에 자연스럽게 통합"""
        # 답변 끝에 자연스럽게 연결
        if response.endswith(('.', '!', '?')):
            return f"{response} {disclaimer}"
        else:
            return f"{response}\n\n{disclaimer}"
    
    def update_preference(self, key: str, value: Any) -> bool:
        """설정 업데이트"""
        try:
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
                return self.save_preferences()
            return False
        except Exception as e:
            print(f"설정 업데이트 실패: {e}")
            return False
    
    def get_preference(self, key: str) -> Any:
        """설정 값 반환"""
        return getattr(self.preferences, key, None)
    
    def reset_to_default(self) -> bool:
        """기본 설정으로 초기화"""
        self.preferences = self.default_preferences
        return self.save_preferences()

# 전역 인스턴스
preference_manager = UserPreferenceManager()
