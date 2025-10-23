# -*- coding: utf-8 -*-
"""
대안 모델 클라이언트
여러 모델을 시도하여 최적의 답변 생성
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """모델 응답 데이터 클래스"""
    response: str
    model_name: str
    success: bool
    error: Optional[str] = None


class AlternativeModelClient:
    """대안 모델 클라이언트"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = [
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
    
    def generate_with_fallback(self, prompt: str) -> str:
        """여러 모델을 시도하여 최적의 답변 생성"""
        for model in self.models:
            try:
                response = self._generate_with_model(model, prompt)
                if self._is_natural_response(response):
                    self.logger.info(f"Successfully generated natural response with {model}")
                    return response
                else:
                    self.logger.warning(f"Generated template response with {model}, trying next model")
            except Exception as e:
                self.logger.error(f"Failed to generate with {model}: {e}")
                continue
        
        return "죄송합니다. 답변을 생성할 수 없습니다."
    
    def _generate_with_model(self, model_name: str, prompt: str) -> str:
        """특정 모델로 답변 생성"""
        try:
            from .gemini_client import GeminiClient
            
            # 모델별 설정 조정
            if model_name == "gemini-2.5-flash-lite":
                client = GeminiClient(model_name=model_name)
                response = client.generate(
                    prompt=prompt,
                    temperature=0.9,
                    max_tokens=1024
                )
                return response.response
            elif model_name == "gemini-1.5-flash":
                client = GeminiClient(model_name=model_name)
                response = client.generate(
                    prompt=prompt,
                    temperature=0.8,
                    max_tokens=1024
                )
                return response.response
            else:  # gemini-1.5-pro
                client = GeminiClient(model_name=model_name)
                response = client.generate(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.response
                
        except Exception as e:
            self.logger.error(f"Error generating with {model_name}: {e}")
            raise
    
    def _is_natural_response(self, response: str) -> bool:
        """자연스러운 답변인지 확인"""
        template_indicators = [
            "### 관련 법령",
            "*정확한 조문 번호와 내용*",
            "## 법률 문의 답변",
            "### 법령 해설",
            "### 적용 사례",
            "### 주의사항",
            "*쉬운 말로 풀어서 설명*",
            "*구체적 예시와 설명*"
        ]
        
        for indicator in template_indicators:
            if indicator in response:
                return False
        
        # 자연스러운 답변의 특징 확인
        natural_indicators = [
            "이에요", "예요", "입니다", "해요", "하세요",
            "쉽게 말해서", "구체적으로", "예를 들어"
        ]
        
        natural_count = sum(1 for indicator in natural_indicators if indicator in response)
        
        # 자연스러운 지표가 2개 이상 있으면 자연스러운 답변으로 판단
        return natural_count >= 2


# 싱글톤 인스턴스
alternative_model_client = AlternativeModelClient()
