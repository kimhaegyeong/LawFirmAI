# -*- coding: utf-8 -*-
"""
Gemini 2.5 Flash Lite 클라이언트
Google Gemini API를 사용한 텍스트 생성 클라이언트
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Gemini API 응답 데이터 클래스"""
    response: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class GeminiClient:
    """Gemini 2.5 Flash Lite 클라이언트"""
    
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash-lite",
                 api_key: Optional[str] = None,
                 timeout: int = 120,
                 request_interval: float = 1.0):
        """
        Gemini 클라이언트 초기화
        
        Args:
            model_name: 사용할 Gemini 모델명
            api_key: Google API 키 (None이면 환경변수에서 로드)
            timeout: 요청 타임아웃 (초)
            request_interval: 요청 간 인터벌 (초)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.request_interval = request_interval
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        
        # API 키 설정
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. 환경변수나 직접 전달해주세요.")
        
        # Gemini API 설정
        genai.configure(api_key=self.api_key)
        
        # 모델 초기화
        try:
            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"Gemini client initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def generate(self, 
                 prompt: str, 
                 context: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> GeminiResponse:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            context: 컨텍스트 정보
            system_prompt: 시스템 프롬프트
            temperature: 생성 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            
        Returns:
            GeminiResponse: 생성된 응답
        """
        try:
            # 요청 간 인터벌 처리
            self._wait_for_interval()
            
            self.logger.info(f"Generating response with model {self.model_name}")
            
            # 기본 시스템 프롬프트 설정 (구체적 답변 요구)
            default_system_prompt = """당신은 한국 법률 전문가입니다. 사용자의 질문에 대해 구체적이고 완전한 답변을 제공해야 합니다.

중요한 지침:
1. 질문에 대한 직접적이고 구체적인 답변을 제공하세요
2. 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
3. 답변이 불완전한 경우에만 추가 정보 요청을 하세요
4. 법률적 정확성을 유지하면서 이해하기 쉽게 설명하세요
5. 관련 법령과 판례를 적절히 인용하세요"""
            
            # 시스템 프롬프트 구성
            final_system_prompt = system_prompt or default_system_prompt
            
            # 프롬프트 구성
            full_prompt = self._build_prompt(prompt, context, final_system_prompt)
            
            # 생성 설정
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or 0.7,
                max_output_tokens=max_tokens or 2048,
                top_p=0.8,
                top_k=40
            )
            
            # API 호출
            start_time = time.time()
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            end_time = time.time()
            
            if response.text:
                duration_ms = int((end_time - start_time) * 1000)
                
                gemini_response = GeminiResponse(
                    response=response.text,
                    model=self.model_name,
                    created_at=datetime.now().isoformat(),
                    done=True,
                    total_duration=duration_ms,
                    prompt_eval_count=len(full_prompt.split()),
                    prompt_eval_duration=duration_ms // 2,
                    eval_count=len(response.text.split()),
                    eval_duration=duration_ms // 2
                )
                
                self.logger.info(f"Generated response: {len(gemini_response.response)} characters")
                return gemini_response
            else:
                raise Exception("No response generated from Gemini")
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _build_prompt(self, 
                     prompt: str, 
                     context: Optional[str] = None,
                     system_prompt: Optional[str] = None) -> str:
        """
        프롬프트 구성
        
        Args:
            prompt: 기본 프롬프트
            context: 컨텍스트 정보
            system_prompt: 시스템 프롬프트
            
        Returns:
            str: 구성된 프롬프트
        """
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if context:
            parts.append(f"Context: {context}")
        
        parts.append(f"Question: {prompt}")
        
        return "\n\n".join(parts)
    
    def _wait_for_interval(self):
        """요청 간 인터벌 대기"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            self.logger.debug(f"Waiting {sleep_time:.2f} seconds for rate limit")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def is_available(self) -> bool:
        """Gemini API 사용 가능 여부 확인"""
        try:
            test_response = self.model.generate_content("Hello")
            return test_response.text is not None
        except Exception as e:
            self.logger.error(f"Gemini API not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "timeout": self.timeout,
            "available": self.is_available()
        }


# 기본 인스턴스 생성
def create_gemini_client() -> GeminiClient:
    """기본 Gemini 클라이언트 생성"""
    return GeminiClient()


if __name__ == "__main__":
    # 테스트 코드
    try:
        client = create_gemini_client()
        
        # 간단한 테스트
        response = client.generate("안녕하세요! 간단한 인사말을 해주세요.")
        print(f"Response: {response.response}")
        print(f"Model: {response.model}")
        print(f"Duration: {response.total_duration}ms")
        
    except Exception as e:
        print(f"Error: {e}")
