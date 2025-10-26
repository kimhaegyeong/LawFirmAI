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
                 max_tokens: Optional[int] = None,
                 question_type: str = "일반") -> GeminiResponse:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            context: 컨텍스트 정보
            system_prompt: 시스템 프롬프트
            temperature: 생성 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            question_type: 질문 유형 (오류 처리용)
            
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
            error_msg = str(e)
            self.logger.error(f"Error generating response: {error_msg}")
            
            # 특정 오류 타입별 처리
            if "response.text quick accessor" in error_msg:
                self.logger.warning(f"Gemini API response accessor error: {error_msg}")
                # 대체 응답 생성
                return self._generate_fallback_response(prompt, question_type, e)
            elif "finish_reason" in error_msg and "2" in error_msg:
                self.logger.warning(f"Gemini API finish_reason error: {error_msg}")
                # 대체 응답 생성
                return self._generate_fallback_response(prompt, question_type, e)
            elif "timeout" in error_msg.lower():
                self.logger.warning(f"Gemini API timeout error: {error_msg}")
                # 대체 응답 생성
                return self._generate_fallback_response(prompt, question_type, e)
            else:
                # 기타 오류는 기존 방식대로 처리
                raise Exception(f"Gemini API error: {error_msg}")
    
    def _generate_fallback_response(self, prompt: str, question_type: str, error: Exception) -> GeminiResponse:
        """
        Gemini API 오류 시 대체 응답 생성
        
        Args:
            prompt: 원본 프롬프트
            question_type: 질문 유형
            error: 발생한 오류
            
        Returns:
            GeminiResponse: 대체 응답
        """
        try:
            self.logger.info(f"Generating fallback response for question_type: {question_type}")
            
            # 질문 유형별 대체 응답 템플릿
            fallback_responses = {
                "법률조문": "죄송합니다. 현재 해당 법률 조문에 대한 상세한 정보를 제공할 수 없습니다. 정확한 법률 조문 내용은 국가법령정보센터(www.law.go.kr)에서 확인하시거나, 법률 전문가와 상담하시기 바랍니다.",
                
                "계약서": "죄송합니다. 현재 계약서 작성에 대한 상세한 안내를 제공할 수 없습니다. 계약서 작성은 법적 효력이 있으므로, 구체적인 내용은 법률 전문가와 상담하시거나 공인중개사, 법무사 등 전문가의 도움을 받으시기 바랍니다.",
                
                "부동산": "죄송합니다. 현재 부동산 관련 절차에 대한 상세한 안내를 제공할 수 없습니다. 부동산 거래는 복잡한 법적 절차가 필요하므로, 공인중개사나 법무사 등 전문가와 상담하시기 바랍니다.",
                
                "가족법": "죄송합니다. 현재 가족법 관련 절차에 대한 상세한 안내를 제공할 수 없습니다. 가족법 관련 문제는 민감하고 복잡하므로, 가정법원이나 가족상담소, 법률 전문가와 상담하시기 바랍니다.",
                
                "민사법": "죄송합니다. 현재 민사법 관련 절차에 대한 상세한 안내를 제공할 수 없습니다. 민사법 관련 문제는 구체적인 상황에 따라 다르므로, 법률 전문가와 상담하시기 바랍니다.",
                
                "일반": "죄송합니다. 현재 질문에 대한 상세한 답변을 제공할 수 없습니다. 법률 관련 질문은 구체적인 상황에 따라 답변이 달라질 수 있으므로, 법률 전문가와 상담하시기 바랍니다."
            }
            
            # 질문 유형에 따른 대체 응답 선택
            fallback_response = fallback_responses.get(question_type, fallback_responses["일반"])
            
            # 오류 정보를 포함한 응답 생성
            full_response = f"{fallback_response}\n\n※ 시스템 오류로 인해 상세한 답변을 제공할 수 없습니다. 잠시 후 다시 시도해주세요."
            
            return GeminiResponse(
                response=full_response,
                model=f"{self.model_name}_fallback",
                created_at=datetime.now().isoformat(),
                done=True,
                total_duration=None,
                prompt_eval_count=None,
                prompt_eval_duration=None,
                eval_count=None,
                eval_duration=None
            )
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback response generation failed: {fallback_error}")
            
            # 최종 대체 응답
            final_response = "죄송합니다. 현재 시스템 오류로 인해 답변을 제공할 수 없습니다. 잠시 후 다시 시도해주시거나, 법률 전문가와 상담하시기 바랍니다."
            
            return GeminiResponse(
                response=final_response,
                model=f"{self.model_name}_final_fallback",
                created_at=datetime.now().isoformat(),
                done=True,
                total_duration=None,
                prompt_eval_count=None,
                prompt_eval_duration=None,
                eval_count=None,
                eval_duration=None
            )
    
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
