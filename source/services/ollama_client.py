# -*- coding: utf-8 -*-
"""
Ollama 클라이언트
Ollama Qwen2.5:7b 모델과 최적 통합을 위한 클라이언트
"""

import logging
import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Ollama 응답 데이터 클래스"""
    response: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """Ollama 클라이언트"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "qwen2.5:7b",
                 timeout: int = 120):
        """Ollama 클라이언트 초기화"""
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        
        # 기본 설정
        self.default_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": ["</s>", "<|im_end|>", "Human:", "사용자:"],
            "stream": False
        }
        
        # 모델 상태 확인
        self._check_model_availability()
    
    def _check_model_availability(self):
        """모델 가용성 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    self.logger.info(f"Model {self.model_name} is available")
                else:
                    self.logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    # 사용 가능한 첫 번째 모델로 대체
                    if model_names:
                        self.model_name = model_names[0]
                        self.logger.info(f"Using alternative model: {self.model_name}")
            else:
                self.logger.warning(f"Failed to check model availability: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
    
    def generate(self, 
                 prompt: str, 
                 context: str = "",
                 temperature: float = None,
                 max_tokens: int = 2048,
                 system_prompt: str = None) -> OllamaResponse:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            context: 컨텍스트 정보
            temperature: 생성 온도 (None이면 기본값 사용)
            max_tokens: 최대 토큰 수
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
            
        Returns:
            OllamaResponse: 생성된 응답
        """
        try:
            self.logger.info(f"Generating response with model {self.model_name}")
            
            # 프롬프트 구성
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # 요청 데이터 구성
            request_data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "options": {
                    "temperature": temperature if temperature is not None else self.default_config["temperature"],
                    "top_p": self.default_config["top_p"],
                    "top_k": self.default_config["top_k"],
                    "repeat_penalty": self.default_config["repeat_penalty"],
                    "stop": self.default_config["stop"],
                    "num_predict": max_tokens
                },
                "stream": self.default_config["stream"]
            }
            
            # API 호출
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                ollama_response = OllamaResponse(
                    response=result.get("response", ""),
                    model=result.get("model", self.model_name),
                    created_at=result.get("created_at", ""),
                    done=result.get("done", True),
                    total_duration=result.get("total_duration"),
                    load_duration=result.get("load_duration"),
                    prompt_eval_count=result.get("prompt_eval_count"),
                    prompt_eval_duration=result.get("prompt_eval_duration"),
                    eval_count=result.get("eval_count"),
                    eval_duration=result.get("eval_duration")
                )
                
                self.logger.info(f"Generated response: {len(ollama_response.response)} characters")
                return ollama_response
                
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = f"Ollama API timeout after {self.timeout} seconds"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def _build_prompt(self, 
                     prompt: str, 
                     context: str = "", 
                     system_prompt: str = None) -> str:
        """프롬프트 구성"""
        try:
            # 시스템 프롬프트 설정
            if system_prompt is None:
                system_prompt = self._get_default_system_prompt()
            
            # 프롬프트 구성
            if context:
                full_prompt = f"""{system_prompt}

컨텍스트:
{context}

사용자 질문:
{prompt}

위 정보를 바탕으로 전문적이고 정확한 답변을 작성하세요."""
            else:
                full_prompt = f"""{system_prompt}

사용자 질문:
{prompt}

전문적이고 정확한 답변을 작성하세요."""
            
            return full_prompt
            
        except Exception as e:
            self.logger.error(f"Error building prompt: {e}")
            return prompt
    
    def _get_default_system_prompt(self) -> str:
        """기본 시스템 프롬프트 반환"""
        return """당신은 법률 전문가입니다. 다음 원칙에 따라 답변하세요:

1. 정확성: 법률 정보는 정확하고 최신의 것을 제공하세요.
2. 명확성: 복잡한 법률 개념을 이해하기 쉽게 설명하세요.
3. 구조화: 답변을 논리적이고 체계적으로 구성하세요.
4. 신중함: 불확실한 정보는 명시하고 전문가 상담을 권장하세요.
5. 실용성: 실제 상황에 적용 가능한 조언을 제공하세요.

답변 형식:
- 핵심 답변을 먼저 제시
- 관련 법률 및 판례 인용
- 단계별 설명 (필요시)
- 주의사항 및 권장사항
- 전문가 상담 권장 (필요시)"""
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = None) -> OllamaResponse:
        """
        채팅 형태의 대화
        
        Args:
            messages: 대화 메시지 리스트 [{"role": "user", "content": "..."}, ...]
            temperature: 생성 온도
            
        Returns:
            OllamaResponse: 생성된 응답
        """
        try:
            self.logger.info(f"Chatting with model {self.model_name}")
            
            # 요청 데이터 구성
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "temperature": temperature if temperature is not None else self.default_config["temperature"],
                    "top_p": self.default_config["top_p"],
                    "top_k": self.default_config["top_k"],
                    "repeat_penalty": self.default_config["repeat_penalty"],
                    "stop": self.default_config["stop"]
                },
                "stream": self.default_config["stream"]
            }
            
            # API 호출
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                ollama_response = OllamaResponse(
                    response=result.get("message", {}).get("content", ""),
                    model=result.get("model", self.model_name),
                    created_at=result.get("created_at", ""),
                    done=result.get("done", True),
                    total_duration=result.get("total_duration"),
                    load_duration=result.get("load_duration"),
                    prompt_eval_count=result.get("prompt_eval_count"),
                    prompt_eval_duration=result.get("prompt_eval_duration"),
                    eval_count=result.get("eval_count"),
                    eval_duration=result.get("eval_duration")
                )
                
                self.logger.info(f"Chat response: {len(ollama_response.response)} characters")
                return ollama_response
                
            else:
                error_msg = f"Ollama chat API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model["name"] == self.model_name:
                        return model
                return {}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Ollama 서비스 가용성 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def pull_model(self, model_name: str = None) -> bool:
        """모델 다운로드"""
        try:
            target_model = model_name or self.model_name
            self.logger.info(f"Pulling model {target_model}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": target_model},
                timeout=300  # 모델 다운로드는 시간이 오래 걸릴 수 있음
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully pulled model {target_model}")
                return True
            else:
                self.logger.error(f"Failed to pull model: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            return False


# 테스트 함수
def test_ollama_client():
    """Ollama 클라이언트 테스트"""
    client = OllamaClient()
    
    if not client.is_available():
        print("Ollama service is not available. Please start Ollama service.")
        return
    
    print("=== Ollama 클라이언트 테스트 ===")
    
    # 모델 정보 확인
    model_info = client.get_model_info()
    print(f"Model info: {model_info}")
    
    # 간단한 생성 테스트
    try:
        response = client.generate(
            prompt="손해배상 청구 방법을 간단히 설명해주세요.",
            context="민법 제750조 불법행위 관련 질문입니다."
        )
        
        print(f"\n생성된 응답:")
        print(f"- 모델: {response.model}")
        print(f"- 응답 길이: {len(response.response)} 문자")
        print(f"- 총 소요 시간: {response.total_duration}ms")
        print(f"- 응답 내용: {response.response[:200]}...")
        
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # 채팅 테스트
    try:
        messages = [
            {"role": "user", "content": "계약 해지 절차에 대해 알려주세요."}
        ]
        
        chat_response = client.chat(messages)
        
        print(f"\n채팅 응답:")
        print(f"- 응답 길이: {len(chat_response.response)} 문자")
        print(f"- 응답 내용: {chat_response.response[:200]}...")
        
    except Exception as e:
        print(f"Chat test failed: {e}")


if __name__ == "__main__":
    test_ollama_client()
