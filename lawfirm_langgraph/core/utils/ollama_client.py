#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama API 클라이언트

Ollama 서버와 통신하여 LLM 모델을 사용하는 클라이언트 클래스
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API 클라이언트"""
    
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        Ollama 클라이언트 초기화
        
        Args:
            model: 사용할 모델명
            base_url: Ollama 서버 URL
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5분 타임아웃
        
        # 서버 연결 확인
        self._check_server_connection()
    
    def _check_server_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info(f"Ollama 서버 연결 성공: {self.base_url}")
                logger.info(f"사용 모델: {self.model}")
            else:
                raise ConnectionError(f"Ollama 서버 응답 오류: {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다: {self.base_url}")
        except Exception as e:
            raise ConnectionError(f"Ollama 서버 연결 확인 중 오류: {e}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            system_prompt: 시스템 프롬프트
            temperature: 생성 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            stream: 스트리밍 모드 사용 여부
            
        Returns:
            생성된 텍스트
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if stream:
                return self._generate_stream(payload)
            else:
                return self._generate_single(payload)
                
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {e}")
            raise
    
    def _generate_single(self, payload: Dict) -> str:
        """단일 응답 생성"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama API 응답 시간 초과")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 요청 오류: {e}")
            raise
    
    def _generate_stream(self, payload: Dict) -> Generator[str, None, None]:
        """스트리밍 응답 생성"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.Timeout:
            logger.error("Ollama API 스트리밍 응답 시간 초과")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 스트리밍 요청 오류: {e}")
            raise
    
    def generate_qa_pairs(
        self, 
        context: str, 
        qa_count: int = 3,
        question_types: List[str] = None,
        temperature: float = 0.7
    ) -> List[Dict[str, str]]:
        """
        Q&A 쌍 생성
        
        Args:
            context: 법률 컨텍스트 (조문, 판례 등)
            qa_count: 생성할 Q&A 개수
            question_types: 질문 유형 리스트
            temperature: 생성 온도
            
        Returns:
            Q&A 쌍 리스트
        """
        if question_types is None:
            question_types = [
                "개념 설명", "실제 적용", "요건/효과", 
                "비교/차이", "절차", "예시", "주의사항"
            ]
        
        system_prompt = """당신은 법률 전문가입니다. 주어진 법률 내용을 바탕으로 실용적이고 정확한 질문-답변을 생성해주세요.

답변 규칙:
1. 법률 조문을 정확히 인용하고 해석
2. 실무에서 자주 묻는 질문 위주로 생성
3. 구체적이고 실용적인 답변 제공
4. 법률 용어는 정확하게 사용
5. JSON 형식으로만 응답

응답 형식:
[
  {
    "question": "질문 내용",
    "answer": "답변 내용",
    "type": "질문 유형"
  }
]"""
        
        prompt = f"""다음 법률 내용을 바탕으로 {qa_count}개의 질문-답변을 생성해주세요.

법률 내용:
{context}

질문 유형: {', '.join(question_types)}

각 질문은 다음 중 하나의 유형으로 생성해주세요:
- 개념 설명: "~란 무엇인가요?"
- 실제 적용: "~한 경우 어떻게 해야 하나요?"
- 요건/효과: "~의 요건은 무엇인가요?"
- 비교/차이: "~와 ~의 차이는 무엇인가요?"
- 절차: "~하려면 어떤 절차를 거쳐야 하나요?"
- 예시: "~의 구체적인 예시를 들어주세요"
- 주의사항: "~할 때 주의할 점은 무엇인가요?"

JSON 형식으로만 응답해주세요."""

        try:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=2000
            )
            
            # JSON 파싱 시도
            qa_pairs = self._parse_qa_response(response)
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Q&A 생성 중 오류: {e}")
            return []
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """Q&A 응답 파싱"""
        try:
            # JSON 부분만 추출
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("JSON 형식을 찾을 수 없습니다.")
                return []
            
            json_str = response[start_idx:end_idx]
            qa_pairs = json.loads(json_str)
            
            # 유효성 검증
            valid_pairs = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and all(key in qa for key in ['question', 'answer']):
                    valid_pairs.append({
                        'question': qa['question'].strip(),
                        'answer': qa['answer'].strip(),
                        'type': qa.get('type', 'unknown')
                    })
            
            return valid_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.error(f"응답 내용: {response[:200]}...")
            return []
        except Exception as e:
            logger.error(f"Q&A 파싱 중 오류: {e}")
            return []
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            test_prompt = "안녕하세요. 간단한 인사말을 해주세요."
            response = self.generate(test_prompt, temperature=0.1, max_tokens=50)
            
            if response and len(response.strip()) > 0:
                logger.info(f"연결 테스트 성공: {response[:50]}...")
                return True
            else:
                logger.error("연결 테스트 실패: 빈 응답")
                return False
                
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """모델 정보 조회"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": self.model}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"모델 정보 조회 오류: {e}")
            return {}


def main():
    """테스트 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 클라이언트 생성
        client = OllamaClient()
        
        # 연결 테스트
        if client.test_connection():
            print("✅ Ollama 클라이언트 연결 성공")
            
            # Q&A 생성 테스트
            test_context = """
            개인정보 보호법 제2조제1항
            "개인정보"란 살아 있는 개인에 관한 정보로서 성명, 주민등록번호 및 영상 등을 통하여 개인을 알아볼 수 있는 정보(해당 정보만으로는 특정 개인을 알아볼 수 없더라도 다른 정보와 쉽게 결합하여 알아볼 수 있는 정보를 포함한다)를 말한다.
            """
            
            qa_pairs = client.generate_qa_pairs(test_context, qa_count=2)
            
            print(f"\n생성된 Q&A 쌍: {len(qa_pairs)}개")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"\n{i}. 질문: {qa['question']}")
                print(f"   답변: {qa['answer']}")
                print(f"   유형: {qa['type']}")
        else:
            print("❌ Ollama 클라이언트 연결 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
