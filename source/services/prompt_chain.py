# -*- coding: utf-8 -*-
"""
프롬프트 체인 처리기
다단계 프롬프트 체인으로 자연스러운 답변 생성
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """체인 단계 데이터 클래스"""
    step_name: str
    prompt: str
    result: Optional[str] = None
    success: bool = False


class PromptChainProcessor:
    """프롬프트 체인 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_with_chain(self, query: str, context: str) -> str:
        """다단계 프롬프트 체인으로 자연스러운 답변 생성"""
        try:
            # 1단계: 질문 분석
            analysis_result = self._analyze_question(query)
            
            # 2단계: 답변 생성
            answer_result = self._generate_answer(query, context, analysis_result)
            
            # 3단계: 품질 검증 및 수정
            final_result = self._validate_and_fix(answer_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Chain processing failed: {e}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
    
    def _analyze_question(self, query: str) -> Dict[str, Any]:
        """1단계: 질문 분석"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            analysis_prompt = f"""다음 질문을 분석해주세요: {query}
            
            분석 결과를 JSON 형태로 제공하세요:
            {{
                "question_type": "법률조문|절차|용어|기타",
                "key_concepts": ["핵심개념1", "핵심개념2"],
                "expected_answer_length": "short|medium|long",
                "complexity": "simple|moderate|complex"
            }}"""
            
            response = gemini_client.generate(analysis_prompt)
            
            # 간단한 파싱 (실제로는 JSON 파싱이 필요하지만 여기서는 단순화)
            return {
                "question_type": "법률조문" if "조" in query else "절차" if "절차" in query else "기타",
                "key_concepts": query.split(),
                "expected_answer_length": "medium",
                "complexity": "moderate"
            }
            
        except Exception as e:
            self.logger.error(f"Question analysis failed: {e}")
            return {
                "question_type": "기타",
                "key_concepts": [],
                "expected_answer_length": "medium",
                "complexity": "moderate"
            }
    
    def _generate_answer(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """2단계: 답변 생성"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            answer_prompt = f"""분석 결과를 바탕으로 질문에 답변하세요:
            
            질문: {query}
            컨텍스트: {context}
            분석 결과: {analysis}
            
            답변 방식:
            - 친근한 변호사와 대화하는 것처럼 자연스럽게
            - 섹션 제목이나 플레이스홀더 사용 금지
            - 구체적이고 실용적인 정보 제공
            - 마치 실제 상담실에서 대화하는 것처럼 자연스럽게
            
            답변:"""
            
            response = gemini_client.generate(answer_prompt)
            return response.response
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return f"'{query}'에 대한 답변을 준비 중입니다."
    
    def _validate_and_fix(self, answer: str) -> str:
        """3단계: 품질 검증 및 수정"""
        try:
            # 템플릿 패턴 감지
            template_patterns = [
                "### 관련 법령",
                "*정확한 조문 번호와 내용*",
                "## 법률 문의 답변",
                "### 법령 해설",
                "### 적용 사례",
                "### 주의사항"
            ]
            
            for pattern in template_patterns:
                if pattern in answer:
                    self.logger.warning(f"Template pattern detected: {pattern}")
                    return self._fix_template_response(answer)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return answer
    
    def _fix_template_response(self, template_response: str) -> str:
        """템플릿 응답 수정"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            fix_prompt = f"""다음 템플릿 기반 답변을 자연스러운 대화체로 변환해주세요:

원본: {template_response}

변환 규칙:
1. 모든 섹션 제목과 플레이스홀더를 제거하세요
2. 하나의 연속된 자연스러운 답변으로 작성하세요
3. 친근하고 대화체로 변환하세요
4. 불필요한 면책 조항은 제거하세요
5. 마치 친한 변호사와 대화하는 것처럼 자연스럽게 작성하세요

변환된 답변:"""
            
            response = gemini_client.generate(fix_prompt)
            return response.response
            
        except Exception as e:
            self.logger.error(f"Template fixing failed: {e}")
            return template_response


# 싱글톤 인스턴스
prompt_chain_processor = PromptChainProcessor()
