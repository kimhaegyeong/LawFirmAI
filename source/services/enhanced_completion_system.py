#!/usr/bin/env python3
"""
강화된 답변 완성 시스템
중간에 끊어진 답변을 강제로 완성하는 시스템
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CompletionResult:
    """완성 결과"""
    completed_answer: str
    was_truncated: bool
    completion_method: str
    confidence: float

class EnhancedCompletionSystem:
    """강화된 답변 완성 시스템"""
    
    def __init__(self):
        # 불완전한 답변 패턴
        self.truncation_patterns = [
            r'드$', r'그리고$', r'또한$', r'마지막으로$', r'결론적으로$',
            r'예를 들어$', r'구체적으로$', r'특히$', r'또한$',
            r'[가-힣]+드$', r'[가-힣]+고$', r'[가-힣]+며$',
            r'\d+\.\s*$', r'[가-힣]+:\s*$', r'[가-힣]+의\s*$'
        ]
        
        # 완성 템플릿
        self.completion_templates = {
            "법률조문": [
                "이 조항은 실제 생활에서 매우 중요한 역할을 합니다.",
                "이러한 요건들이 충족되면 손해배상 책임이 발생하게 됩니다.",
                "이렇게 이해하시면 됩니다."
            ],
            "계약서": [
                "이렇게 계약서를 작성하시면 안전한 거래가 가능합니다.",
                "계약서에 이런 내용들을 포함하시는 것이 중요합니다.",
                "이렇게 진행하시면 됩니다."
            ],
            "부동산": [
                "이러한 절차를 거쳐 안전한 부동산 거래를 하실 수 있습니다.",
                "이렇게 단계별로 진행하시면 됩니다.",
                "이렇게 하시면 안전한 매매가 가능합니다."
            ],
            "가족법": [
                "이러한 절차를 거쳐 이혼 소송을 진행하실 수 있습니다.",
                "이렇게 단계별로 진행하시면 됩니다.",
                "이렇게 하시면 됩니다."
            ],
            "민사법": [
                "이러한 방법으로 손해배상을 청구하실 수 있습니다.",
                "이렇게 단계별로 진행하시면 됩니다.",
                "이렇게 하시면 됩니다."
            ],
            "일반": [
                "이렇게 진행하시면 됩니다.",
                "이렇게 하시면 됩니다.",
                "더 궁금한 점이 있으시면 언제든지 물어보세요."
            ]
        }
    
    def force_complete_answer(self, answer: str, question: str, category: str = "일반") -> CompletionResult:
        """답변을 강제로 완성"""
        try:
            # 1. 불완전한 패턴 감지
            is_truncated = self._detect_truncation(answer)
            
            if not is_truncated:
                return CompletionResult(
                    completed_answer=answer,
                    was_truncated=False,
                    completion_method="no_completion_needed",
                    confidence=1.0
                )
            
            # 2. 강제 완성 시도
            completed_answer = self._apply_forced_completion(answer, category)
            
            # 3. 완성도 검증
            if self._is_properly_completed(completed_answer):
                return CompletionResult(
                    completed_answer=completed_answer,
                    was_truncated=True,
                    completion_method="forced_completion",
                    confidence=0.9
                )
            
            # 4. 폴백 완성
            fallback_answer = self._apply_fallback_completion(answer, category)
            return CompletionResult(
                completed_answer=fallback_answer,
                was_truncated=True,
                completion_method="fallback_completion",
                confidence=0.7
            )
            
        except Exception as e:
            print(f"강제 완성 실패: {e}")
            return CompletionResult(
                completed_answer=self._apply_emergency_completion(answer),
                was_truncated=True,
                completion_method="emergency_completion",
                confidence=0.5
            )
    
    def _detect_truncation(self, answer: str) -> bool:
        """불완전한 답변 감지"""
        for pattern in self.truncation_patterns:
            if re.search(pattern, answer.strip()):
                return True
        
        # 문장이 적절히 끝나지 않은 경우
        if not answer.strip().endswith(('.', '!', '?', '니다.', '습니다.', '요.')):
            return True
        
        return False
    
    def _apply_forced_completion(self, answer: str, category: str) -> str:
        """강제 완성 적용"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            completion_prompt = f"""
다음 답변을 반드시 완성해주세요. 절대 중간에 끊지 마세요.

답변: {answer}
카테고리: {category}

요구사항:
1. 답변을 반드시 완전히 마무리하세요
2. 마지막 문장을 완전히 끝내세요
3. 자연스러운 결론을 추가하세요
4. 최소 50자 이상 추가하세요
5. 법률 관련 내용이므로 정확하고 신중하게 작성하세요

완성된 답변:"""
            
            response = gemini_client.generate(completion_prompt, question_type=category)
            return response.response
            
        except Exception as e:
            print(f"강제 완성 실패: {e}")
            return self._apply_fallback_completion(answer, category)
    
    def _apply_fallback_completion(self, answer: str, category: str) -> str:
        """폴백 완성 적용"""
        templates = self.completion_templates.get(category, self.completion_templates["일반"])
        
        # 불완전한 패턴에 따라 다른 완성 템플릿 적용
        if answer.strip().endswith('드'):
            return f"{answer.strip()} 이렇게 진행하시면 됩니다."
        elif answer.strip().endswith(('그리고', '또한')):
            return f"{answer.strip()} 더 궁금한 점이 있으시면 언제든지 물어보세요."
        elif answer.strip().endswith(('마지막으로', '결론적으로')):
            return f"{answer.strip()} 이렇게 하시면 됩니다."
        else:
            return f"{answer.strip()} {templates[0]}"
    
    def _apply_emergency_completion(self, answer: str) -> str:
        """긴급 완성 적용"""
        if answer.strip().endswith(('드', '그리고', '또한')):
            return f"{answer.strip()} 이렇게 하시면 됩니다."
        else:
            return f"{answer.strip()} 더 궁금한 점이 있으시면 언제든지 물어보세요."
    
    def _is_properly_completed(self, answer: str) -> bool:
        """완성도 검증"""
        # 최소 길이 확인
        if len(answer.strip()) < 100:
            return False
        
        # 적절한 마무리 확인
        if not answer.strip().endswith(('.', '!', '?', '니다.', '습니다.', '요.')):
            return False
        
        # 불완전한 패턴이 없는지 확인
        for pattern in self.truncation_patterns:
            if re.search(pattern, answer.strip()):
                return False
        
        return True

# 전역 인스턴스
enhanced_completion_system = EnhancedCompletionSystem()
