#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
답변 완성도 검증기
답변이 완전한지 검증하고 불완전한 경우 재생성을 요청합니다.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CompletionCheck:
    """완성도 검사 결과"""
    is_complete: bool
    completion_score: float  # 0.0 ~ 1.0
    issues: List[str]
    suggestions: List[str]

class AnswerCompletionValidator:
    """답변 완성도 검증기"""

    def __init__(self):
        # 불완전한 답변의 지표들
        self.incomplete_indicators = [
            # 문장이 중간에 끊어진 경우
            r'드$', r'그리고$', r'또한$', r'마지막으로$', r'결론적으로$',
            r'예를 들어$', r'구체적으로$', r'특히$', r'또한$',
            # 불완전한 문장 패턴
            r'[가-힣]+드$', r'[가-힣]+고$', r'[가-힣]+며$',
            # 숫자나 목록이 중간에 끊어진 경우
            r'\d+\.\s*$', r'[가-힣]+:\s*$', r'[가-힣]+의\s*$'
        ]

        # 최소 길이 기준 (질문 유형별)
        self.min_length_standards = {
            '법률조문': 300,
            '계약서': 400,
            '부동산': 350,
            '가족법': 300,
            '민사법': 300,
            '일반': 200
        }

        # 완성도 평가 기준
        self.completion_criteria = {
            'has_introduction': 0.1,      # 도입부가 있는가
            'has_main_content': 0.4,      # 주요 내용이 충분한가
            'has_examples': 0.2,          # 예시가 있는가
            'has_conclusion': 0.1,        # 결론이 있는가
            'is_properly_ended': 0.2      # 적절히 마무리되었는가
        }

    def check_completion(self, answer: str, question: str = "", category: str = "일반") -> CompletionCheck:
        """답변의 완성도를 종합적으로 검사"""
        issues = []
        suggestions = []
        completion_score = 0.0

        # 1. 기본 길이 검사
        min_length = self.min_length_standards.get(category, 200)
        if len(answer.strip()) < min_length:
            issues.append(f"답변이 너무 짧습니다 (최소 {min_length}자 필요)")
            suggestions.append("더 자세한 설명을 추가하세요")

        # 2. 불완전한 문장 검사
        incomplete_found = False
        for pattern in self.incomplete_indicators:
            if re.search(pattern, answer.strip()):
                incomplete_found = True
                break

        if incomplete_found:
            issues.append("답변이 중간에 끊어져 있습니다")
            suggestions.append("문장을 완전히 마무리하세요")

        # 3. 완성도 요소별 평가
        completion_score = self._calculate_completion_score(answer, question)

        # 4. 전체 완성도 판단 (더 엄격한 기준)
        is_complete = (
            len(answer.strip()) >= min_length and
            not incomplete_found and
            completion_score >= 0.8  # 0.7에서 0.8로 상향 조정
        )

        if not is_complete:
            if completion_score < 0.5:
                suggestions.append("답변의 구조를 개선하세요")
            if not self._has_proper_ending(answer):
                suggestions.append("적절한 마무리를 추가하세요")

        return CompletionCheck(
            is_complete=is_complete,
            completion_score=completion_score,
            issues=issues,
            suggestions=suggestions
        )

    def _calculate_completion_score(self, answer: str, question: str = "") -> float:
        """완성도 점수 계산"""
        score = 0.0

        # 도입부 검사
        if self._has_introduction(answer, question):
            score += self.completion_criteria['has_introduction']

        # 주요 내용 검사
        if self._has_sufficient_content(answer):
            score += self.completion_criteria['has_main_content']

        # 예시 검사
        if self._has_examples(answer):
            score += self.completion_criteria['has_examples']

        # 결론 검사
        if self._has_conclusion(answer):
            score += self.completion_criteria['has_conclusion']

        # 적절한 마무리 검사
        if self._has_proper_ending(answer):
            score += self.completion_criteria['is_properly_ended']

        return min(score, 1.0)

    def _has_introduction(self, answer: str, question: str = "") -> bool:
        """도입부가 있는지 검사"""
        intro_patterns = [
            r'안녕하세요', r'네,', r'아,', r'말씀이시군요', r'궁금하시군요',
            r'설명해 드리겠습니다', r'알려드리겠습니다'
        ]

        for pattern in intro_patterns:
            if re.search(pattern, answer[:100]):
                return True
        return False

    def _has_sufficient_content(self, answer: str) -> bool:
        """충분한 주요 내용이 있는지 검사"""
        # 문단 수 검사 (최소 2개 문단)
        paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return False

        # 주요 내용 키워드 검사
        content_indicators = [
            r'요건', r'조건', r'절차', r'방법', r'과정', r'단계',
            r'첫째', r'둘째', r'셋째', r'1\.', r'2\.', r'3\.',
            r'중요', r'필요', r'주의', r'권장'
        ]

        content_count = 0
        for pattern in content_indicators:
            if re.search(pattern, answer):
                content_count += 1

        return content_count >= 2

    def _has_examples(self, answer: str) -> bool:
        """예시가 있는지 검사"""
        example_patterns = [
            r'예를 들어', r'구체적으로', r'실제로', r'예시',
            r'예시로', r'사례로', r'경우로'
        ]

        for pattern in example_patterns:
            if re.search(pattern, answer):
                return True
        return False

    def _has_conclusion(self, answer: str) -> bool:
        """결론이 있는지 검사"""
        conclusion_patterns = [
            r'결론적으로', r'요약하면', r'정리하면', r'마지막으로',
            r'따라서', r'그러므로', r'이상으로', r'이렇게'
        ]

        # 답변의 마지막 1/3 부분에서 검사
        last_third = answer[-len(answer)//3:]
        for pattern in conclusion_patterns:
            if re.search(pattern, last_third):
                return True
        return False

    def _has_proper_ending(self, answer: str) -> bool:
        """적절한 마무리가 있는지 검사"""
        # 문장이 적절히 끝나는지 검사
        proper_endings = ['.', '!', '?', '니다.', '습니다.', '요.']

        last_char = answer.strip()[-1] if answer.strip() else ''
        return last_char in proper_endings

    def request_completion(self, incomplete_answer: str, question: str, category: str = "일반") -> str:
        """불완전한 답변을 완성하도록 요청 (강화된 버전)"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()

            # 더 구체적인 완성 요청 프롬프트
            completion_prompt = f"""
다음 답변을 완성해주세요. 답변이 중간에 끊어져 있으므로 자연스럽게 완성해주세요.

질문: {question}
카테고리: {category}
불완전한 답변: {incomplete_answer}

완성 요구사항:
1. 문장을 완전히 마무리하세요 (절대 중간에 끊지 마세요)
2. 필요한 경우 요약이나 결론을 추가하세요
3. 자연스럽고 친근한 톤을 유지하세요
4. 구체적인 예시나 실용적인 조언을 포함하세요
5. 면책 조항은 추가하지 마세요 (별도로 처리됩니다)
6. 최소 200자 이상의 완전한 답변을 작성하세요
7. 마지막 문장을 반드시 완전히 마무리하세요

완성된 답변:"""

            response = gemini_client.generate(completion_prompt, question_type=category)
            completed_answer = response.response

            # 완성된 답변이 여전히 불완전한 경우 추가 처리
            if not self.is_complete(completed_answer):
                self.logger.warning("첫 번째 완성 시도가 여전히 불완전함. 추가 완성 시도")
                return self._force_completion(completed_answer, question, category)

            return completed_answer

        except Exception as e:
            # 예외 발생 시 폴백 처리
            return self._add_simple_ending(incomplete_answer)

    def _force_completion(self, answer: str, question: str, category: str) -> str:
        """강제로 답변을 완성하는 메서드"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()

            force_completion_prompt = f"""
다음 답변을 반드시 완성해주세요. 절대 중간에 끊지 마세요.

질문: {question}
답변: {answer}

요구사항:
1. 답변을 반드시 완전히 마무리하세요
2. 마지막 문장을 완전히 끝내세요
3. 자연스러운 결론을 추가하세요
4. 최소 100자 이상 추가하세요

완성된 답변:"""

            response = gemini_client.generate(force_completion_prompt, question_type=category)
            return response.response

        except Exception as e:
            # 예외 발생 시 폴백 처리
            return self._add_simple_ending(answer)

    def _add_simple_ending(self, answer: str) -> str:
        """간단한 마무리 추가 (폴백)"""
        if answer.strip().endswith(('드', '그리고', '또한')):
            return f"{answer.strip()} 더 궁금한 점이 있으시면 언제든지 물어보세요."
        elif not answer.strip().endswith(('.', '!', '?')):
            return f"{answer.strip()} 이렇게 진행하시면 됩니다."
        return answer.strip()

# 전역 인스턴스
completion_validator = AnswerCompletionValidator()
