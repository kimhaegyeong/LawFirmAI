# -*- coding: utf-8 -*-
"""답변 길이 조절 클래스"""

import re
import logging
from typing import Optional
from ..config.formatter_config import AnswerLengthConfig


class AnswerLengthAdjuster:
    """답변 길이 조절 담당"""
    
    def __init__(
        self, 
        config: Optional[AnswerLengthConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or AnswerLengthConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def adjust_length(
        self,
        answer: str,
        query_type: str,
        query_complexity: str,
        grounding_score: Optional[float] = None,
        quality_score: Optional[float] = None
    ) -> str:
        """답변 길이를 질의 유형에 맞게 조절"""
        if not answer:
            return answer

        current_length = len(answer)

        if query_complexity == "simple":
            min_len, max_len = self.config.simple_question
        elif query_complexity == "moderate":
            min_len, max_len = self.config.legal_analysis
        elif query_complexity == "complex":
            min_len, max_len = self.config.complex_question
        else:
            min_len, max_len = self.config.get_targets(query_type)

        original_max_len = max_len
        if grounding_score is not None and grounding_score >= 0.7:
            max_len = int(max_len * 1.5)
            self.logger.info(f"[ANSWER LENGTH] Quality-based adjustment: max_len increased from {original_max_len} to {max_len} (grounding_score: {grounding_score:.2f})")
        elif quality_score is not None and quality_score >= 0.8:
            max_len = int(max_len * 1.3)
            self.logger.info(f"[ANSWER LENGTH] Quality-based adjustment: max_len increased from {original_max_len} to {max_len} (quality_score: {quality_score:.2f})")
        elif grounding_score is not None and grounding_score >= 0.5:
            max_len = int(max_len * 1.2)
            self.logger.debug(f"[ANSWER LENGTH] Quality-based adjustment: max_len increased from {original_max_len} to {max_len} (grounding_score: {grounding_score:.2f})")

        if min_len <= current_length <= max_len:
            self.logger.debug(f"[ANSWER LENGTH] Length OK: {current_length} (target: {min_len}-{max_len})")
            return answer

        if current_length > max_len:
            return self._truncate_smart(answer, max_len)

        self.logger.debug(f"[ANSWER LENGTH] Too short: {current_length} (target: {min_len}-{max_len}), keeping as is")
        return answer
    
    def _truncate_smart(self, answer: str, max_len: int) -> str:
        """스마트하게 답변 자르기 (중요 섹션 우선)"""
        self.logger.info(f"[ANSWER LENGTH] Too long: {len(answer)}, adjusting to max {max_len}")
        
        sections = re.split(r'\n\n+', answer)

        important_sections = []
        other_sections = []

        for section in sections:
            if (re.search(r'\[법령:', section) or
                re.search(r'대법원', section) or
                re.search(r'제\s*\d+\s*조', section)):
                important_sections.append(section)
            else:
                other_sections.append(section)

        result = []
        current_len = 0

        for section in important_sections:
            if current_len + len(section) <= max_len:
                result.append(section)
                current_len += len(section)
            else:
                remaining = max_len - current_len - 10
                if remaining > 100:
                    result.append(section[:remaining] + "...")
                break

        for section in other_sections:
            if current_len + len(section) <= max_len:
                result.append(section)
                current_len += len(section)
            else:
                break

        adjusted_answer = '\n\n'.join(result)
        self.logger.info(f"[ANSWER LENGTH] Adjusted: {len(answer)} -> {len(adjusted_answer)}")
        return adjusted_answer

