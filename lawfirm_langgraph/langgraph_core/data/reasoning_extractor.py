# -*- coding: utf-8 -*-
"""
ì¶”ë¡  ê³¼ì • ë¶„ë¦¬ ë°?ê²€ì¦?ëª¨ë“ˆ
LLM ?‘ë‹µ?ì„œ ì¶”ë¡  ê³¼ì •(Chain-of-Thought)??ì¶”ì¶œ?˜ê³  ?¤ì œ ?µë?ë§?ë¶„ë¦¬
"""

import logging
import re
from typing import Any, Dict, Optional

from langgraph_core.utils.workflow_constants import AnswerExtractionPatterns


class ReasoningExtractor:
    """
    ì¶”ë¡  ê³¼ì • ë¶„ë¦¬ ë°?ê²€ì¦??´ë˜??

    LLM ?‘ë‹µ?ì„œ ì¶”ë¡  ê³¼ì •ê³??¤ì œ ?µë???ë¶„ë¦¬?˜ê³ ,
    ?ˆì§ˆ??ê²€ì¦í•˜??ê¸°ëŠ¥???œê³µ?©ë‹ˆ??
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        ReasoningExtractor ì´ˆê¸°??

        Args:
            logger: ë¡œê±° ?¸ìŠ¤?´ìŠ¤ (?†ìœ¼ë©??ë™ ?ì„±)
        """
        self.logger = logger or logging.getLogger(__name__)
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """?•ê·œ???¨í„´??ì»´íŒŒ?¼í•˜??ìºì‹± (?±ëŠ¥ ìµœì ??"""
        # ì¶”ë¡  ê³¼ì • ?¹ì…˜ ?¨í„´ ì»´íŒŒ??
        self._compiled_reasoning_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.REASONING_SECTION_PATTERNS
        ]

        # ì¶œë ¥ ?¹ì…˜ ?¨í„´ ì»´íŒŒ??
        self._compiled_output_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS
        ]

        # ?µë? ?¹ì…˜ ?¨í„´ ì»´íŒŒ??
        self._compiled_answer_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS
        ]

        # Step ?¤ë” ?¨í„´ ì»´íŒŒ??(Step 1, 2, 3)
        self._compiled_step_header_patterns = {
            "step1": [
                re.compile(r'###\s*Step\s*1[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*?¨ê³„\s*1[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*1\s*[:ï¼?', re.IGNORECASE | re.MULTILINE),
            ],
            "step2": [
                re.compile(r'###\s*Step\s*2[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*?¨ê³„\s*2[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*2\s*[:ï¼?', re.IGNORECASE | re.MULTILINE),
            ],
            "step3": [
                re.compile(r'###\s*Step\s*3[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*?¨ê³„\s*3[:ï¼?', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*3\s*[:ï¼?', re.IGNORECASE | re.MULTILINE),
            ],
        }

        # ?¤ìŒ ë§ˆì»¤ ?¨í„´ ì»´íŒŒ??
        self._compiled_next_marker_patterns = {
            "step1": [
                re.compile(r'\n\s*###\s*Step\s*2', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*###\s*?¨ê³„\s*2', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*?“¤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*ì¶œë ¥', re.IGNORECASE | re.MULTILINE),
            ],
            "step2": [
                re.compile(r'\n\s*###\s*Step\s*3', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*###\s*?¨ê³„\s*3', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*?“¤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*ì¶œë ¥', re.IGNORECASE | re.MULTILINE),
            ],
            "step3": [
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*?“¤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*ì¶œë ¥', re.IGNORECASE | re.MULTILINE),
            ],
        }

        # ë¶€ë¶??•ë¦¬ ?¨í„´ ì»´íŒŒ??
        self._compiled_partial_cleaning_patterns = [
            re.compile(r'##\s*?§ \s*ì¶”ë¡ [^\n]*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'###\s*Step\s*[123][^\n]*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'###\s*?¨ê³„\s*[123][^\n]*', re.IGNORECASE | re.MULTILINE),
        ]

        # ?¤ìŒ ?¹ì…˜ ?¨í„´ ì»´íŒŒ??(ë©´ì±…ì¡°í•­, ì°¸ê³ ?ë£Œ ??
        self._compiled_next_section_patterns = [
            re.compile(r'###?\s*?“š', re.IGNORECASE),
            re.compile(r'###?\s*ì°¸ê³ ?ë£Œ', re.IGNORECASE),
            re.compile(r'###?\s*?’¡', re.IGNORECASE),
            re.compile(r'###?\s*? ë¢°??, re.IGNORECASE),
            re.compile(r'---'),
            re.compile(r'?’¼\s*ë©´ì±…', re.IGNORECASE),
        ]

    def extract_reasoning(self, answer: str) -> Dict[str, Any]:
        """
        LLM ?µë??ì„œ ì¶”ë¡  ê³¼ì •(Chain-of-Thought) ?¹ì…˜??ì¶”ì¶œ

        Args:
            answer: LLM ?ë³¸ ?µë? ë¬¸ì??

        Returns:
            Dict with keys:
                - "reasoning": ì¶”ë¡  ê³¼ì • ?„ì²´ ?´ìš©
                - "step1": Step 1 ?´ìš©
                - "step2": Step 2 ?´ìš©
                - "step3": Step 3 ?´ìš©
                - "has_reasoning": ì¶”ë¡  ê³¼ì • ì¡´ì¬ ?¬ë?
                - "reasoning_section_count": ì¶”ë¡  ?¹ì…˜ ê°œìˆ˜
        """
        if not answer or not isinstance(answer, str):
            return {
                "reasoning": "",
                "step1": "",
                "step2": "",
                "step3": "",
                "has_reasoning": False,
                "reasoning_section_count": 0
            }

        # ?±ëŠ¥ ìµœì ?? ?€?©ëŸ‰ ?‘ë‹µ???€??ì¡°ê¸° ì¢…ë£Œ ì¡°ê±´
        answer_length = len(answer)
        MAX_REASONING_SEARCH_LENGTH = 10000  # 10KB ?´ìƒ??ê²½ìš° ?±ëŠ¥ ìµœì ???ìš©

        if answer_length > MAX_REASONING_SEARCH_LENGTH:
            # ?€?©ëŸ‰ ?‘ë‹µ??ê²½ìš° ì²˜ìŒ 10KBë§?ê²€??(ì¶”ë¡  ê³¼ì •?€ ë³´í†µ ?ë?ë¶„ì— ?ˆìŒ)
            search_text = answer[:MAX_REASONING_SEARCH_LENGTH]
            self.logger.debug(f"Large response ({answer_length} chars), searching first {MAX_REASONING_SEARCH_LENGTH} chars only")
        else:
            search_text = answer

        result = {
            "reasoning": "",
            "step1": "",
            "step2": "",
            "step3": "",
            "has_reasoning": False,
            "reasoning_section_count": 0
        }

        # ì¶”ë¡  ê³¼ì • ?¹ì…˜ ì°¾ê¸° (?¬ëŸ¬ ê³³ì— ë¶„ì‚°??ê²½ìš°??ì²˜ë¦¬)
        reasoning_sections = []  # ëª¨ë“  ì¶”ë¡  ?¹ì…˜???€??

        # ëª¨ë“  ì¶”ë¡  ?¹ì…˜ ì°¾ê¸° (search_text ?¬ìš©)
        reasoning_found = False
        for compiled_pattern in self._compiled_reasoning_patterns:
            # ëª¨ë“  ë§¤ì¹­ ì°¾ê¸° (ì²?ë²ˆì§¸ë§Œì´ ?„ë‹ˆ??
            for match in compiled_pattern.finditer(search_text):
                reasoning_found = True
                reasoning_start_idx = match.start()
                # ?¤ìŒ ?¹ì…˜ ì°¾ê¸° (ì¶œë ¥ ?¹ì…˜ ?ëŠ” ?µë? ?¹ì…˜) - search_text ê¸°ì?
                remaining_text = search_text[reasoning_start_idx:]
                next_section_patterns = (
                    self._compiled_output_patterns +
                    self._compiled_answer_patterns
                )

                reasoning_end_idx = None
                for compiled_next_pattern in next_section_patterns:
                    next_match = compiled_next_pattern.search(remaining_text)
                    if next_match:
                        reasoning_end_idx = reasoning_start_idx + next_match.start()
                        break

                if reasoning_end_idx is None:
                    # ?¤ìŒ ?¹ì…˜??ì°¾ì? ëª»í•œ ê²½ìš°, ì¶”ë¡  ?¹ì…˜ ?ê¹Œì§€ ?„ì²´ ?¬í•¨
                    # search_text ê¸°ì??´ë?ë¡??ë³¸ answer ê¸¸ì´ë¡?ì¡°ì •
                    if answer_length > MAX_REASONING_SEARCH_LENGTH and reasoning_start_idx < MAX_REASONING_SEARCH_LENGTH:
                        # ?€?©ëŸ‰ ?‘ë‹µ?ì„œ ê²€??ë²”ìœ„ë¥?ë²—ì–´??ê²½ìš° ?ë³¸ ?ê¹Œì§€
                        reasoning_end_idx = answer_length
                    else:
                        reasoning_end_idx = answer_length if reasoning_start_idx < answer_length else len(search_text)

                # ì¤‘ë³µ ì²´í¬ (?´ë? ?¬í•¨???¹ì…˜?¸ì? ?•ì¸)
                is_duplicate = False
                for existing_start, existing_end in reasoning_sections:
                    if reasoning_start_idx >= existing_start and reasoning_start_idx < existing_end:
                        is_duplicate = True
                        break
                    if existing_start >= reasoning_start_idx and existing_start < reasoning_end_idx:
                        # ê¸°ì¡´ ?¹ì…˜???„ì¬ ?¹ì…˜???¬í•¨?˜ì–´ ?ˆìœ¼ë©??•ì¥
                        reasoning_sections.remove((existing_start, existing_end))
                        break

                if not is_duplicate:
                    reasoning_sections.append((reasoning_start_idx, reasoning_end_idx))

        # ?¬ëŸ¬ ì¶”ë¡  ?¹ì…˜???ˆëŠ” ê²½ìš° ë³‘í•©
        if reasoning_sections:
            # ?œì‘ ?„ì¹˜ ?œìœ¼ë¡??•ë ¬
            reasoning_sections.sort(key=lambda x: x[0])

            # ê²¹ì¹˜???¹ì…˜ ë³‘í•©
            merged_sections = []
            current_start, current_end = reasoning_sections[0]

            for start_idx, end_idx in reasoning_sections[1:]:
                if start_idx <= current_end:
                    # ê²¹ì¹˜??ê²½ìš° ë³‘í•©
                    current_end = max(current_end, end_idx)
                else:
                    # ê²¹ì¹˜ì§€ ?ŠëŠ” ê²½ìš° ?€?¥í•˜ê³??ˆë¡œ ?œì‘
                    merged_sections.append((current_start, current_end))
                    current_start, current_end = start_idx, end_idx

            merged_sections.append((current_start, current_end))

            # ë³‘í•©???¹ì…˜?¤ì„ ?˜ë‚˜???ìŠ¤?¸ë¡œ ?©ì¹˜ê¸?
            reasoning_text_parts = []
            for start_idx, end_idx in merged_sections:
                # ?ë³¸ answer?ì„œ ì¶”ì¶œ (?¸ë±?¤ëŠ” search_text ê¸°ì??´ì?ë§?answer?ì„œ ê°€?¸ì˜´)
                if start_idx < answer_length:
                    end_idx = min(end_idx, answer_length)
                    section_text = answer[start_idx:end_idx].strip()
                    if section_text:
                        reasoning_text_parts.append(section_text)

            reasoning_text = "\n\n".join(reasoning_text_parts)
            result["reasoning"] = reasoning_text
            result["has_reasoning"] = True
            result["reasoning_section_count"] = len(merged_sections)  # ?¹ì…˜ ê°œìˆ˜ ?€??

            # Step 1, 2, 3 ì¶”ì¶œ (Step ?¤ë”ë¥?ì§ì ‘ ì°¾ì•„??ê·??¬ì´???´ìš© ì¶”ì¶œ)
            # ?¬ëŸ¬ ?¹ì…˜?ì„œ Step ì¶”ì¶œ ??ë³‘í•©
            step_keys = ["step1", "step2", "step3"]

            # ?±ëŠ¥ ìµœì ?? reasoning_textê°€ ?ˆë¬´ ê¸?ê²½ìš° ì¡°ê¸° ì¢…ë£Œ
            MAX_STEP_SEARCH_LENGTH = 5000  # Step ì¶”ì¶œ??5KB ?œí•œ
            step_search_text = reasoning_text[:MAX_STEP_SEARCH_LENGTH] if len(reasoning_text) > MAX_STEP_SEARCH_LENGTH else reasoning_text

            # ê°?Step???€??ëª¨ë“  ?¹ì…˜?ì„œ ì¶”ì¶œ?˜ì—¬ ë³‘í•©
            for step_key in step_keys:
                step_contents = []  # ?¬ëŸ¬ ?¹ì…˜?ì„œ ì¶”ì¶œ??Step ?´ìš© ?€??

                # ë³‘í•©??ì¶”ë¡  ?ìŠ¤?¸ì—???´ë‹¹ Step ì°¾ê¸° (step_search_text ?¬ìš©)
                compiled_headers = self._compiled_step_header_patterns[step_key]
                for compiled_header in compiled_headers:
                    # ëª¨ë“  Step ?¤ë” ì°¾ê¸° (?¬ëŸ¬ ?¹ì…˜???ˆì„ ???ˆìŒ)
                    full_header_pattern = compiled_header.pattern + r'[^\n]*\n'
                    full_header_compiled = re.compile(full_header_pattern, re.IGNORECASE | re.MULTILINE)

                    # ëª¨ë“  ë§¤ì¹­ ì°¾ê¸° (step_search_text ?¬ìš©)
                    for match in full_header_compiled.finditer(step_search_text):
                        step_start = match.end()

                        # ?¤ìŒ ë§ˆì»¤ ì°¾ê¸° (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©) - step_search_text ê¸°ì?
                        remaining_text = step_search_text[step_start:]
                        compiled_markers = self._compiled_next_marker_patterns[step_key]
                        step_end = None

                        for compiled_marker in compiled_markers:
                            next_match = compiled_marker.search(remaining_text)
                            if next_match:
                                step_end = step_start + next_match.start()
                                break

                        if step_end is None:
                            # ?¤ìŒ ë§ˆì»¤ë¥?ì°¾ì? ëª»í•œ ê²½ìš° ?ê¹Œì§€ (reasoning_text ê¸°ì??¼ë¡œ ì¡°ì •)
                            step_end = min(step_start + len(remaining_text), len(reasoning_text))

                        # Step ?´ìš© ì¶”ì¶œ (?¤ë” ì¤??œì™¸, ë¹?ì¤??œê±°) - reasoning_text?ì„œ
                        step_content = reasoning_text[step_start:step_end].strip() if step_start < len(reasoning_text) else ""
                        if step_content:
                            step_contents.append(step_content)

                    # ì²?ë²ˆì§¸ ?¤ë” ?¨í„´?ì„œ ì°¾ìœ¼ë©??¤ìŒ ?¨í„´?¼ë¡œ ?˜ì–´ê°€ì§€ ?ŠìŒ
                    if step_contents:
                        break

                # ?¬ëŸ¬ ?¹ì…˜?ì„œ ì¶”ì¶œ??Step ?´ìš© ë³‘í•© (ì¤‘ë³µ ?œê±° ë°??•ë ¬)
                if step_contents:
                    # ì¤‘ë³µ ?œê±° (?™ì¼???´ìš©?€ ??ë²ˆë§Œ)
                    unique_contents = []
                    seen_contents = set()
                    for content in step_contents:
                        # ?•ê·œ?”í•˜??ë¹„êµ (ê³µë°±, ì¤„ë°”ê¿??•ë¦¬)
                        normalized = re.sub(r'\s+', ' ', content.strip())
                        if normalized and normalized not in seen_contents:
                            seen_contents.add(normalized)
                            unique_contents.append(content)

                    # ë³‘í•© (ë¹?ì¤„ë¡œ êµ¬ë¶„)
                    if unique_contents:
                        merged_step_content = "\n\n".join(unique_contents)
                        result[step_key] = merged_step_content
                elif step_key == "step3":
                    # Step 3??ê²½ìš° ?´ìš©??ë¹„ì–´?ˆì„ ?˜ë„ ?ˆì?ë§??¤ë”??ì¡´ì¬?˜ë?ë¡??€??
                    result[step_key] = ""

        return result

    def extract_actual_answer(self, llm_response: str) -> str:
        """
        LLM ?‘ë‹µ?ì„œ ?¤ì œ ?µë?ë§?ì¶”ì¶œ (ì¶”ë¡  ê³¼ì • ?œì™¸)

        ?°ì„ ?œìœ„:
        1. "## ?“¤ ì¶œë ¥" ?¹ì…˜
        2. "## ?µë?" ?¹ì…˜ (ì¶”ë¡  ê³¼ì • ?œì™¸)
        3. ?„ì²´ ?´ìš©?ì„œ ì¶”ë¡  ê³¼ì • ?œê±° ???¨ì? ë¶€ë¶?

        Args:
            llm_response: LLM ?ë³¸ ?‘ë‹µ ë¬¸ì??

        Returns:
            ?¤ì œ ?µë?ë§??¬í•¨??ë¬¸ì??
        """
        if not llm_response or not isinstance(llm_response, str):
            return ""

        # 1?¨ê³„: ì¶œë ¥ ?¹ì…˜ ì°¾ê¸° (ìµœìš°?? ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
        for compiled_pattern in self._compiled_output_patterns:
            match = compiled_pattern.search(llm_response)
            if match:
                # ì¶œë ¥ ?¹ì…˜ ?´í›„??ëª¨ë“  ?´ìš©
                output_start = match.end()
                remaining = llm_response[output_start:].strip()

                # ?¤ìŒ ?¹ì…˜(ë©´ì±…ì¡°í•­, ì°¸ê³ ?ë£Œ ?? ?„ê¹Œì§€ ì¶”ì¶œ (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
                next_section_idx = len(remaining)
                for compiled_next_pattern in self._compiled_next_section_patterns:
                    next_match = compiled_next_pattern.search(remaining)
                    if next_match:
                        next_section_idx = min(next_section_idx, next_match.start())

                answer = remaining[:next_section_idx].strip()
                if answer:
                    return answer

        # 2?¨ê³„: ?µë? ?¹ì…˜ ì°¾ê¸° (ì¶”ë¡  ê³¼ì • ?œì™¸)
        reasoning_info = self.extract_reasoning(llm_response)
        if reasoning_info["has_reasoning"]:
            # ì¶”ë¡  ê³¼ì • ?œê±°
            reasoning_start = llm_response.find(reasoning_info["reasoning"])
            if reasoning_start != -1:
                # ì¶”ë¡  ê³¼ì • ?´í›„ ë¶€ë¶?ì¶”ì¶œ
                after_reasoning_start = reasoning_start + len(reasoning_info["reasoning"])
                after_reasoning = llm_response[after_reasoning_start:].strip()

                # ?µë? ?¹ì…˜??ì¶”ë¡  ê³¼ì • ?´í›„???ˆëŠ”ì§€ ?•ì¸ (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
                for compiled_pattern in self._compiled_answer_patterns:
                    match = compiled_pattern.search(after_reasoning)
                    if match:
                        answer_start = match.end()
                        answer = after_reasoning[answer_start:].strip()
                        # ?¤ìŒ ?¹ì…˜ ?„ê¹Œì§€ (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
                        next_section_idx = len(answer)
                        for compiled_next_pattern in self._compiled_next_section_patterns:
                            next_match = compiled_next_pattern.search(answer)
                            if next_match:
                                next_section_idx = min(next_section_idx, next_match.start())

                        answer = answer[:next_section_idx].strip()
                        if answer:
                            return answer

        # 3?¨ê³„: ?„ì²´ ?´ìš©?ì„œ ì¶”ë¡  ê³¼ì • ?œê±° (ëª¨ë“  ?¹ì…˜ ?„ì „ ?œê±°)
        if reasoning_info["has_reasoning"]:
            reasoning_text = reasoning_info["reasoning"]
            reasoning_section_count = reasoning_info.get("reasoning_section_count", 1)

            # ?¬ëŸ¬ ?¹ì…˜???ˆëŠ” ê²½ìš° ê°??¹ì…˜??ê°œë³„?ìœ¼ë¡??œê±°
            if reasoning_section_count > 1:
                # ?ë³¸?ì„œ ëª¨ë“  ì¶”ë¡  ?¹ì…˜???¸ë±?¤ë? ì°¾ì•„???€??
                reasoning_section_indices = []
                for compiled_pattern in self._compiled_reasoning_patterns:
                    # ëª¨ë“  ì¶”ë¡  ?¹ì…˜ ì°¾ê¸°
                    for match in compiled_pattern.finditer(llm_response):
                        reasoning_start = match.start()
                        # ?¤ìŒ ?¹ì…˜ ì°¾ê¸° (ì¶œë ¥ ?¹ì…˜ ?ëŠ” ?µë? ?¹ì…˜)
                        remaining_text = llm_response[reasoning_start:]
                        next_section_patterns = (
                            self._compiled_output_patterns +
                            self._compiled_answer_patterns
                        )
                        reasoning_end = None
                        for compiled_next_pattern in next_section_patterns:
                            next_match = compiled_next_pattern.search(remaining_text)
                            if next_match:
                                reasoning_end = reasoning_start + next_match.start()
                                break

                        if reasoning_end is None:
                            reasoning_end = len(llm_response)

                        # ì¤‘ë³µ ì²´í¬
                        is_duplicate = False
                        for existing_start, existing_end in reasoning_section_indices:
                            if reasoning_start >= existing_start and reasoning_start < existing_end:
                                is_duplicate = True
                                break
                            if existing_start >= reasoning_start and existing_start < reasoning_end:
                                # ê¸°ì¡´ ?¹ì…˜???„ì¬ ?¹ì…˜???¬í•¨?˜ì–´ ?ˆìœ¼ë©?êµì²´
                                reasoning_section_indices.remove((existing_start, existing_end))
                                break

                        if not is_duplicate and reasoning_end > reasoning_start:
                            reasoning_section_indices.append((reasoning_start, reasoning_end))

                # ??ˆœ?¼ë¡œ ?•ë ¬?˜ì—¬ ?œê±° (?¤ì—?œë????œê±°?˜ë©´ ?¸ë±??ë³€???†ìŒ)
                reasoning_section_indices.sort(key=lambda x: x[0], reverse=True)

                # ê°??¹ì…˜????ˆœ?¼ë¡œ ?œê±°
                cleaned_response = llm_response
                for reasoning_start, reasoning_end in reasoning_section_indices:
                    cleaned_response = (
                        cleaned_response[:reasoning_start] +
                        cleaned_response[reasoning_end:]
                    )

                cleaned_response = cleaned_response.strip()
            else:
                # ?¨ì¼ ?¹ì…˜??ê²½ìš° ê¸°ì¡´ ë°©ì‹ ?¬ìš©
                cleaned_response = llm_response.replace(reasoning_text, "").strip()

            # ?µë? ?¹ì…˜ ì°¾ê¸° (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
            for compiled_pattern in self._compiled_answer_patterns:
                match = compiled_pattern.search(cleaned_response)
                if match:
                    answer_start = match.end()
                    answer = cleaned_response[answer_start:].strip()
                    # ?¤ìŒ ?¹ì…˜ ?„ê¹Œì§€ (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
                    next_section_idx = len(answer)
                    for compiled_next_pattern in self._compiled_next_section_patterns:
                        next_match = compiled_next_pattern.search(answer)
                        if next_match:
                            next_section_idx = min(next_section_idx, next_match.start())

                    answer = answer[:next_section_idx].strip()
                    if answer:
                        return answer

            # ?µë? ?¹ì…˜???†ìœ¼ë©?ì¶”ë¡  ê³¼ì • ?œê±° ???¨ì? ?´ìš© ë°˜í™˜
            if cleaned_response:
                return cleaned_response

        # 4?¨ê³„: ì¶”ë¡  ê³¼ì •???†ê±°??ì°¾ì? ëª»í•œ ê²½ìš° ë¶€ë¶?ë¶„ë¦¬ ?œë„ (ì»´íŒŒ?¼ëœ ?¨í„´ ?¬ìš©)
        # ì¶”ë¡  ê³¼ì • ?¹ì…˜ ?¤ì›Œ?œê? ?ˆëŠ” ê²½ìš° ë¶€ë¶„ì ?¼ë¡œ ?œê±° ?œë„
        cleaned_response = llm_response
        for compiled_pattern in self._compiled_partial_cleaning_patterns:
            cleaned_response = compiled_pattern.sub('', cleaned_response)

        # ë¹?ì¤??•ë¦¬
        cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)

        # ë³€ê²½ì´ ?ˆì—ˆ?¼ë©´ ë°˜í™˜
        if cleaned_response.strip() != llm_response.strip():
            return cleaned_response.strip()

        # ëª¨ë“  ë°©ë²• ?¤íŒ¨ ???ë³¸ ë°˜í™˜
        return llm_response

    def extract_by_output_section(self, llm_response: str) -> str:
        """
        ì¶œë ¥ ?¹ì…˜?ì„œ ?¤ì œ ?µë? ì¶”ì¶œ (?¬ì‹œ??ë¡œì§??helper)

        Args:
            llm_response: LLM ?ë³¸ ?‘ë‹µ

        Returns:
            ì¶”ì¶œ???µë? ?ëŠ” ë¹?ë¬¸ì??
        """
        if not llm_response or not isinstance(llm_response, str):
            return ""

        for compiled_pattern in self._compiled_output_patterns:
            match = compiled_pattern.search(llm_response)
            if match:
                output_start = match.end()
                remaining = llm_response[output_start:].strip()

                next_section_idx = len(remaining)
                for compiled_next_pattern in self._compiled_next_section_patterns:
                    next_match = compiled_next_pattern.search(remaining)
                    if next_match:
                        next_section_idx = min(next_section_idx, next_match.start())

                answer = remaining[:next_section_idx].strip()
                if answer:
                    return answer

        return ""

    def extract_by_removing_reasoning(self, llm_response: str, reasoning_info: Dict[str, Any]) -> str:
        """
        ì¶”ë¡  ê³¼ì • ?œê±° ???¤ì œ ?µë? ì¶”ì¶œ (?¬ì‹œ??ë¡œì§??helper)

        Args:
            llm_response: LLM ?ë³¸ ?‘ë‹µ
            reasoning_info: ì¶”ì¶œ??ì¶”ë¡  ê³¼ì • ?•ë³´

        Returns:
            ì¶”ì¶œ???µë? ?ëŠ” ë¹?ë¬¸ì??
        """
        if not reasoning_info.get("has_reasoning"):
            return ""

        reasoning_text = reasoning_info.get("reasoning", "")
        if not reasoning_text:
            return ""

        reasoning_section_count = reasoning_info.get("reasoning_section_count", 1)

        # ?¬ëŸ¬ ?¹ì…˜???ˆëŠ” ê²½ìš° ê°??¹ì…˜??ê°œë³„?ìœ¼ë¡??œê±°
        if reasoning_section_count > 1:
            reasoning_section_indices = []
            for compiled_pattern in self._compiled_reasoning_patterns:
                for match in compiled_pattern.finditer(llm_response):
                    reasoning_start = match.start()
                    remaining_text = llm_response[reasoning_start:]
                    next_section_patterns = (
                        self._compiled_output_patterns +
                        self._compiled_answer_patterns
                    )
                    reasoning_end = None
                    for compiled_next_pattern in next_section_patterns:
                        next_match = compiled_next_pattern.search(remaining_text)
                        if next_match:
                            reasoning_end = reasoning_start + next_match.start()
                            break

                    if reasoning_end is None:
                        reasoning_end = len(llm_response)

                    is_duplicate = False
                    for existing_start, existing_end in reasoning_section_indices:
                        if reasoning_start >= existing_start and reasoning_start < existing_end:
                            is_duplicate = True
                            break
                        if existing_start >= reasoning_start and existing_start < reasoning_end:
                            reasoning_section_indices.remove((existing_start, existing_end))
                            break

                    if not is_duplicate and reasoning_end > reasoning_start:
                        reasoning_section_indices.append((reasoning_start, reasoning_end))

            reasoning_section_indices.sort(key=lambda x: x[0], reverse=True)

            cleaned_response = llm_response
            for reasoning_start, reasoning_end in reasoning_section_indices:
                cleaned_response = (
                    cleaned_response[:reasoning_start] +
                    cleaned_response[reasoning_end:]
                )

            cleaned_response = cleaned_response.strip()
        else:
            cleaned_response = llm_response.replace(reasoning_text, "").strip()

        # ?µë? ?¹ì…˜ ì°¾ê¸°
        for compiled_pattern in self._compiled_answer_patterns:
            match = compiled_pattern.search(cleaned_response)
            if match:
                answer_start = match.end()
                answer = cleaned_response[answer_start:].strip()

                next_section_idx = len(answer)
                for compiled_next_pattern in self._compiled_next_section_patterns:
                    next_match = compiled_next_pattern.search(answer)
                    if next_match:
                        next_section_idx = min(next_section_idx, next_match.start())

                answer = answer[:next_section_idx].strip()
                if answer:
                    return answer

        # ?µë? ?¹ì…˜???†ìœ¼ë©?ì¶”ë¡  ê³¼ì • ?œê±° ???¨ì? ?´ìš© ë°˜í™˜
        if cleaned_response:
            return cleaned_response

        return ""

    def extract_by_partial_cleaning(self, llm_response: str) -> str:
        """
        ë¶€ë¶??•ë¦¬ ë°©ë²•?¼ë¡œ ì¶”ì¶œ (?¬ì‹œ??ë¡œì§??helper)

        Args:
            llm_response: LLM ?ë³¸ ?‘ë‹µ

        Returns:
            ì¶”ì¶œ???µë? ?ëŠ” ë¹?ë¬¸ì??
        """
        cleaned_response = llm_response
        for compiled_pattern in self._compiled_partial_cleaning_patterns:
            cleaned_response = compiled_pattern.sub('', cleaned_response)

        cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)

        if cleaned_response.strip() != llm_response.strip():
            return cleaned_response.strip()

        return ""

    def clean_reasoning_keywords(self, answer: str) -> str:
        """
        ?¤ì œ ?µë??ì„œ ì¶”ë¡  ê³¼ì • ?¤ì›Œ???”ì—¬ ?•ì¸ ë°??•ë¦¬

        Args:
            answer: ì¶”ì¶œ???¤ì œ ?µë?

        Returns:
            ì¶”ë¡  ê³¼ì • ?¤ì›Œ?œê? ?•ë¦¬???µë?
        """
        if not answer or not isinstance(answer, str):
            return answer

        # ì¶”ë¡  ê³¼ì • ê´€???¤ì›Œ???¨í„´
        reasoning_keywords = [
            r'##\s*?§ \s*ì¶”ë¡ ',
            r'###\s*Step\s*[123]',
            r'###\s*?¨ê³„\s*[123]',
            r'Chain-of-Thought',
            r'CoT',
            r'ì¶”ë¡ ê³¼ì •?‘ì„±',
        ]

        cleaned_answer = answer
        found_keywords = []

        # ?¤ì›Œ??ê²€??ë°??œê±°
        for keyword_pattern in reasoning_keywords:
            compiled_pattern = re.compile(keyword_pattern, re.IGNORECASE | re.MULTILINE)
            matches = list(compiled_pattern.finditer(cleaned_answer))
            if matches:
                found_keywords.append(keyword_pattern)
                # ?¤ì›Œ?œê? ?¬í•¨??ì¤??„ì²´ ?œê±°
                for match in reversed(matches):  # ??ˆœ?¼ë¡œ ?œê±°?˜ì—¬ ?¸ë±??ë³€??ë°©ì?
                    start_pos = match.start()
                    # ì¤??œì‘ ì°¾ê¸°
                    line_start = cleaned_answer.rfind('\n', 0, start_pos) + 1
                    # ì¤???ì°¾ê¸°
                    line_end = cleaned_answer.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(cleaned_answer)

                    # ?´ë‹¹ ì¤??œê±°
                    cleaned_answer = cleaned_answer[:line_start] + cleaned_answer[line_end + 1:]
                    cleaned_answer = cleaned_answer.lstrip()  # ?œì‘ ë¶€ë¶?ë¹?ì¤??œê±°

        # ë¹?ì¤??•ë¦¬ (?°ì†??ë¹?ì¤„ì„ ?˜ë‚˜ë¡?
        cleaned_answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_answer)

        # ?¤ì›Œ??ë°œê²¬ ??ë¡œê¹…
        if found_keywords:
            self.logger.warning(
                f"Found and removed reasoning keywords from answer: {found_keywords}"
            )

        return cleaned_answer.strip()

    def verify_extraction_quality(
        self,
        original_answer: str,
        actual_answer: str,
        reasoning_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ì¶”ë¡  ê³¼ì • ë¶„ë¦¬ ???ˆì§ˆ ê²€ì¦?

        Args:
            original_answer: ?ë³¸ ?µë?
            actual_answer: ì¶”ë¡  ê³¼ì • ?œê±°???¤ì œ ?µë?
            reasoning_info: ì¶”ì¶œ??ì¶”ë¡  ê³¼ì • ?•ë³´

        Returns:
            Dict: ?ˆì§ˆ ë©”íŠ¸ë¦?
                - "is_valid": ? íš¨ ?¬ë?
                - "warnings": ê²½ê³  ëª©ë¡
                - "errors": ?ëŸ¬ ëª©ë¡
                - "score": ?ˆì§ˆ ?ìˆ˜ (0.0-1.0)
        """
        quality = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "score": 1.0,
        }

        if not original_answer or not actual_answer:
            quality["is_valid"] = False
            quality["errors"].append("?ë³¸ ?µë? ?ëŠ” ?¤ì œ ?µë???ë¹„ì–´?ˆìŒ")
            quality["score"] = 0.0
            return quality

        # 1. ê¸¸ì´ ê²€ì¦?
        if len(actual_answer) == 0:
            quality["is_valid"] = False
            quality["errors"].append("?¤ì œ ?µë???ë¹„ì–´?ˆìŒ")
            quality["score"] = 0.0
            return quality

        # 2. ì¶”ë¡  ê³¼ì •???¤ì œ ?µë????¬í•¨?˜ì? ?Šì•˜?”ì? ?•ì¸
        if reasoning_info.get("has_reasoning"):
            reasoning_text = reasoning_info.get("reasoning", "")
            if reasoning_text and reasoning_text in actual_answer:
                quality["is_valid"] = False
                quality["errors"].append("ì¶”ë¡  ê³¼ì •???¤ì œ ?µë????¬í•¨?˜ì–´ ?ˆìŒ")
                quality["score"] = 0.0
            elif any(keyword in actual_answer for keyword in ["?§ ", "ì¶”ë¡ ê³¼ì •", "Step 1", "Step 2", "Step 3"]):
                quality["warnings"].append("ì¶”ë¡  ê³¼ì • ?¤ì›Œ?œê? ?¤ì œ ?µë????¨ì•„?ˆì„ ???ˆìŒ")
                quality["score"] = 0.8

        # 3. ?µë? ê¸¸ì´ ê²€ì¦?(?ˆë¬´ ì§§ìœ¼ë©?ë¬¸ì œ)
        extraction_ratio = len(actual_answer) / len(original_answer) if original_answer else 0.0
        if extraction_ratio < 0.1:
            quality["warnings"].append(f"?¤ì œ ?µë????ˆë¬´ ì§§ìŒ (ë¹„ìœ¨: {extraction_ratio:.1%})")
            quality["score"] = min(quality["score"], 0.7)
        elif extraction_ratio > 0.95:
            quality["warnings"].append("ì¶”ë¡  ê³¼ì •??ê±°ì˜ ?œê±°?˜ì? ?Šì•˜?????ˆìŒ")
            quality["score"] = min(quality["score"], 0.9)

        # 4. ?¤ì›Œ???¬í•¨ ?¬ë? ê²€ì¦?(ë²•ë¥  ê´€???¤ì›Œ?œê? ?¤ì œ ?µë????ˆëŠ”ì§€)
        legal_keywords = ["ë²?, "ì¡°ë¬¸", "?ë?", "ë²•ë¥ ", "ë¯¼ë²•", "?•ë²•", "?ë²•"]
        original_keywords = [kw for kw in legal_keywords if kw in original_answer]
        actual_keywords = [kw for kw in legal_keywords if kw in actual_answer]

        if original_keywords and len(actual_keywords) < len(original_keywords) * 0.5:
            quality["warnings"].append("ë²•ë¥  ?¤ì›Œ?œê? ë§ì´ ?œê±°?˜ì—ˆ?????ˆìŒ")
            quality["score"] = min(quality["score"], 0.8)

        # 5. ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
        if quality["errors"]:
            quality["score"] = 0.0
        elif quality["warnings"]:
            quality["score"] = quality["score"] * 0.9

        return quality
