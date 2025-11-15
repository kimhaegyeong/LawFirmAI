# -*- coding: utf-8 -*-
"""답변 정리 클래스"""

import re
import logging
from typing import Optional


class AnswerCleaner:
    """답변 텍스트 정리 담당"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def remove_metadata_sections(self, answer_text: str) -> str:
        """답변 텍스트에서 메타 정보 섹션 제거"""
        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        lines = answer_text.split('\n')
        cleaned_lines = []
        in_confidence_section = False
        in_reference_section = False
        in_disclaimer_section = False

        for i, line in enumerate(lines):
            if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                in_confidence_section = True
                continue

            if re.match(r'^###\s*📚\s*참고\s*자료', line, re.IGNORECASE):
                in_reference_section = True
                continue

            if line.strip() == '---':
                next_line_idx = i + 1
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx]
                    if re.search(r'면책|본 답변은.*일반적인|변호사와.*상담|개별.*사안', next_line, re.IGNORECASE):
                        in_disclaimer_section = True
                        continue
                continue
            elif re.match(r'^\s*💼\s*\*\*면책\s*조항\*\*', line, re.IGNORECASE):
                in_disclaimer_section = True
                continue

            if in_confidence_section:
                if re.match(r'^###\s+', line) or line.strip() == '---':
                    in_confidence_section = False
                    continue
                continue

            if in_reference_section:
                if re.match(r'^###\s+', line) or line.strip() == '---':
                    in_reference_section = False
                    continue
                continue

            if in_disclaimer_section:
                if re.match(r'^###\s+', line) or re.match(r'^##\s+', line):
                    in_disclaimer_section = False
                    continue
                continue

            if re.match(r'^\*\*상세\s*점수:\*\*', line, re.IGNORECASE):
                continue
            if re.match(r'^\*\*설명:\*\*', line, re.IGNORECASE):
                continue
            if re.match(r'^-\s*답변\s*품질:', line, re.IGNORECASE):
                continue
            if re.match(r'^-\s*신뢰도:', line, re.IGNORECASE):
                continue

            cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'\*\*상세\s*점수:\*\*.*?\n', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'-\s*답변\s*품질:\s*[\d.]+%?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_text = re.sub(r'\*\*설명:\*\*\s*신뢰도:.*?\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_text = re.sub(r'-\s*신뢰도:\s*[\d.]+%?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()
    
    def remove_answer_header(self, answer_text: str) -> str:
        """답변 텍스트에서 '## 답변' 헤더 제거"""
        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        answer_text = re.sub(r'^##\s*답변\s*\n+', '', answer_text, flags=re.MULTILINE | re.IGNORECASE)
        answer_text = answer_text.lstrip('\n')

        return answer_text
    
    def remove_duplicate_headers(self, answer_text: str) -> str:
        """중복 헤더 제거"""
        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        lines = answer_text.split('\n')
        result_lines = []
        seen_headers = set()
        skip_next_empty = False

        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,3})\s+(.+)', line)
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                clean_header = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨🎉💯🔔]+\s*', '', header_text).strip()
                normalized_header = re.sub(r'\s+', ' ', clean_header.lower())
                header_key = f"{level}:{normalized_header}"

                if normalized_header in ["답변", "answer", "답"]:
                    if "답변" in seen_headers or "answer" in seen_headers:
                        skip_next_empty = True
                        continue

                if header_key in seen_headers:
                    skip_next_empty = True
                    continue

                seen_headers.add(normalized_header)
                seen_headers.add(header_key)
                skip_next_empty = False
            elif skip_next_empty and line.strip() == "":
                continue
            else:
                skip_next_empty = False

            result_lines.append(line)

        answer_text = '\n'.join(result_lines)

        lines = answer_text.split('\n')
        cleaned_lines = []
        seen_answer_header = False
        i = 0

        while i < len(lines):
            line = lines[i]
            if re.match(r'^##\s*답변\s*$', line, re.IGNORECASE):
                if not seen_answer_header:
                    cleaned_lines.append(line)
                    seen_answer_header = True
                if i + 1 < len(lines) and re.match(r'^###\s*.*답변', lines[i + 1], re.IGNORECASE):
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif re.match(r'^###\s*.*답변', line, re.IGNORECASE):
                i += 1
                continue
            else:
                cleaned_lines.append(line)
                i += 1

        answer_text = '\n'.join(cleaned_lines)
        answer_text = re.sub(
            r'(##\s*답변\s*\n+)(###\s*.*답변\s*\n+)',
            r'\1',
            answer_text,
            flags=re.MULTILINE | re.IGNORECASE
        )
        answer_text = re.sub(
            r'##\s*답변\s*\n+\s*##\s*답변',
            '## 답변',
            answer_text,
            flags=re.IGNORECASE | re.MULTILINE
        )

        return answer_text
    
    def remove_intermediate_text(self, answer_text: str) -> str:
        """중간 생성 텍스트 제거"""
        if not answer_text or not isinstance(answer_text, str):
            return answer_text

        lines = answer_text.split('\n')
        cleaned_lines = []
        skip_section = False

        skip_patterns = [
            r'^##\s*STEP\s*0',
            r'^##\s*원본\s*품질\s*평가',
            r'^##\s*질문\s*정보',
            r'^##\s*원본\s*답변',
            r'^\*\*질문\*\*:',
            r'^\*\*질문\s*유형\*\*:',
            r'^평가\s*결과',
            r'원본\s*에\s*개선이\s*필요하면',
            r'^\*\*평가\s*결\s*과\s*에\s*따른\s*작업',
        ]

        for i, line in enumerate(lines):
            is_section_start = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip_section = True
                    is_section_start = True
                    break

            if is_section_start:
                continue

            if skip_section:
                if re.match(r'^##\s+[가-힣]', line):
                    skip_section = False
                    if not any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                        cleaned_lines.append(line)
                    continue
                
                if re.search(r'\[문서:|\[법령:|민법\s*제\d+조|형법\s*제\d+조', line):
                    skip_section = False
                    cleaned_lines.append(line)
                    continue

                if re.match(r'^\s*[•\-\*]\s*\[.*?\].*?', line):
                    continue

                if re.match(r'^안녕하세요.*?궁금하시군요\.?\s*$', line, re.IGNORECASE):
                    continue
                
                if line.strip() == "" and i > 0 and lines[i-1].strip() == "":
                    if i + 1 < len(lines) and lines[i+1].strip() and not any(re.match(p, lines[i+1], re.IGNORECASE) for p in skip_patterns):
                        skip_section = False
                        cleaned_lines.append(line)
                        continue

                continue
            else:
                if re.match(r'^\s*[•\-\*]\s*\[.*?\].*?', line):
                    continue

                if re.search(r'\[.*?\].*?(충분|명확|일관|포함)', line):
                    continue

                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

