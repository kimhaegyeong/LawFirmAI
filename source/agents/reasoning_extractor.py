# -*- coding: utf-8 -*-
"""
추론 과정 분리 및 검증 모듈
LLM 응답에서 추론 과정(Chain-of-Thought)을 추출하고 실제 답변만 분리
"""

import logging
import re
from typing import Any, Dict, Optional

from source.agents.workflow_constants import AnswerExtractionPatterns


class ReasoningExtractor:
    """
    추론 과정 분리 및 검증 클래스

    LLM 응답에서 추론 과정과 실제 답변을 분리하고,
    품질을 검증하는 기능을 제공합니다.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        ReasoningExtractor 초기화

        Args:
            logger: 로거 인스턴스 (없으면 자동 생성)
        """
        self.logger = logger or logging.getLogger(__name__)
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """정규식 패턴을 컴파일하여 캐싱 (성능 최적화)"""
        # 추론 과정 섹션 패턴 컴파일
        self._compiled_reasoning_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.REASONING_SECTION_PATTERNS
        ]

        # 출력 섹션 패턴 컴파일
        self._compiled_output_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.OUTPUT_SECTION_PATTERNS
        ]

        # 답변 섹션 패턴 컴파일
        self._compiled_answer_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in AnswerExtractionPatterns.ANSWER_SECTION_PATTERNS
        ]

        # Step 헤더 패턴 컴파일 (Step 1, 2, 3)
        self._compiled_step_header_patterns = {
            "step1": [
                re.compile(r'###\s*Step\s*1[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*단계\s*1[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*1\s*[:：]', re.IGNORECASE | re.MULTILINE),
            ],
            "step2": [
                re.compile(r'###\s*Step\s*2[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*단계\s*2[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*2\s*[:：]', re.IGNORECASE | re.MULTILINE),
            ],
            "step3": [
                re.compile(r'###\s*Step\s*3[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*단계\s*3[:：]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'###\s*Step\s*3\s*[:：]', re.IGNORECASE | re.MULTILINE),
            ],
        }

        # 다음 마커 패턴 컴파일
        self._compiled_next_marker_patterns = {
            "step1": [
                re.compile(r'\n\s*###\s*Step\s*2', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*###\s*단계\s*2', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*📤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*출력', re.IGNORECASE | re.MULTILINE),
            ],
            "step2": [
                re.compile(r'\n\s*###\s*Step\s*3', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*###\s*단계\s*3', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*📤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*출력', re.IGNORECASE | re.MULTILINE),
            ],
            "step3": [
                re.compile(r'\n\s*##\s*[^#]', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*📤', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\n\s*##\s*출력', re.IGNORECASE | re.MULTILINE),
            ],
        }

        # 부분 정리 패턴 컴파일
        self._compiled_partial_cleaning_patterns = [
            re.compile(r'##\s*🧠\s*추론[^\n]*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'###\s*Step\s*[123][^\n]*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'###\s*단계\s*[123][^\n]*', re.IGNORECASE | re.MULTILINE),
        ]

        # 다음 섹션 패턴 컴파일 (면책조항, 참고자료 등)
        self._compiled_next_section_patterns = [
            re.compile(r'###?\s*📚', re.IGNORECASE),
            re.compile(r'###?\s*참고자료', re.IGNORECASE),
            re.compile(r'###?\s*💡', re.IGNORECASE),
            re.compile(r'###?\s*신뢰도', re.IGNORECASE),
            re.compile(r'---'),
            re.compile(r'💼\s*면책', re.IGNORECASE),
        ]

    def extract_reasoning(self, answer: str) -> Dict[str, Any]:
        """
        LLM 답변에서 추론 과정(Chain-of-Thought) 섹션을 추출

        Args:
            answer: LLM 원본 답변 문자열

        Returns:
            Dict with keys:
                - "reasoning": 추론 과정 전체 내용
                - "step1": Step 1 내용
                - "step2": Step 2 내용
                - "step3": Step 3 내용
                - "has_reasoning": 추론 과정 존재 여부
                - "reasoning_section_count": 추론 섹션 개수
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

        # 성능 최적화: 대용량 응답에 대한 조기 종료 조건
        answer_length = len(answer)
        MAX_REASONING_SEARCH_LENGTH = 10000  # 10KB 이상인 경우 성능 최적화 적용

        if answer_length > MAX_REASONING_SEARCH_LENGTH:
            # 대용량 응답의 경우 처음 10KB만 검색 (추론 과정은 보통 앞부분에 있음)
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

        # 추론 과정 섹션 찾기 (여러 곳에 분산된 경우도 처리)
        reasoning_sections = []  # 모든 추론 섹션을 저장

        # 모든 추론 섹션 찾기 (search_text 사용)
        reasoning_found = False
        for compiled_pattern in self._compiled_reasoning_patterns:
            # 모든 매칭 찾기 (첫 번째만이 아니라)
            for match in compiled_pattern.finditer(search_text):
                reasoning_found = True
                reasoning_start_idx = match.start()
                # 다음 섹션 찾기 (출력 섹션 또는 답변 섹션) - search_text 기준
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
                    # 다음 섹션을 찾지 못한 경우, 추론 섹션 끝까지 전체 포함
                    # search_text 기준이므로 원본 answer 길이로 조정
                    if answer_length > MAX_REASONING_SEARCH_LENGTH and reasoning_start_idx < MAX_REASONING_SEARCH_LENGTH:
                        # 대용량 응답에서 검색 범위를 벗어난 경우 원본 끝까지
                        reasoning_end_idx = answer_length
                    else:
                        reasoning_end_idx = answer_length if reasoning_start_idx < answer_length else len(search_text)

                # 중복 체크 (이미 포함된 섹션인지 확인)
                is_duplicate = False
                for existing_start, existing_end in reasoning_sections:
                    if reasoning_start_idx >= existing_start and reasoning_start_idx < existing_end:
                        is_duplicate = True
                        break
                    if existing_start >= reasoning_start_idx and existing_start < reasoning_end_idx:
                        # 기존 섹션이 현재 섹션에 포함되어 있으면 확장
                        reasoning_sections.remove((existing_start, existing_end))
                        break

                if not is_duplicate:
                    reasoning_sections.append((reasoning_start_idx, reasoning_end_idx))

        # 여러 추론 섹션이 있는 경우 병합
        if reasoning_sections:
            # 시작 위치 순으로 정렬
            reasoning_sections.sort(key=lambda x: x[0])

            # 겹치는 섹션 병합
            merged_sections = []
            current_start, current_end = reasoning_sections[0]

            for start_idx, end_idx in reasoning_sections[1:]:
                if start_idx <= current_end:
                    # 겹치는 경우 병합
                    current_end = max(current_end, end_idx)
                else:
                    # 겹치지 않는 경우 저장하고 새로 시작
                    merged_sections.append((current_start, current_end))
                    current_start, current_end = start_idx, end_idx

            merged_sections.append((current_start, current_end))

            # 병합된 섹션들을 하나의 텍스트로 합치기
            reasoning_text_parts = []
            for start_idx, end_idx in merged_sections:
                # 원본 answer에서 추출 (인덱스는 search_text 기준이지만 answer에서 가져옴)
                if start_idx < answer_length:
                    end_idx = min(end_idx, answer_length)
                    section_text = answer[start_idx:end_idx].strip()
                    if section_text:
                        reasoning_text_parts.append(section_text)

            reasoning_text = "\n\n".join(reasoning_text_parts)
            result["reasoning"] = reasoning_text
            result["has_reasoning"] = True
            result["reasoning_section_count"] = len(merged_sections)  # 섹션 개수 저장

            # Step 1, 2, 3 추출 (Step 헤더를 직접 찾아서 그 사이의 내용 추출)
            # 여러 섹션에서 Step 추출 후 병합
            step_keys = ["step1", "step2", "step3"]

            # 성능 최적화: reasoning_text가 너무 긴 경우 조기 종료
            MAX_STEP_SEARCH_LENGTH = 5000  # Step 추출도 5KB 제한
            step_search_text = reasoning_text[:MAX_STEP_SEARCH_LENGTH] if len(reasoning_text) > MAX_STEP_SEARCH_LENGTH else reasoning_text

            # 각 Step에 대해 모든 섹션에서 추출하여 병합
            for step_key in step_keys:
                step_contents = []  # 여러 섹션에서 추출한 Step 내용 저장

                # 병합된 추론 텍스트에서 해당 Step 찾기 (step_search_text 사용)
                compiled_headers = self._compiled_step_header_patterns[step_key]
                for compiled_header in compiled_headers:
                    # 모든 Step 헤더 찾기 (여러 섹션에 있을 수 있음)
                    full_header_pattern = compiled_header.pattern + r'[^\n]*\n'
                    full_header_compiled = re.compile(full_header_pattern, re.IGNORECASE | re.MULTILINE)

                    # 모든 매칭 찾기 (step_search_text 사용)
                    for match in full_header_compiled.finditer(step_search_text):
                        step_start = match.end()

                        # 다음 마커 찾기 (컴파일된 패턴 사용) - step_search_text 기준
                        remaining_text = step_search_text[step_start:]
                        compiled_markers = self._compiled_next_marker_patterns[step_key]
                        step_end = None

                        for compiled_marker in compiled_markers:
                            next_match = compiled_marker.search(remaining_text)
                            if next_match:
                                step_end = step_start + next_match.start()
                                break

                        if step_end is None:
                            # 다음 마커를 찾지 못한 경우 끝까지 (reasoning_text 기준으로 조정)
                            step_end = min(step_start + len(remaining_text), len(reasoning_text))

                        # Step 내용 추출 (헤더 줄 제외, 빈 줄 제거) - reasoning_text에서
                        step_content = reasoning_text[step_start:step_end].strip() if step_start < len(reasoning_text) else ""
                        if step_content:
                            step_contents.append(step_content)

                    # 첫 번째 헤더 패턴에서 찾으면 다음 패턴으로 넘어가지 않음
                    if step_contents:
                        break

                # 여러 섹션에서 추출한 Step 내용 병합 (중복 제거 및 정렬)
                if step_contents:
                    # 중복 제거 (동일한 내용은 한 번만)
                    unique_contents = []
                    seen_contents = set()
                    for content in step_contents:
                        # 정규화하여 비교 (공백, 줄바꿈 정리)
                        normalized = re.sub(r'\s+', ' ', content.strip())
                        if normalized and normalized not in seen_contents:
                            seen_contents.add(normalized)
                            unique_contents.append(content)

                    # 병합 (빈 줄로 구분)
                    if unique_contents:
                        merged_step_content = "\n\n".join(unique_contents)
                        result[step_key] = merged_step_content
                elif step_key == "step3":
                    # Step 3의 경우 내용이 비어있을 수도 있지만 헤더는 존재하므로 저장
                    result[step_key] = ""

        return result

    def extract_actual_answer(self, llm_response: str) -> str:
        """
        LLM 응답에서 실제 답변만 추출 (추론 과정 제외)

        우선순위:
        1. "## 📤 출력" 섹션
        2. "## 답변" 섹션 (추론 과정 제외)
        3. 전체 내용에서 추론 과정 제거 후 남은 부분

        Args:
            llm_response: LLM 원본 응답 문자열

        Returns:
            실제 답변만 포함한 문자열
        """
        if not llm_response or not isinstance(llm_response, str):
            return ""

        # 1단계: 출력 섹션 찾기 (최우선, 컴파일된 패턴 사용)
        for compiled_pattern in self._compiled_output_patterns:
            match = compiled_pattern.search(llm_response)
            if match:
                # 출력 섹션 이후의 모든 내용
                output_start = match.end()
                remaining = llm_response[output_start:].strip()

                # 다음 섹션(면책조항, 참고자료 등) 전까지 추출 (컴파일된 패턴 사용)
                next_section_idx = len(remaining)
                for compiled_next_pattern in self._compiled_next_section_patterns:
                    next_match = compiled_next_pattern.search(remaining)
                    if next_match:
                        next_section_idx = min(next_section_idx, next_match.start())

                answer = remaining[:next_section_idx].strip()
                if answer:
                    return answer

        # 2단계: 답변 섹션 찾기 (추론 과정 제외)
        reasoning_info = self.extract_reasoning(llm_response)
        if reasoning_info["has_reasoning"]:
            # 추론 과정 제거
            reasoning_start = llm_response.find(reasoning_info["reasoning"])
            if reasoning_start != -1:
                # 추론 과정 이후 부분 추출
                after_reasoning_start = reasoning_start + len(reasoning_info["reasoning"])
                after_reasoning = llm_response[after_reasoning_start:].strip()

                # 답변 섹션이 추론 과정 이후에 있는지 확인 (컴파일된 패턴 사용)
                for compiled_pattern in self._compiled_answer_patterns:
                    match = compiled_pattern.search(after_reasoning)
                    if match:
                        answer_start = match.end()
                        answer = after_reasoning[answer_start:].strip()
                        # 다음 섹션 전까지 (컴파일된 패턴 사용)
                        next_section_idx = len(answer)
                        for compiled_next_pattern in self._compiled_next_section_patterns:
                            next_match = compiled_next_pattern.search(answer)
                            if next_match:
                                next_section_idx = min(next_section_idx, next_match.start())

                        answer = answer[:next_section_idx].strip()
                        if answer:
                            return answer

        # 3단계: 전체 내용에서 추론 과정 제거 (모든 섹션 완전 제거)
        if reasoning_info["has_reasoning"]:
            reasoning_text = reasoning_info["reasoning"]
            reasoning_section_count = reasoning_info.get("reasoning_section_count", 1)

            # 여러 섹션이 있는 경우 각 섹션을 개별적으로 제거
            if reasoning_section_count > 1:
                # 원본에서 모든 추론 섹션의 인덱스를 찾아서 저장
                reasoning_section_indices = []
                for compiled_pattern in self._compiled_reasoning_patterns:
                    # 모든 추론 섹션 찾기
                    for match in compiled_pattern.finditer(llm_response):
                        reasoning_start = match.start()
                        # 다음 섹션 찾기 (출력 섹션 또는 답변 섹션)
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

                        # 중복 체크
                        is_duplicate = False
                        for existing_start, existing_end in reasoning_section_indices:
                            if reasoning_start >= existing_start and reasoning_start < existing_end:
                                is_duplicate = True
                                break
                            if existing_start >= reasoning_start and existing_start < reasoning_end:
                                # 기존 섹션이 현재 섹션에 포함되어 있으면 교체
                                reasoning_section_indices.remove((existing_start, existing_end))
                                break

                        if not is_duplicate and reasoning_end > reasoning_start:
                            reasoning_section_indices.append((reasoning_start, reasoning_end))

                # 역순으로 정렬하여 제거 (뒤에서부터 제거하면 인덱스 변화 없음)
                reasoning_section_indices.sort(key=lambda x: x[0], reverse=True)

                # 각 섹션을 역순으로 제거
                cleaned_response = llm_response
                for reasoning_start, reasoning_end in reasoning_section_indices:
                    cleaned_response = (
                        cleaned_response[:reasoning_start] +
                        cleaned_response[reasoning_end:]
                    )

                cleaned_response = cleaned_response.strip()
            else:
                # 단일 섹션인 경우 기존 방식 사용
                cleaned_response = llm_response.replace(reasoning_text, "").strip()

            # 답변 섹션 찾기 (컴파일된 패턴 사용)
            for compiled_pattern in self._compiled_answer_patterns:
                match = compiled_pattern.search(cleaned_response)
                if match:
                    answer_start = match.end()
                    answer = cleaned_response[answer_start:].strip()
                    # 다음 섹션 전까지 (컴파일된 패턴 사용)
                    next_section_idx = len(answer)
                    for compiled_next_pattern in self._compiled_next_section_patterns:
                        next_match = compiled_next_pattern.search(answer)
                        if next_match:
                            next_section_idx = min(next_section_idx, next_match.start())

                    answer = answer[:next_section_idx].strip()
                    if answer:
                        return answer

            # 답변 섹션이 없으면 추론 과정 제거 후 남은 내용 반환
            if cleaned_response:
                return cleaned_response

        # 4단계: 추론 과정이 없거나 찾지 못한 경우 부분 분리 시도 (컴파일된 패턴 사용)
        # 추론 과정 섹션 키워드가 있는 경우 부분적으로 제거 시도
        cleaned_response = llm_response
        for compiled_pattern in self._compiled_partial_cleaning_patterns:
            cleaned_response = compiled_pattern.sub('', cleaned_response)

        # 빈 줄 정리
        cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)

        # 변경이 있었으면 반환
        if cleaned_response.strip() != llm_response.strip():
            return cleaned_response.strip()

        # 모든 방법 실패 시 원본 반환
        return llm_response

    def extract_by_output_section(self, llm_response: str) -> str:
        """
        출력 섹션에서 실제 답변 추출 (재시도 로직용 helper)

        Args:
            llm_response: LLM 원본 응답

        Returns:
            추출된 답변 또는 빈 문자열
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
        추론 과정 제거 후 실제 답변 추출 (재시도 로직용 helper)

        Args:
            llm_response: LLM 원본 응답
            reasoning_info: 추출된 추론 과정 정보

        Returns:
            추출된 답변 또는 빈 문자열
        """
        if not reasoning_info.get("has_reasoning"):
            return ""

        reasoning_text = reasoning_info.get("reasoning", "")
        if not reasoning_text:
            return ""

        reasoning_section_count = reasoning_info.get("reasoning_section_count", 1)

        # 여러 섹션이 있는 경우 각 섹션을 개별적으로 제거
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

        # 답변 섹션 찾기
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

        # 답변 섹션이 없으면 추론 과정 제거 후 남은 내용 반환
        if cleaned_response:
            return cleaned_response

        return ""

    def extract_by_partial_cleaning(self, llm_response: str) -> str:
        """
        부분 정리 방법으로 추출 (재시도 로직용 helper)

        Args:
            llm_response: LLM 원본 응답

        Returns:
            추출된 답변 또는 빈 문자열
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
        실제 답변에서 추론 과정 키워드 잔여 확인 및 정리

        Args:
            answer: 추출된 실제 답변

        Returns:
            추론 과정 키워드가 정리된 답변
        """
        if not answer or not isinstance(answer, str):
            return answer

        # 추론 과정 관련 키워드 패턴
        reasoning_keywords = [
            r'##\s*🧠\s*추론',
            r'###\s*Step\s*[123]',
            r'###\s*단계\s*[123]',
            r'Chain-of-Thought',
            r'CoT',
            r'추론과정작성',
        ]

        cleaned_answer = answer
        found_keywords = []

        # 키워드 검사 및 제거
        for keyword_pattern in reasoning_keywords:
            compiled_pattern = re.compile(keyword_pattern, re.IGNORECASE | re.MULTILINE)
            matches = list(compiled_pattern.finditer(cleaned_answer))
            if matches:
                found_keywords.append(keyword_pattern)
                # 키워드가 포함된 줄 전체 제거
                for match in reversed(matches):  # 역순으로 제거하여 인덱스 변화 방지
                    start_pos = match.start()
                    # 줄 시작 찾기
                    line_start = cleaned_answer.rfind('\n', 0, start_pos) + 1
                    # 줄 끝 찾기
                    line_end = cleaned_answer.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(cleaned_answer)

                    # 해당 줄 제거
                    cleaned_answer = cleaned_answer[:line_start] + cleaned_answer[line_end + 1:]
                    cleaned_answer = cleaned_answer.lstrip()  # 시작 부분 빈 줄 제거

        # 빈 줄 정리 (연속된 빈 줄을 하나로)
        cleaned_answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_answer)

        # 키워드 발견 시 로깅
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
        추론 과정 분리 후 품질 검증

        Args:
            original_answer: 원본 답변
            actual_answer: 추론 과정 제거된 실제 답변
            reasoning_info: 추출된 추론 과정 정보

        Returns:
            Dict: 품질 메트릭
                - "is_valid": 유효 여부
                - "warnings": 경고 목록
                - "errors": 에러 목록
                - "score": 품질 점수 (0.0-1.0)
        """
        quality = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "score": 1.0,
        }

        if not original_answer or not actual_answer:
            quality["is_valid"] = False
            quality["errors"].append("원본 답변 또는 실제 답변이 비어있음")
            quality["score"] = 0.0
            return quality

        # 1. 길이 검증
        if len(actual_answer) == 0:
            quality["is_valid"] = False
            quality["errors"].append("실제 답변이 비어있음")
            quality["score"] = 0.0
            return quality

        # 2. 추론 과정이 실제 답변에 포함되지 않았는지 확인
        if reasoning_info.get("has_reasoning"):
            reasoning_text = reasoning_info.get("reasoning", "")
            if reasoning_text and reasoning_text in actual_answer:
                quality["is_valid"] = False
                quality["errors"].append("추론 과정이 실제 답변에 포함되어 있음")
                quality["score"] = 0.0
            elif any(keyword in actual_answer for keyword in ["🧠", "추론과정", "Step 1", "Step 2", "Step 3"]):
                quality["warnings"].append("추론 과정 키워드가 실제 답변에 남아있을 수 있음")
                quality["score"] = 0.8

        # 3. 답변 길이 검증 (너무 짧으면 문제)
        extraction_ratio = len(actual_answer) / len(original_answer) if original_answer else 0.0
        if extraction_ratio < 0.1:
            quality["warnings"].append(f"실제 답변이 너무 짧음 (비율: {extraction_ratio:.1%})")
            quality["score"] = min(quality["score"], 0.7)
        elif extraction_ratio > 0.95:
            quality["warnings"].append("추론 과정이 거의 제거되지 않았을 수 있음")
            quality["score"] = min(quality["score"], 0.9)

        # 4. 키워드 포함 여부 검증 (법률 관련 키워드가 실제 답변에 있는지)
        legal_keywords = ["법", "조문", "판례", "법률", "민법", "형법", "상법"]
        original_keywords = [kw for kw in legal_keywords if kw in original_answer]
        actual_keywords = [kw for kw in legal_keywords if kw in actual_answer]

        if original_keywords and len(actual_keywords) < len(original_keywords) * 0.5:
            quality["warnings"].append("법률 키워드가 많이 제거되었을 수 있음")
            quality["score"] = min(quality["score"], 0.8)

        # 5. 품질 점수 계산
        if quality["errors"]:
            quality["score"] = 0.0
        elif quality["warnings"]:
            quality["score"] = quality["score"] * 0.9

        return quality
