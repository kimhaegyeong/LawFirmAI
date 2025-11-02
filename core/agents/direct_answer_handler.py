# -*- coding: utf-8 -*-
"""
직접 답변 핸들러 모듈
검색 없이 간단한 질문에 직접 답변하는 로직을 독립 모듈로 분리
"""

import logging
from typing import Any, Dict, Optional

from core.agents.prompt_chain_executor import PromptChainExecutor
from core.agents.response_parsers import AnswerParser, ClassificationParser
from core.agents.workflow_utils import WorkflowUtils


class DirectAnswerHandler:
    """
    직접 답변 생성 클래스

    간단한 질문(인사말, 용어 정의 등)에 대해 검색 없이 직접 답변을 생성합니다.
    Prompt Chaining을 사용하여 질문 유형 분석 → 프롬프트 생성 → 답변 생성 → 품질 검증 → 개선을 수행합니다.
    """

    def __init__(
        self,
        llm: Any,
        llm_fast: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        DirectAnswerHandler 초기화

        Args:
            llm: 메인 LLM 인스턴스
            llm_fast: 빠른 LLM 인스턴스 (간단한 질문용)
            logger: 로거 (없으면 자동 생성)
        """
        self.llm = llm
        self.llm_fast = llm_fast
        self.logger = logger or logging.getLogger(__name__)

    def generate_direct_answer_with_chain(self, query: str) -> Optional[str]:
        """
        Prompt Chaining을 사용한 직접 답변 생성 (다단계 체인)

        Step 1: 질문 유형 분석 (인사말 vs 용어 정의)
        Step 2: 적절한 프롬프트 생성
        Step 3: 초기 답변 생성
        Step 4: 답변 품질 검증
        Step 5: 필요시 답변 개선

        Args:
            query: 사용자 질문

        Returns:
            Optional[str]: 생성된 답변 또는 None
        """
        try:
            # 빠른 모델 사용
            llm = self.llm_fast if self.llm_fast else self.llm

            # PromptChainExecutor 인스턴스 생성
            chain_executor = PromptChainExecutor(llm, self.logger)

            # 체인 스텝 정의
            chain_steps = []

            # Step 1: 질문 유형 분석
            def build_query_type_analysis_prompt(prev_output, initial_input):
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                return f"""다음 질문의 유형을 분석해주세요.

질문: {query_value}

다음 유형 중 하나를 선택하세요:
- greeting (인사말): "안녕하세요", "고마워요", "감사합니다" 등
- term_definition (용어 정의): 법률 용어나 개념의 정의를 묻는 질문
- simple_question (간단한 질문): 일반 법률 상식으로 답변 가능한 간단한 질문

다음 형식으로 응답해주세요:
{{
    "query_type": "greeting" | "term_definition" | "simple_question",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "query_type_analysis",
                "prompt_builder": build_query_type_analysis_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: ClassificationParser.parse_query_type_analysis_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "query_type" in output,
                "required": True
            })

            # Step 2: 적절한 프롬프트 생성
            def build_prompt_generation_prompt(prev_output, initial_input):
                # prev_output은 Step 1의 결과 (query_type 포함)
                if not isinstance(prev_output, dict):
                    prev_output = {}

                query_type_value = prev_output.get("query_type", "simple_question")
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                if query_type_value == "greeting":
                    return f"""사용자의 인사에 친절하게 응답하세요:

{query_value}

간단하고 친절하게 응답해주세요. (1-2문장)"""
                elif query_type_value == "term_definition":
                    return f"""다음 법률 용어에 대해 간단명료하게 정의를 제공하세요:

용어: {query_value}

다음 형식을 따라주세요:
1. 용어의 정의 (1-2문장)
2. 간단한 설명 (1문장)
총 2-3문장으로 간결하게 작성해주세요."""
                else:
                    # simple_question
                    return f"""다음 법률 질문에 간단명료하게 답하세요:

질문: {query_value}

법률 용어나 개념에 대한 정의나 간단한 설명을 제공하세요. 검색 없이 일반적인 법률 지식으로 답변하세요. (2-4문장)"""

            chain_steps.append({
                "name": "prompt_generation",
                "prompt_builder": build_prompt_generation_prompt,
                "input_extractor": lambda prev: prev,  # Step 1의 출력 사용
                "output_parser": lambda response, prev: response,  # 프롬프트 문자열 그대로 반환
                "validator": lambda output: output and len(str(output).strip()) > 10,
                "required": True
            })

            # Step 3: 초기 답변 생성
            def build_initial_answer_prompt(prev_output, initial_input):
                # prev_output은 Step 2의 결과 (프롬프트 문자열)
                if isinstance(prev_output, str):
                    return prev_output
                elif isinstance(prev_output, dict):
                    return prev_output.get("prompt", "")
                return ""

            chain_steps.append({
                "name": "initial_answer_generation",
                "prompt_builder": build_initial_answer_prompt,
                "input_extractor": lambda prev: prev,  # Step 2의 출력 사용
                "output_parser": lambda response, prev: WorkflowUtils.normalize_answer(response),
                "validator": lambda output: output and len(output.strip()) > 10,
                "required": True
            })

            # Step 4: 답변 품질 검증
            def build_quality_validation_prompt(prev_output, initial_input):
                # prev_output은 Step 3의 결과 (답변 문자열)
                answer = prev_output if isinstance(prev_output, str) else str(prev_output)
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                return f"""다음 답변의 품질을 검증해주세요.

질문: {query_value}
답변: {answer[:500]}

다음 기준으로 검증하세요:
1. **적절한 길이**: 너무 짧지도 길지도 않음 (10-500자)
2. **질문에 대한 직접적인 답변**: 질문에 맞는 답변인가?
3. **명확성**: 답변이 명확하고 이해하기 쉬운가?
4. **완성도**: 답변이 완전한가?

다음 형식으로 응답해주세요:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "needs_improvement": true | false
}}
"""

            chain_steps.append({
                "name": "quality_validation",
                "prompt_builder": build_quality_validation_prompt,
                "input_extractor": lambda prev: prev,  # Step 3의 출력 사용
                "output_parser": lambda response, prev: AnswerParser.parse_quality_validation_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "is_valid" in output,
                "required": False,  # 선택 단계
            })

            # Step 5: 필요시 답변 개선
            def build_answer_improvement_prompt(prev_output, initial_input):
                # prev_output은 Step 4의 결과 (validation 결과)
                if not isinstance(prev_output, dict):
                    return None

                needs_improvement = prev_output.get("needs_improvement", False)
                if not needs_improvement:
                    return None  # 개선 불필요

                # Step 3의 답변 가져오기
                original_answer = ""
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                # 체인 히스토리에서 초기 답변 찾기
                if hasattr(chain_executor, 'chain_history'):
                    for step in chain_executor.chain_history:
                        if step.get("step_name") == "initial_answer_generation" and step.get("success"):
                            original_answer = step.get("output", "")
                            break

                if not original_answer:
                    return None

                issues = prev_output.get("issues", [])

                return f"""다음 답변을 개선해주세요.

질문: {query_value}
원본 답변: {original_answer}
문제점: {', '.join(issues) if issues else "없음"}

다음 문제점을 해결하여 개선된 답변을 작성해주세요:
{chr(10).join([f"- {issue}" for issue in issues[:3]]) if issues else "없음"}
"""

            chain_steps.append({
                "name": "answer_improvement",
                "prompt_builder": build_answer_improvement_prompt,
                "input_extractor": lambda prev: prev,  # Step 4의 출력 사용
                "output_parser": lambda response, prev: WorkflowUtils.normalize_answer(response),
                "validator": lambda output: output and len(output.strip()) > 10,
                "required": False,  # 선택 단계 (개선 필요시에만)
                "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("needs_improvement", False)
            })

            # 체인 실행
            initial_input_dict = {"query": query}

            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=2,
                stop_on_failure=False
            )

            # 결과 추출
            chain_history = chain_result.get("chain_history", [])

            # Step 3 결과: 초기 답변
            answer = None
            for step in chain_history:
                if step.get("step_name") == "initial_answer_generation" and step.get("success"):
                    answer = step.get("output", "")
                    break

            # Step 5 결과: 개선된 답변 (있는 경우 우선)
            improved_answer = None
            for step in chain_history:
                if step.get("step_name") == "answer_improvement" and step.get("success"):
                    improved_answer = step.get("output", "")
                    break

            # 최종 답변 선택 (개선된 답변 우선)
            final_answer = improved_answer if improved_answer else answer

            # 체인 실행 결과 로깅
            chain_summary = chain_executor.get_chain_summary()
            self.logger.info(
                f"✅ [DIRECT ANSWER CHAIN] Executed {chain_summary['total_steps']} steps, "
                f"{chain_summary['successful_steps']} successful, "
                f"Total time: {chain_summary['total_time']:.2f}s"
            )

            return final_answer if final_answer and len(final_answer.strip()) >= 10 else None

        except Exception as e:
            self.logger.error(f"❌ [DIRECT ANSWER CHAIN ERROR] Prompt chain failed: {e}")
            return None

    def generate_fallback_answer(self, query: str) -> Optional[str]:
        """
        체인 실패 시 폴백 직접 답변 생성

        Args:
            query: 사용자 질문

        Returns:
            Optional[str]: 생성된 답변 또는 None
        """
        try:
            llm = self.llm_fast if self.llm_fast else self.llm

            # 간단한 프롬프트 생성
            if any(word in query.lower() for word in ["안녕", "고마워", "감사"]):
                prompt = f"""사용자의 인사에 친절하게 응답하세요:

{query}

간단하고 친절하게 응답해주세요."""
            else:
                # 용어 정의 질문
                prompt = f"""다음 법률 질문에 간단명료하게 답하세요:

질문: {query}

법률 용어나 개념에 대한 정의나 간단한 설명을 제공하세요. 검색 없이 일반적인 법률 지식으로 답변하세요."""

            response = llm.invoke(prompt)
            answer = WorkflowUtils.extract_response_content(response)

            # 최소 길이 체크
            if answer and len(answer.strip()) >= 10:
                return answer

            return None

        except Exception as e:
            self.logger.error(f"Fallback LLM invocation failed: {e}")
            return None
