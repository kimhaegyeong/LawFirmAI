# -*- coding: utf-8 -*-
"""
답변 생성 모듈
LLM을 사용한 답변 생성 및 품질 개선 로직을 독립 모듈로 분리
"""

import logging
import time
from typing import Any, Dict, List, Optional

from source.utils.prompt_chain_executor import PromptChainExecutor
from source.data.quality_validators import AnswerValidator
from source.data.response_parsers import AnswerParser
from source.utils.state_definitions import LegalWorkflowState
from source.utils.workflow_constants import WorkflowConstants
from source.utils.workflow_utils import WorkflowUtils


class AnswerGenerator:
    """
    답변 생성 및 개선 클래스

    LLM을 사용하여 답변을 생성하고, Prompt Chaining을 통해 품질을 개선합니다.
    """

    def __init__(
        self,
        llm: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        AnswerGenerator 초기화

        Args:
            llm: LangChain LLM 인스턴스
            logger: 로거 인스턴스 (없으면 자동 생성)
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def generate_answer_with_chain(
        self,
        optimized_prompt: str,
        query: str,
        context_dict: Dict[str, Any],
        quality_feedback: Optional[Dict[str, Any]] = None,
        is_retry: bool = False
    ) -> str:
        """
        Prompt Chaining을 사용한 답변 생성 및 개선

        Step 1: 초기 답변 생성
        Step 2: 답변 검증 및 문제점 추출
        Step 3: 개선 지시 생성 (문제점이 있는 경우)
        Step 4: 개선된 답변 생성 (Step 3가 있는 경우)
        Step 5: 최종 검증

        Args:
            optimized_prompt: 최적화된 프롬프트
            query: 원본 질문
            context_dict: 컨텍스트 정보
            quality_feedback: 품질 피드백 (재시도 시 사용)
            is_retry: 재시도 여부

        Returns:
            생성된 답변 문자열
        """
        try:
            # PromptChainExecutor 인스턴스 생성
            chain_executor = PromptChainExecutor(self.llm, self.logger)

            # 체인 스텝 정의
            chain_steps = []

            # Step 1: 초기 답변 생성
            def build_initial_answer_prompt(prev_output, initial_input):
                # 초기 입력에서 프롬프트 추출
                if isinstance(initial_input, dict):
                    return initial_input.get("prompt", optimized_prompt)
                return optimized_prompt

            chain_steps.append({
                "name": "initial_answer_generation",
                "prompt_builder": build_initial_answer_prompt,
                "input_extractor": lambda prev: prev,  # 초기 입력을 그대로 전달
                "output_parser": lambda response, prev: WorkflowUtils.normalize_answer(response),
                "validator": lambda output: output and len(output.strip()) > 10,
                "required": True
            })

            # Step 2: 답변 검증 및 문제점 추출
            def build_validation_prompt(prev_output, initial_input):
                # prev_output이 딕셔너리인 경우 (Step 1의 출력이 딕셔너리로 변환된 경우) 처리
                if isinstance(prev_output, dict):
                    answer = prev_output.get("answer") or prev_output.get("content") or str(prev_output)
                else:
                    answer = prev_output if isinstance(prev_output, str) else str(prev_output)
                validation_criteria = """
다음 기준으로 답변을 검증하세요:

1. **길이**: 최소 50자 이상
2. **내용 완성도**: 질문에 대한 직접적인 답변 포함
3. **법적 근거**: 관련 법령, 조항, 판례 인용 여부
4. **구조**: 명확한 섹션과 논리적 흐름
5. **일관성**: 답변 전체의 논리적 일관성

답변:
{answer}

다음 형식으로 검증 결과를 제공하세요:
{{
    "is_valid": true/false,
    "quality_score": 0.0-1.0,
    "issues": [
        "문제점 1",
        "문제점 2"
    ],
    "strengths": [
        "강점 1",
        "강점 2"
    ],
    "recommendations": [
        "개선 권고 1",
        "개선 권고 2"
    ]
}}
""".format(answer=answer[:2000])  # 답변이 너무 길면 잘라냄
                return validation_criteria

            chain_steps.append({
                "name": "answer_validation",
                "prompt_builder": build_validation_prompt,
                "input_extractor": lambda prev: prev,  # 이전 단계의 답변을 검증
                "output_parser": lambda response, prev: AnswerParser.parse_validation_response(response),
                "validator": lambda output: output and isinstance(output, dict),
                "required": True,
                "skip_if": lambda prev: not prev or len(str(prev).strip()) < 10  # 너무 짧으면 건너뛰기
            })

            # Step 3: 개선 지시 생성 (문제점이 있는 경우)
            def build_improvement_instructions_prompt(prev_output, initial_input):
                # prev_output은 validation_result, initial_input에서 원본 답변 찾기
                validation_result = prev_output
                if not isinstance(validation_result, dict):
                    validation_result = {}

                # 초기 답변 찾기 (체인 히스토리에서 또는 initial_input에서)
                original_answer = ""
                if isinstance(initial_input, dict):
                    # 체인 히스토리에서 초기 답변 찾기
                    chain_history = initial_input.get("_chain_history", [])
                    for step in chain_history:
                        if step.get("step_name") == "initial_answer_generation":
                            original_answer = step.get("output", "")
                            break

                if not original_answer:
                    original_answer = str(prev_output) if not isinstance(prev_output, dict) else ""
                if not isinstance(validation_result, dict) or validation_result.get("is_valid", True):
                    return None  # 문제가 없으면 건너뛰기

                issues = validation_result.get("issues", [])
                recommendations = validation_result.get("recommendations", [])
                quality_score = validation_result.get("quality_score", 1.0)

                if quality_score >= 0.75:  # 품질이 충분히 높으면 건너뛰기
                    return None

                improvement_prompt = f"""
다음 답변의 검증 결과를 바탕으로 개선 지시를 작성하세요.

**원본 답변**:
{original_answer[:1500]}

**검증 결과**:
- 품질 점수: {quality_score:.2f}/1.0
- 문제점: {', '.join(issues[:5]) if issues else '없음'}
- 권고사항: {', '.join(recommendations[:5]) if recommendations else '없음'}

**개선 지시 작성 요청**:
위 검증 결과를 바탕으로 답변을 개선하기 위한 구체적인 지시사항을 작성하세요.

다음 형식으로 제공하세요:
{{
    "needs_improvement": true,
    "improvement_instructions": [
        "개선 지시 1: 구체적으로 어떤 부분을 어떻게 개선할지",
        "개선 지시 2: ..."
    ],
    "preserve_content": [
        "보존할 내용 1",
        "보존할 내용 2"
    ],
    "focus_areas": [
        "중점 개선 영역 1",
        "중점 개선 영역 2"
    ]
}}
"""
                return improvement_prompt

            chain_steps.append({
                "name": "improvement_instructions",
                "prompt_builder": build_improvement_instructions_prompt,
                "input_extractor": lambda prev: prev,  # 이전 단계 출력 사용 (validation 결과)
                "output_parser": lambda response, prev: AnswerParser.parse_improvement_instructions(response),
                "validator": lambda output: output is None or (isinstance(output, dict) and output.get("needs_improvement")),
                "required": False,  # 선택 단계 (문제가 없으면 건너뛰기)
                "skip_if": lambda prev: prev is None or (isinstance(prev, dict) and prev.get("is_valid", True) and prev.get("quality_score", 1.0) >= 0.75)
            })

            # Step 4: 개선된 답변 생성 (Step 3가 있는 경우)
            def build_improved_answer_prompt(prev_output, initial_input):
                improvement_instructions = prev_output
                if not improvement_instructions or not isinstance(improvement_instructions, dict):
                    return None  # 개선 지시가 없으면 건너뛰기

                # initial_input에서 원본 프롬프트 찾기
                if isinstance(initial_input, dict):
                    original_prompt = initial_input.get("prompt", optimized_prompt)
                else:
                    original_prompt = optimized_prompt
                if not improvement_instructions or not isinstance(improvement_instructions, dict):
                    return None  # 개선 지시가 없으면 건너뛰기

                if not improvement_instructions.get("needs_improvement", False):
                    return None

                # 원본 프롬프트와 개선 지시를 결합
                improvement_text = "\n".join(improvement_instructions.get("improvement_instructions", []))
                preserve_content = "\n".join(improvement_instructions.get("preserve_content", []))

                improved_prompt = f"""{original_prompt}

---

## 🔧 개선 요청

위 프롬프트로 생성한 답변을 다음 지시사항에 따라 개선하세요:

**개선 지시사항**:
{improvement_text}

**보존할 내용** (반드시 포함):
{preserve_content if preserve_content else "원본 답변의 모든 법적 정보와 근거"}

**중점 개선 영역**:
{', '.join(improvement_instructions.get("focus_areas", []))}

위 지시사항에 따라 답변을 개선하되, 원본 답변의 법적 근거와 정보는 반드시 보존하세요.
"""
                return improved_prompt

            chain_steps.append({
                "name": "improved_answer_generation",
                "prompt_builder": build_improved_answer_prompt,
                "input_extractor": lambda prev: prev,  # 개선 지시를 입력으로 사용
                "output_parser": lambda response, prev: WorkflowUtils.normalize_answer(response),
                "validator": lambda output: output and len(output.strip()) > 10,
                "required": False,  # 선택 단계 (개선 지시가 없으면 건너뛰기)
                "skip_if": lambda prev: prev is None or (isinstance(prev, dict) and not prev.get("needs_improvement", False))
            })

            # Step 5: 최종 검증 (개선된 답변이 있는 경우)
            def build_final_validation_prompt(prev_output, initial_input):
                # prev_output이 딕셔너리인 경우 처리
                if isinstance(prev_output, dict):
                    answer = prev_output.get("answer") or prev_output.get("content") or str(prev_output)
                else:
                    answer = prev_output if isinstance(prev_output, str) else str(prev_output)

                final_validation_prompt = f"""
다음 답변의 최종 품질을 검증하세요:

답변:
{answer[:2000]}

다음 기준으로 최종 검증을 수행하세요:
1. 답변이 질문에 직접적으로 답변하는가?
2. 법적 근거가 충분한가?
3. 구조와 논리적 흐름이 명확한가?
4. 길이가 적절한가? (최소 50자 이상)

다음 형식으로 최종 검증 결과를 제공하세요:
{{
    "final_score": 0.0-1.0,
    "meets_quality_threshold": true/false,
    "summary": "검증 요약"
}}
"""
                return final_validation_prompt

            chain_steps.append({
                "name": "final_validation",
                "prompt_builder": build_final_validation_prompt,
                "input_extractor": lambda prev: prev,  # 최종 답변 검증
                "output_parser": lambda response, prev: AnswerParser.parse_final_validation_response(response),
                "validator": lambda output: output is None or isinstance(output, dict),
                "required": False,  # 선택 단계
                "skip_if": lambda prev: not prev or len(str(prev).strip()) < 10
            })

            # 체인 실행
            # initial_input을 각 prompt_builder에서 사용할 수 있도록 전달
            # 각 prompt_builder의 두 번째 파라미터로 전달됨
            initial_input_dict = {
                "prompt": optimized_prompt,
                "query": query,
                "context_dict": context_dict,
                "quality_feedback": quality_feedback,
                "is_retry": is_retry
            }

            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=2,  # 각 단계 최대 2회 재시도
                stop_on_failure=False  # 일부 단계 실패해도 계속 진행
            )

            # 최종 답변 추출
            final_output = chain_result.get("final_output")

            # 체인 히스토리에서 답변 찾기 (우선순위: 개선된 답변 > 초기 답변)
            final_answer = ""
            chain_history = chain_result.get("chain_history", [])

            # Step 4 (improved_answer_generation)의 출력 찾기
            for step in reversed(chain_history):  # 역순으로 검색 (최신 우선)
                if step.get("step_name") == "improved_answer_generation" and step.get("success"):
                    output = step.get("output")
                    if isinstance(output, str) and len(output.strip()) > 10:
                        final_answer = output
                        break

            # 개선된 답변이 없으면 Step 1의 출력 사용
            if not final_answer:
                for step in chain_history:
                    if step.get("step_name") == "initial_answer_generation" and step.get("success"):
                        output = step.get("output")
                        if isinstance(output, str) and len(output.strip()) > 10:
                            final_answer = output
                            break

            # 여전히 없으면 final_output에서 추출
            if not final_answer:
                if isinstance(final_output, str):
                    final_answer = final_output
                elif isinstance(final_output, dict):
                    final_answer = final_output.get("improved_answer") or final_output.get("initial_answer") or ""
                else:
                    final_answer = str(final_output) if final_output else ""

            # 체인 실행 결과 로깅
            chain_summary = chain_executor.get_chain_summary()
            self.logger.info(
                f"✅ [PROMPT CHAIN] Executed {chain_summary['total_steps']} steps, "
                f"{chain_summary['successful_steps']} successful, "
                f"{chain_summary['failed_steps']} failed, "
                f"Total time: {chain_summary['total_time']:.2f}s"
            )

            # 최종 답변이 비어있으면 폴백
            if not final_answer or len(final_answer.strip()) < 10:
                self.logger.warning("⚠️ [CHAIN] Final answer is empty, using fallback")
                # 초기 프롬프트로 단순 생성
                response = self.call_llm_with_retry(optimized_prompt)
                final_answer = WorkflowUtils.normalize_answer(response)

            return final_answer

        except Exception as e:
            self.logger.error(f"❌ [CHAIN ERROR] Prompt chain failed: {e}")
            # 폴백: 기존 방식 사용
            response = self.call_llm_with_retry(optimized_prompt)
            return WorkflowUtils.normalize_answer(response)

    def validate_answer_uses_context(
        self,
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        답변-컨텍스트 일치도 검증 (quality_validators 모듈 사용)

        Args:
            answer: 생성된 답변
            context: 컨텍스트 정보
            query: 원본 질문
            retrieved_docs: 검색된 문서 목록

        Returns:
            검증 결과 딕셔너리
        """
        result = AnswerValidator.validate_answer_uses_context(
            answer=answer,
            context=context,
            query=query,
            retrieved_docs=retrieved_docs
        )

        # 로깅
        self.logger.info(
            f"✅ [ANSWER-CONTEXT VALIDATION] Coverage: {result.get('coverage_score', 0.0):.2f}, "
            f"Keyword: {result.get('keyword_coverage', 0.0):.2f}, Citation: {result.get('citation_coverage', 0.0):.2f}, "
            f"Uses context: {result.get('uses_context', False)}, Needs regeneration: {result.get('needs_regeneration', False)}"
        )

        return result

    def track_search_to_answer_pipeline(
        self,
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """
        검색-답변 파이프라인 품질 추적

        Args:
            state: 워크플로우 상태

        Returns:
            파이프라인 메트릭 딕셔너리
        """
        try:
            metadata = WorkflowUtils.get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            search_meta = metadata.get("search", {})
            context_validation = metadata.get("context_validation", {})
            answer_validation = metadata.get("answer_validation", {})

            pipeline_metrics = {
                # 검색 품질
                "search_quality": {
                    "doc_count": search_meta.get("final_count", 0),
                    "avg_relevance": search_meta.get("avg_relevance", 0.0),
                    "semantic_results": search_meta.get("semantic_results_count", 0),
                    "keyword_results": search_meta.get("keyword_results_count", 0)
                },
                # 컨텍스트 품질
                "context_quality": {
                    "relevance_score": context_validation.get("relevance_score", 0.0),
                    "coverage_score": context_validation.get("coverage_score", 0.0),
                    "sufficiency_score": context_validation.get("sufficiency_score", 0.0),
                    "overall_score": context_validation.get("overall_score", 0.0),
                    "docs_included": metadata.get("context_validation", {}).get("docs_included", 0)
                },
                # 답변 품질
                "answer_quality": {
                    "uses_context": answer_validation.get("uses_context", False),
                    "coverage_score": answer_validation.get("coverage_score", 0.0),
                    "keyword_coverage": answer_validation.get("keyword_coverage", 0.0),
                    "citation_coverage": answer_validation.get("citation_coverage", 0.0),
                    "citations_found": answer_validation.get("citations_found", 0),
                    "citations_expected": answer_validation.get("citations_expected", 0)
                },
                # 파이프라인 종합 점수
                "pipeline_score": 0.0
            }

            # 종합 점수 계산
            search_score = search_meta.get("avg_relevance", 0.5)
            context_score = context_validation.get("overall_score", 0.5)
            answer_score = answer_validation.get("coverage_score", 0.5)

            pipeline_score = (search_score * 0.3 + context_score * 0.3 + answer_score * 0.4)
            pipeline_metrics["pipeline_score"] = pipeline_score

            # 메타데이터에 저장
            metadata["pipeline_metrics"] = pipeline_metrics
            WorkflowUtils.set_state_value(state, "metadata", metadata)

            self.logger.info(
                f"📊 [PIPELINE TRACKING] Overall score: {pipeline_score:.2f}, "
                f"Search: {search_score:.2f}, Context: {context_score:.2f}, Answer: {answer_score:.2f}"
            )

            return pipeline_metrics

        except Exception as e:
            self.logger.warning(f"Pipeline tracking failed: {e}")
            return {}

    def call_llm_with_retry(self, prompt: str, max_retries: int = WorkflowConstants.MAX_RETRIES) -> str:
        """
        LLM 호출 (재시도 로직 포함)

        Args:
            prompt: 프롬프트 문자열
            max_retries: 최대 재시도 횟수

        Returns:
            LLM 응답 문자열
        """
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return WorkflowUtils.extract_response_content(response)
            except Exception as e:
                self.logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(WorkflowConstants.RETRY_DELAY)

        return "LLM 호출에 실패했습니다."

    def get_quality_feedback_for_retry(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """
        재시도를 위한 품질 피드백 생성

        Args:
            state: 워크플로우 상태

        Returns:
            피드백 정보 딕셔너리
        """
        quality_meta = WorkflowUtils.get_quality_metadata(state)
        quality_score = quality_meta["quality_score"]

        # 메타데이터에서 품질 체크 정보 가져오기
        metadata = WorkflowUtils.get_state_value(state, "metadata", {})
        quality_metadata = metadata.get("quality_metadata", {})
        quality_checks = quality_metadata.get("quality_checks", {})
        legal_validation = WorkflowUtils.get_state_value(state, "legal_basis_validation", {})

        if not isinstance(legal_validation, dict):
            legal_validation = {}

        feedback = {
            "previous_score": quality_score,
            "failed_checks": [],
            "recommendations": [],
            "retry_strategy": None
        }

        # 실패한 체크 확인
        if not quality_checks.get("has_answer", True):
            feedback["failed_checks"].append("답변이 비어있음")
            feedback["recommendations"].append("반드시 답변을 생성하세요")

        if not quality_checks.get("min_length", True):
            answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
            current_length = len(answer)
            min_length = WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION
            feedback["failed_checks"].append(f"답변이 너무 짧음 ({current_length}자)")
            feedback["recommendations"].append(f"최소 {min_length}자 이상의 상세한 답변을 제공하세요")

        if not quality_checks.get("has_sources", True):
            feedback["failed_checks"].append("법률 소스가 없음")
            feedback["recommendations"].append("관련 법령, 판례, 법률 조항을 인용하여 답변하세요")

        if not quality_checks.get("legal_basis_valid", True):
            issues = legal_validation.get("issues", [])
            if isinstance(issues, list):
                issues_str = ", ".join(str(issue) for issue in issues[:3])
            else:
                issues_str = str(issues)[:100]
            feedback["failed_checks"].append("법령 검증 실패")
            feedback["recommendations"].extend([
                "법적 근거를 명확히 제시하세요",
                f"문제점: {issues_str}"
            ])
            feedback["retry_strategy"] = "search"  # 검색 재시도 권장

        # 품질 점수 기반 피드백
        if quality_score < 0.4:
            feedback["recommendations"].append("답변의 품질이 매우 낮습니다. 더 구체적이고 상세하게 작성하세요")
        elif quality_score < 0.6:
            feedback["recommendations"].append("답변의 구조와 내용을 개선하여 더 명확하게 설명하세요")

        # 길이 피드백
        answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
        if len(answer) < 50:
            feedback["recommendations"].append(
                "답변이 너무 짧습니다. 다음을 포함하여 상세히 작성하세요:\n"
                "- 질문에 대한 직접적인 답변\n"
                "- 관련 법령 및 조항\n"
                "- 실무 적용 시 주의사항\n"
                "- 참고할 만한 판례 (있는 경우)"
            )

        return feedback

    def determine_retry_prompt_type(self, quality_feedback: Dict[str, Any]) -> str:
        """
        피드백 기반 재시도 프롬프트 타입 결정

        Args:
            quality_feedback: 품질 피드백 딕셔너리

        Returns:
            프롬프트 타입 문자열
        """
        failed_checks = quality_feedback.get("failed_checks", [])

        # 기본 프롬프트 타입
        base_type = "korean_legal_expert"

        # 법령 검증 실패 → 법적 근거 강조
        if any("법령" in check or "법" in check for check in failed_checks):
            return f"{base_type}_with_legal_basis"
        # 길이 부족 → 상세 설명 강조
        elif any("짧" in check or "길이" in check for check in failed_checks):
            return f"{base_type}_detailed"
        # 소스 없음 → 출처 인용 강조
        elif any("소스" in check or "출처" in check for check in failed_checks):
            return f"{base_type}_with_sources"
        # 일반 개선
        else:
            return f"{base_type}_improved"

    def assess_improvement_potential(
        self,
        quality_score: float,
        quality_checks: Dict[str, bool],
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """
        재시도 시 개선 가능성 평가

        Args:
            quality_score: 현재 품질 점수
            quality_checks: 품질 체크 결과 딕셔너리
            state: 워크플로우 상태

        Returns:
            개선 가능성 평가 결과 딕셔너리
        """
        potential = 0.0
        strategy = None
        reasons = []

        # 1. 컨텍스트가 없어서 실패한 경우 → 검색 재시도로 개선 가능성 높음
        retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])
        if len(retrieved_docs) == 0 and not quality_checks.get("has_sources", True):
            potential += 0.4
            strategy = "retry_search"
            reasons.append("컨텍스트 부족으로 검색 재시도 시 개선 가능")

        # 2. 답변이 짧은 경우 → 프롬프트 개선으로 개선 가능성 높음
        answer = WorkflowUtils.normalize_answer(WorkflowUtils.get_state_value(state, "answer", ""))
        if len(answer) < 50 and quality_score > 0.3:
            potential += 0.5
            if strategy is None:
                strategy = "retry_generate"
            reasons.append("답변 길이 문제로 프롬프트 개선 시 개선 가능")

        # 3. 법령 검증 실패 → 검색 개선으로 해결 가능성 높음
        if not quality_checks.get("legal_basis_valid", True):
            potential += 0.6
            strategy = "retry_search"
            reasons.append("법령 검증 실패로 관련 문서 검색 시 개선 가능")

        # 4. 품질 점수가 너무 낮으면 (0.2 이하) → 재시도 효과 낮음
        if quality_score < 0.2:
            potential *= 0.5
            reasons.append("품질 점수가 너무 낮아 재시도 효과가 제한적일 수 있음")

        return {
            "potential": min(potential, 1.0),
            "strategy": strategy,
            "reasons": reasons
        }

    def generate_fallback_answer(self, state: LegalWorkflowState) -> str:
        """
        폴백 답변 생성

        Args:
            state: 워크플로우 상태

        Returns:
            폴백 답변 문자열
        """
        query = WorkflowUtils.get_state_value(state, "query", "")
        query_type = WorkflowUtils.get_state_value(state, "query_type", "")
        retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])

        # retrieved_docs 안전하게 처리
        context_parts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)
            if content:
                context_parts.append(content)

        context = "\n".join(context_parts[:5])  # 최대 5개 문서만 사용

        return f"""## 답변

질문: {query}

이 질문은 {query_type} 영역에 해당합니다.

## 관련 법률 정보
{context}

## 주요 포인트
1. 위 정보를 바탕으로 구체적인 조치를 취하시기 바랍니다.
2. 정확한 법률적 조언을 위해서는 전문가와 상담하시는 것을 권장합니다.
3. 관련 법조문과 판례를 추가로 확인하시기 바랍니다.

## 주의사항
- 이 답변은 일반적인 정보 제공 목적이며, 구체적인 법률적 조언이 아닙니다.
- 실제 사안에 대해서는 전문 변호사와 상담하시기 바랍니다."""
