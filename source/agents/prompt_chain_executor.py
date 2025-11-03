# -*- coding: utf-8 -*-
"""
Prompt Chaining 실행기
각 LLM 호출을 순차적으로 연결하여 이전 단계의 출력을 다음 단계의 입력으로 사용
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional


class PromptChainExecutor:
    """
    Prompt chaining을 위한 실행기

    각 LLM 호출을 순차적으로 실행하고,
    이전 단계의 출력을 다음 단계의 입력으로 전달
    """

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        초기화

        Args:
            llm: LLM 인스턴스 (invoke 메서드를 가진 객체)
            logger: 로거 인스턴스
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.chain_history: List[Dict[str, Any]] = []

    def execute_chain(
        self,
        chain_steps: List[Dict[str, Any]],
        initial_input: Optional[Any] = None,
        max_iterations: int = 3,
        stop_on_failure: bool = False,
        validate_final_output: bool = True
    ) -> Dict[str, Any]:
        """
        체인 실행

        Args:
            chain_steps: 체인 단계 정의 리스트
            initial_input: 초기 입력 (첫 번째 단계에 전달)
            max_iterations: 최대 반복 횟수 (재시도)
            stop_on_failure: 실패 시 체인 중단 여부
            validate_final_output: 최종 출력 검증 여부

        Returns:
            {
                "success": bool,
                "final_output": Any,  # 최종 출력
                "chain_history": List[Dict],  # 각 단계 실행 히스토리
                "steps_executed": List[str],  # 실행된 단계 이름
                "errors": List[str],  # 에러 리스트
                "validation_results": Dict,  # 검증 결과
            }
        """
        # initial_input을 인스턴스 변수로 저장 (prompt_builder에서 접근 가능하도록)
        self._current_initial_input = initial_input

        start_time = time.time()
        self.chain_history = []
        errors = []
        steps_executed = []
        previous_output = initial_input
        validation_results = {}

        self.logger.info(f"🔄 [CHAIN START] Executing {len(chain_steps)} steps")

        try:
            for step_idx, step_config in enumerate(chain_steps):
                step_name = step_config.get("name", f"step_{step_idx + 1}")
                required = step_config.get("required", True)

                # skip_if 조건 확인
                skip_if = step_config.get("skip_if")
                if skip_if and callable(skip_if) and previous_output:
                    try:
                        if skip_if(previous_output):
                            self.logger.info(f"⏭️ [CHAIN] Skipping step '{step_name}' (skip_if condition met)")
                            steps_executed.append(f"{step_name} (skipped)")
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error in skip_if for step '{step_name}': {e}")

                # 단계 실행
                step_result = self._execute_step(
                    step_config,
                    previous_output,
                    step_idx,
                    max_iterations
                )

                if not step_result["success"]:
                    error_msg = f"Step '{step_name}' failed: {step_result.get('error', 'Unknown error')}"
                    errors.append(error_msg)
                    self.logger.error(f"❌ [CHAIN] {error_msg}")

                    if required:
                        if stop_on_failure:
                            self.logger.error(f"🛑 [CHAIN] Stopping chain due to required step failure")
                            break
                        else:
                            # 필수 단계 실패 시 이전 출력 사용
                            self.logger.warning(f"⚠️ [CHAIN] Using previous output for failed required step")
                    else:
                        # 선택 단계 실패 시 건너뛰기
                        self.logger.warning(f"⚠️ [CHAIN] Skipping optional step '{step_name}' after failure")
                        continue

                # 단계 성공 시 출력 업데이트
                if step_result.get("output") is not None:
                    previous_output = step_result["output"]

                steps_executed.append(step_name)
                self.chain_history.append({
                    "step_name": step_name,
                    "step_idx": step_idx,
                    "success": step_result["success"],
                    "output": step_result.get("output"),
                    "error": step_result.get("error"),
                    "execution_time": step_result.get("execution_time", 0)
                })

            # 최종 출력 검증 (Phase 7)
            if validate_final_output and previous_output is not None:
                validation_results = self._validate_final_output(
                    previous_output,
                    chain_steps,
                    self.chain_history
                )

                if not validation_results.get("is_valid", True):
                    self.logger.warning(
                        f"⚠️ [CHAIN VALIDATION] Final output validation failed: "
                        f"{validation_results.get('issues', [])}"
                    )

            # 최종 결과
            total_time = time.time() - start_time
            success = len(errors) == 0 and len(steps_executed) > 0
            if validate_final_output:
                success = success and validation_results.get("is_valid", True)

            result = {
                "success": success,
                "final_output": previous_output,
                "chain_history": self.chain_history,
                "steps_executed": steps_executed,
                "errors": errors,
                "total_execution_time": total_time,
                "validation_results": validation_results if validate_final_output else {}
            }

            self.logger.info(
                f"{'✅' if success else '⚠️'} [CHAIN END] "
                f"Executed {len(steps_executed)} steps in {total_time:.2f}s, "
                f"Errors: {len(errors)}, "
                f"Validation: {'✅' if validation_results.get('is_valid', True) else '❌'}"
            )

            return result

        except Exception as e:
            error_msg = f"Chain execution failed: {e}"
            self.logger.error(f"❌ [CHAIN ERROR] {error_msg}")
            errors.append(error_msg)

            return {
                "success": False,
                "final_output": previous_output,
                "chain_history": self.chain_history,
                "steps_executed": steps_executed,
                "errors": errors,
                "total_execution_time": time.time() - start_time,
                "validation_results": {}
            }

    def _validate_final_output(
        self,
        final_output: Any,
        chain_steps: List[Dict[str, Any]],
        chain_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        최종 출력 검증

        Args:
            final_output: 최종 출력
            chain_steps: 체인 단계 정의
            chain_history: 체인 실행 히스토리

        Returns:
            {
                "is_valid": bool,
                "issues": List[str],
                "quality_score": float,
                "recommendations": List[str]
            }
        """
        issues = []
        quality_score = 1.0

        try:
            # 1. 출력이 None이 아닌지 확인
            if final_output is None:
                issues.append("Final output is None")
                quality_score = 0.0
                return {
                    "is_valid": False,
                    "issues": issues,
                    "quality_score": quality_score,
                    "recommendations": ["Check chain execution for errors"]
                }

            # 2. 필수 단계가 성공적으로 실행되었는지 확인
            required_steps = [step for step in chain_steps if step.get("required", True)]
            successful_required_steps = [
                step for step in chain_history
                if step.get("success", False) and any(
                    step.get("step_name") == req_step.get("name")
                    for req_step in required_steps
                )
            ]

            if len(successful_required_steps) < len(required_steps):
                issues.append(
                    f"Required steps not all successful: "
                    f"{len(successful_required_steps)}/{len(required_steps)}"
                )
                quality_score -= 0.3

            # 3. 출력 타입 검증
            if isinstance(final_output, str):
                if len(final_output.strip()) == 0:
                    issues.append("Final output is empty string")
                    quality_score -= 0.5
                elif len(final_output.strip()) < 10:
                    issues.append("Final output is too short (< 10 characters)")
                    quality_score -= 0.2
            elif isinstance(final_output, dict):
                if len(final_output) == 0:
                    issues.append("Final output is empty dictionary")
                    quality_score -= 0.3
            elif isinstance(final_output, list):
                if len(final_output) == 0:
                    issues.append("Final output is empty list")
                    quality_score -= 0.3

            # 4. 에러가 있는지 확인
            errors_in_history = [step for step in chain_history if step.get("error")]
            if errors_in_history:
                issues.append(f"Found {len(errors_in_history)} errors in chain history")
                quality_score -= min(0.5, len(errors_in_history) * 0.1)

            # 5. 품질 점수 정규화
            quality_score = max(0.0, min(1.0, quality_score))

            recommendations = []
            if issues:
                recommendations.append("Review chain execution history for details")
                if quality_score < 0.7:
                    recommendations.append("Consider retrying with different parameters")

            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "quality_score": quality_score,
                "recommendations": recommendations
            }

        except Exception as e:
            self.logger.warning(f"Error during final output validation: {e}")
            return {
                "is_valid": True,  # 검증 실패 시 기본적으로 유효하다고 가정
                "issues": [f"Validation error: {e}"],
                "quality_score": 0.8,
                "recommendations": ["Manual review recommended"]
            }

    def _execute_step(
        self,
        step_config: Dict[str, Any],
        previous_output: Any,
        step_idx: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        단일 단계 실행

        Args:
            step_config: 단계 설정
            previous_output: 이전 단계 출력
            step_idx: 단계 인덱스
            max_iterations: 최대 반복 횟수

        Returns:
            {
                "success": bool,
                "output": Any,
                "error": str,
                "execution_time": float
            }
        """
        step_start_time = time.time()
        step_name = step_config.get("name", f"step_{step_idx + 1}")

        try:
            # 입력 추출
            step_input = self._extract_step_input(step_config, previous_output)

            # 프롬프트 생성
            # initial_input 전달을 위해 체인 시작 시 저장한 initial_input 사용
            current_initial_input = getattr(self, '_current_initial_input', None)
            prompt = self._build_prompt(step_config, step_input, previous_output, current_initial_input)

            if not prompt:
                return {
                    "success": False,
                    "output": None,
                    "error": "Failed to build prompt",
                    "execution_time": time.time() - step_start_time
                }

            # LLM 호출 (재시도 포함)
            llm_response = None
            last_error = None

            for attempt in range(max_iterations):
                try:
                    self.logger.debug(
                        f"🔄 [CHAIN STEP] '{step_name}' - Attempt {attempt + 1}/{max_iterations}"
                    )

                    llm_response = self.llm.invoke(prompt)

                    # 응답 추출
                    response_content = self._extract_response_content(llm_response)

                    # 출력 파싱
                    parsed_output = self._parse_output(step_config, response_content, previous_output)

                    # 검증 (있는 경우)
                    validator = step_config.get("validator")
                    if validator and callable(validator):
                        if not validator(parsed_output):
                            if attempt < max_iterations - 1:
                                self.logger.warning(
                                    f"⚠️ [CHAIN STEP] '{step_name}' validation failed, retrying..."
                                )
                                continue
                            else:
                                return {
                                    "success": False,
                                    "output": parsed_output,
                                    "error": "Output validation failed",
                                    "execution_time": time.time() - step_start_time
                                }

                    # 성공
                    execution_time = time.time() - step_start_time
                    self.logger.info(
                        f"✅ [CHAIN STEP] '{step_name}' completed in {execution_time:.2f}s"
                    )

                    return {
                        "success": True,
                        "output": parsed_output,
                        "error": None,
                        "execution_time": execution_time
                    }

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(
                        f"⚠️ [CHAIN STEP] '{step_name}' attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < max_iterations - 1:
                        time.sleep(0.5)  # 재시도 전 대기
                        continue

            # 모든 재시도 실패
            return {
                "success": False,
                "output": None,
                "error": f"All {max_iterations} attempts failed. Last error: {last_error}",
                "execution_time": time.time() - step_start_time
            }

        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": f"Step execution error: {e}",
                "execution_time": time.time() - step_start_time
            }

    def _extract_step_input(
        self,
        step_config: Dict[str, Any],
        previous_output: Any
    ) -> Any:
        """
        단계 입력 추출

        Args:
            step_config: 단계 설정
            previous_output: 이전 단계 출력

        Returns:
            단계 입력
        """
        input_extractor = step_config.get("input_extractor")

        if input_extractor and callable(input_extractor):
            try:
                return input_extractor(previous_output)
            except Exception as e:
                self.logger.warning(f"Input extractor failed: {e}, using previous_output directly")
                return previous_output

        # 기본값: 이전 출력을 그대로 사용
        return previous_output

    def _build_prompt(
        self,
        step_config: Dict[str, Any],
        step_input: Any,
        previous_output: Any,
        initial_input: Any = None
    ) -> Optional[str]:
        """
        프롬프트 생성

        Args:
            step_config: 단계 설정
            step_input: 단계 입력
            previous_output: 이전 단계 출력
            initial_input: 초기 입력 (체인 시작 시 전달된 입력)

        Returns:
            프롬프트 문자열 또는 None
        """
        prompt_template = step_config.get("prompt_template")
        prompt_builder = step_config.get("prompt_builder")

        # prompt_builder 함수가 있으면 우선 사용
        if prompt_builder and callable(prompt_builder):
            try:
                # prompt_builder 함수 시그니처 확인
                import inspect
                sig = inspect.signature(prompt_builder)
                param_count = len(sig.parameters)

                if param_count == 1:
                    # 단일 파라미터 (prev_output만)
                    return prompt_builder(previous_output)
                elif param_count == 2:
                    # 두 파라미터 (prev_output, initial_input)
                    return prompt_builder(previous_output, initial_input if initial_input is not None else step_input)
                else:
                    # 기존 방식 (step_input, previous_output, step_config)
                    return prompt_builder(step_input, previous_output, step_config)
            except Exception as e:
                self.logger.error(f"Prompt builder failed: {e}")
                return None

        # prompt_template 문자열이 있으면 사용
        if prompt_template:
            try:
                # 간단한 문자열 포맷팅 (딕셔너리 입력 지원)
                if isinstance(step_input, dict):
                    return prompt_template.format(**step_input, previous_output=previous_output)
                else:
                    return prompt_template.format(input=step_input, previous_output=previous_output)
            except Exception as e:
                self.logger.error(f"Prompt template formatting failed: {e}")
                return None

        # 둘 다 없으면 에러
        self.logger.error("Neither prompt_template nor prompt_builder provided")
        return None

    def _extract_response_content(self, response: Any) -> str:
        """
        LLM 응답에서 내용 추출

        Args:
            response: LLM 응답

        Returns:
            응답 내용 문자열
        """
        if isinstance(response, str):
            return response

        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                return content.get("content", content.get("answer", str(content)))

        if isinstance(response, dict):
            return response.get("content", response.get("answer", str(response)))

        return str(response)

    def _parse_output(
        self,
        step_config: Dict[str, Any],
        response_content: str,
        previous_output: Any
    ) -> Any:
        """
        출력 파싱

        Args:
            step_config: 단계 설정
            response_content: LLM 응답 내용
            previous_output: 이전 단계 출력

        Returns:
            파싱된 출력
        """
        output_parser = step_config.get("output_parser")

        if output_parser and callable(output_parser):
            try:
                return output_parser(response_content, previous_output)
            except Exception as e:
                self.logger.warning(f"Output parser failed: {e}, using raw response")
                return response_content

        # 기본값: 원본 응답 반환
        return response_content

    def get_chain_summary(self) -> Dict[str, Any]:
        """
        체인 실행 요약 반환

        Returns:
            {
                "total_steps": int,
                "successful_steps": int,
                "failed_steps": int,
                "total_time": float,
                "steps": List[Dict]
            }
        """
        if not self.chain_history:
            return {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "total_time": 0.0,
                "steps": []
            }

        successful_steps = sum(1 for step in self.chain_history if step["success"])
        failed_steps = len(self.chain_history) - successful_steps
        total_time = sum(step.get("execution_time", 0) for step in self.chain_history)

        return {
            "total_steps": len(self.chain_history),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "total_time": total_time,
            "steps": self.chain_history
        }
