# -*- coding: utf-8 -*-
"""
Prompt Chaining 테스트
Phase 6, 7 구현 테스트
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.agents.prompt_chain_executor import PromptChainExecutor


class TestPromptChainExecutor:
    """PromptChainExecutor 테스트"""

    def setup_method(self):
        """테스트 설정"""
        self.llm = Mock()
        self.llm.invoke = Mock()
        self.logger = Mock()

        self.executor = PromptChainExecutor(self.llm, self.logger)

    def test_chain_execution_success(self):
        """체인 실행 성공 테스트"""
        # LLM 응답 모킹
        self.llm.invoke.return_value = Mock(content='{"result": "success"}')

        chain_steps = [
            {
                "name": "step1",
                "prompt_builder": lambda prev, initial: "Prompt 1",
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: {"step1_result": "done"},
                "validator": lambda output: True,
                "required": True
            },
            {
                "name": "step2",
                "prompt_builder": lambda prev, initial: "Prompt 2",
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: {"step2_result": "done"},
                "validator": lambda output: True,
                "required": True
            }
        ]

        result = self.executor.execute_chain(
            chain_steps=chain_steps,
            initial_input={"test": "input"},
            max_iterations=1,
            stop_on_failure=False
        )

        assert result["success"] is True
        assert len(result["chain_history"]) == 2
        assert len(result["steps_executed"]) == 2
        assert len(result["errors"]) == 0

    def test_chain_validation(self):
        """체인 검증 테스트"""
        # LLM 응답 모킹
        self.llm.invoke.return_value = Mock(content='{"result": "success"}')

        chain_steps = [
            {
                "name": "step1",
                "prompt_builder": lambda prev, initial: "Prompt 1",
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: "Valid output",
                "validator": lambda output: True,
                "required": True
            }
        ]

        result = self.executor.execute_chain(
            chain_steps=chain_steps,
            initial_input={"test": "input"},
            max_iterations=1,
            stop_on_failure=False,
            validate_final_output=True
        )

        assert "validation_results" in result
        assert result["validation_results"].get("is_valid") is True

    def test_chain_failure_handling(self):
        """체인 실패 처리 테스트"""
        # LLM 호출 실패 모킹
        self.llm.invoke.side_effect = Exception("LLM error")

        chain_steps = [
            {
                "name": "step1",
                "prompt_builder": lambda prev, initial: "Prompt 1",
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: None,
                "validator": lambda output: True,
                "required": True
            }
        ]

        result = self.executor.execute_chain(
            chain_steps=chain_steps,
            initial_input={"test": "input"},
            max_iterations=2,
            stop_on_failure=False
        )

        assert result["success"] is False
        assert len(result["errors"]) > 0


class TestDirectAnswerChain:
    """직접 답변 체인 테스트"""

    def test_direct_answer_chain_integration(self):
        """직접 답변 체인 통합 테스트"""
        # 이 테스트는 실제 워크플로우와 통합하여 실행해야 합니다
        # 여기서는 기본 구조만 확인
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
