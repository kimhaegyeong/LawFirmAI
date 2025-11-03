# -*- coding: utf-8 -*-
"""
Prompt Chaining ?ŒìŠ¤??
Phase 6, 7 êµ¬í˜„ ?ŒìŠ¤??
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from source.agents.prompt_chain_executor import PromptChainExecutor


class TestPromptChainExecutor:
    """PromptChainExecutor ?ŒìŠ¤??""

    def setup_method(self):
        """?ŒìŠ¤???¤ì •"""
        self.llm = Mock()
        self.llm.invoke = Mock()
        self.logger = Mock()

        self.executor = PromptChainExecutor(self.llm, self.logger)

    def test_chain_execution_success(self):
        """ì²´ì¸ ?¤í–‰ ?±ê³µ ?ŒìŠ¤??""
        # LLM ?‘ë‹µ ëª¨í‚¹
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
        """ì²´ì¸ ê²€ì¦??ŒìŠ¤??""
        # LLM ?‘ë‹µ ëª¨í‚¹
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
        """ì²´ì¸ ?¤íŒ¨ ì²˜ë¦¬ ?ŒìŠ¤??""
        # LLM ?¸ì¶œ ?¤íŒ¨ ëª¨í‚¹
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
    """ì§ì ‘ ?µë? ì²´ì¸ ?ŒìŠ¤??""

    def test_direct_answer_chain_integration(self):
        """ì§ì ‘ ?µë? ì²´ì¸ ?µí•© ?ŒìŠ¤??""
        # ???ŒìŠ¤?¸ëŠ” ?¤ì œ ?Œí¬?Œë¡œ?°ì? ?µí•©?˜ì—¬ ?¤í–‰?´ì•¼ ?©ë‹ˆ??
        # ?¬ê¸°?œëŠ” ê¸°ë³¸ êµ¬ì¡°ë§??•ì¸
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
