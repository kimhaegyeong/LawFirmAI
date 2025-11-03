# -*- coding: utf-8 -*-
"""
Prompt Chaining ê°„ë‹¨ ?ŒìŠ¤??(pytest ?†ì´)
Phase 6, 7 êµ¬í˜„ ?ŒìŠ¤??
"""

import sys
import os
from unittest.mock import Mock, MagicMock

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_chain_executor():
    """PromptChainExecutor ê¸°ë³¸ ?ŒìŠ¤??""
    try:
        from source.agents.prompt_chain_executor import PromptChainExecutor

        # LLM ëª¨í‚¹
        llm = Mock()
        llm.invoke = Mock(return_value=MagicMock(content='{"result": "success"}'))

        # Logger ëª¨í‚¹
        logger = Mock()

        executor = PromptChainExecutor(llm, logger)

        # ê°„ë‹¨??ì²´ì¸ ?•ì˜
        chain_steps = [
            {
                "name": "test_step",
                "prompt_builder": lambda prev, initial: "Test prompt",
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: {"test": "result"},
                "validator": lambda output: True,
                "required": True
            }
        ]

        result = executor.execute_chain(
            chain_steps=chain_steps,
            initial_input={"test": "input"},
            max_iterations=1,
            stop_on_failure=False,
            validate_final_output=True
        )

        print(f"??Chain execution test passed")
        print(f"   Success: {result['success']}")
        print(f"   Steps executed: {len(result['steps_executed'])}")
        print(f"   Validation: {result.get('validation_results', {}).get('is_valid', False)}")
        return True

    except Exception as e:
        print(f"??Chain execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chain_validation():
    """ì²´ì¸ ê²€ì¦??ŒìŠ¤??""
    try:
        from source.agents.prompt_chain_executor import PromptChainExecutor

        llm = Mock()
        llm.invoke = Mock(return_value=MagicMock(content='Valid output'))

        logger = Mock()
        executor = PromptChainExecutor(llm, logger)

        # None ì¶œë ¥ ?ŒìŠ¤??(ê²€ì¦??¤íŒ¨)
        validation = executor._validate_final_output(
            final_output=None,
            chain_steps=[],
            chain_history=[]
        )

        assert validation["is_valid"] is False
        assert validation["quality_score"] == 0.0
        print(f"??Chain validation test passed (None output)")

        # ? íš¨??ì¶œë ¥ ?ŒìŠ¤??
        validation = executor._validate_final_output(
            final_output="Valid output string",
            chain_steps=[],
            chain_history=[]
        )

        assert validation["is_valid"] is True
        assert validation["quality_score"] > 0.5
        print(f"??Chain validation test passed (Valid output)")
        return True

    except Exception as e:
        print(f"??Chain validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Chaining ?ŒìŠ¤???œì‘")
    print("=" * 60)

    results = []

    # ?ŒìŠ¤??1: Chain Executor
    print("\n[?ŒìŠ¤??1] Chain Executor ?ŒìŠ¤??)
    results.append(test_chain_executor())

    # ?ŒìŠ¤??2: Chain Validation
    print("\n[?ŒìŠ¤??2] Chain Validation ?ŒìŠ¤??)
    results.append(test_chain_validation())

    # ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 60)
    print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("=" * 60)
    print(f"ì´??ŒìŠ¤?? {len(results)}")
    print(f"?±ê³µ: {sum(results)}")
    print(f"?¤íŒ¨: {len(results) - sum(results)}")

    if all(results):
        print("\n??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
        sys.exit(0)
    else:
        print("\n???¼ë? ?ŒìŠ¤???¤íŒ¨")
        sys.exit(1)
