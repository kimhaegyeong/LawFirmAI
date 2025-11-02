# -*- coding: utf-8 -*-
"""
Prompt Chaining 간단 테스트 (pytest 없이)
Phase 6, 7 구현 테스트
"""

import sys
import os
from unittest.mock import Mock, MagicMock

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_chain_executor():
    """PromptChainExecutor 기본 테스트"""
    try:
        from core.agents.prompt_chain_executor import PromptChainExecutor

        # LLM 모킹
        llm = Mock()
        llm.invoke = Mock(return_value=MagicMock(content='{"result": "success"}'))

        # Logger 모킹
        logger = Mock()

        executor = PromptChainExecutor(llm, logger)

        # 간단한 체인 정의
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

        print(f"✅ Chain execution test passed")
        print(f"   Success: {result['success']}")
        print(f"   Steps executed: {len(result['steps_executed'])}")
        print(f"   Validation: {result.get('validation_results', {}).get('is_valid', False)}")
        return True

    except Exception as e:
        print(f"❌ Chain execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chain_validation():
    """체인 검증 테스트"""
    try:
        from core.agents.prompt_chain_executor import PromptChainExecutor

        llm = Mock()
        llm.invoke = Mock(return_value=MagicMock(content='Valid output'))

        logger = Mock()
        executor = PromptChainExecutor(llm, logger)

        # None 출력 테스트 (검증 실패)
        validation = executor._validate_final_output(
            final_output=None,
            chain_steps=[],
            chain_history=[]
        )

        assert validation["is_valid"] is False
        assert validation["quality_score"] == 0.0
        print(f"✅ Chain validation test passed (None output)")

        # 유효한 출력 테스트
        validation = executor._validate_final_output(
            final_output="Valid output string",
            chain_steps=[],
            chain_history=[]
        )

        assert validation["is_valid"] is True
        assert validation["quality_score"] > 0.5
        print(f"✅ Chain validation test passed (Valid output)")
        return True

    except Exception as e:
        print(f"❌ Chain validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Chaining 테스트 시작")
    print("=" * 60)

    results = []

    # 테스트 1: Chain Executor
    print("\n[테스트 1] Chain Executor 테스트")
    results.append(test_chain_executor())

    # 테스트 2: Chain Validation
    print("\n[테스트 2] Chain Validation 테스트")
    results.append(test_chain_validation())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"총 테스트: {len(results)}")
    print(f"성공: {sum(results)}")
    print(f"실패: {len(results) - sum(results)}")

    if all(results):
        print("\n✅ 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트 실패")
        sys.exit(1)
