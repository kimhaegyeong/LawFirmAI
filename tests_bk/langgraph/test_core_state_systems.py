# -*- coding: utf-8 -*-
"""
?µì‹¬ State ?œìŠ¤???…ë¦½ ?ŒìŠ¤??
?˜ì¡´???†ì´ ?µì‹¬ ê¸°ëŠ¥ë§?ê²€ì¦?
"""

import logging
import sys
from pathlib import Path

# ë¡œê¹… ?¤ì • (?ëŸ¬ ?µì œ)
logging.getLogger().setLevel(logging.CRITICAL)

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.agents.node_input_output_spec import (
    NODE_SPECS,
    get_all_node_names,
    get_node_spec,
    validate_node_input,
    validate_workflow_flow,
)
from source.agents.state_adapter import StateAdapter, adapt_state, flatten_state
from source.agents.state_reduction import StateReducer


def test_node_specs_core():
    """?¸ë“œ ?¤í™ ?µì‹¬ ê²€ì¦?""
    print("=" * 80)
    print("1. ?¸ë“œ ?¤í™ ?µì‹¬ ê²€ì¦?)
    print("=" * 80)

    all_nodes = get_all_node_names()
    print(f"??ì´?{len(all_nodes)}ê°??¸ë“œ ?¤í™ ?•ì˜??)

    # ?„ìˆ˜ ?¸ë“œ ?•ì¸
    required = ["classify_query", "assess_urgency", "retrieve_documents", "generate_answer_enhanced"]
    for node in required:
        spec = get_node_spec(node)
        assert spec is not None, f"???¸ë“œ {node} ?†ìŒ"
        print(f"  ??{node}: {len(spec.required_input)}ê°??…ë ¥, {len(spec.output)}ê°?ì¶œë ¥")

    return True


def test_state_adapter_core():
    """State Adapter ?µì‹¬ ê²€ì¦?""
    print("\n" + "=" * 80)
    print("2. State Adapter ?µì‹¬ ê²€ì¦?)
    print("=" * 80)

    # Flat State
    flat_state = {
        "query": "?ŒìŠ¤??ì§ˆë¬¸",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85
    }

    # Flat ??Nested
    nested = adapt_state(flat_state)
    assert "input" in nested
    assert nested["input"]["query"] == flat_state["query"]
    print("??Flat ??Nested ë³€???±ê³µ")

    # Nested ??Flat
    flat_again = flatten_state(nested)
    assert flat_again["query"] == flat_state["query"]
    print("??Nested ??Flat ë³€???±ê³µ")

    # Round-trip
    assert flat_again["query"] == flat_state["query"]
    print("??Round-trip ê²€ì¦??±ê³µ")

    return True


def test_state_reduction_core():
    """State Reduction ?µì‹¬ ê²€ì¦?""
    print("\n" + "=" * 80)
    print("3. State Reduction ?µì‹¬ ê²€ì¦?)
    print("=" * 80)

    # ?€?©ëŸ‰ State
    full_state = {
        "query": "?ŒìŠ¤??,
        "session_id": "test",
        "retrieved_docs": [{"content": "test " * 100} for _ in range(20)],
        "processing_steps": [f"step_{i}" for i in range(50)]
    }

    reducer = StateReducer(aggressive_reduction=True)

    # ê°??¸ë“œë³?Reduction
    test_nodes = ["classify_query", "retrieve_documents"]
    for node_name in test_nodes:
        reduced = reducer.reduce_state_for_node(full_state, node_name)
        assert isinstance(reduced, dict), f"??{node_name} ?¤íŒ¨"
        print(f"  ??{node_name} State Reduction ?±ê³µ")

    return True


def test_workflow_validation_core():
    """?Œí¬?Œë¡œ??ê²€ì¦??µì‹¬"""
    print("\n" + "=" * 80)
    print("4. ?Œí¬?Œë¡œ??ê²€ì¦??µì‹¬")
    print("=" * 80)

    result = validate_workflow_flow()
    print(f"ì´?{result['total_nodes']}ê°??¸ë“œ")
    print(f"ê²€ì¦?ê²°ê³¼: {'??Valid' if result['valid'] else '? ï¸ Issues'}")

    if result['issues']:
        print(f"\n? ï¸ {len(result['issues'])}ê°œì˜ ?´ìŠˆ:")
        for issue in result['issues'][:3]:
            print(f"  - {issue}")

    return True


def main():
    """ë©”ì¸ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("LangGraph State ?œìŠ¤???µì‹¬ ?ŒìŠ¤??)
    print("=" * 80)

    tests = [
        ("?¸ë“œ ?¤í™", test_node_specs_core),
        ("State Adapter", test_state_adapter_core),
        ("State Reduction", test_state_reduction_core),
        ("?Œí¬?Œë¡œ??ê²€ì¦?, test_workflow_validation_core),
    ]

    results = []
    for name, func in tests:
        try:
            result = func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"??{name} ?¤íŒ¨: {e}")

    # ê²°ê³¼
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??ê²°ê³¼")
    print("=" * 80)

    passed = sum(1 for _, r, _ in results if r)
    failed = sum(1 for _, r, _ in results if not r)

    for name, result, error in results:
        status = "???µê³¼" if result else "???¤íŒ¨"
        error_msg = f" ({error})" if error else ""
        print(f"  {status}: {name}{error_msg}")

    print(f"\nì´?{len(results)}ê°??ŒìŠ¤??ì¤?{passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")

    if failed == 0:
        print("\n?‰ ëª¨ë“  ?µì‹¬ ?ŒìŠ¤???µê³¼!")
        return True
    else:
        print(f"\n? ï¸ {failed}ê°??ŒìŠ¤???¤íŒ¨")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
