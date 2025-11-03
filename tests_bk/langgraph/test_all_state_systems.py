# -*- coding: utf-8 -*-
"""
LangGraph State ?œìŠ¤???„ì²´ ?µí•© ?ŒìŠ¤??
ëª¨ë“  ê¸°ëŠ¥????ë²ˆì— ê²€ì¦?
"""

import os
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env ?Œì¼ ë¡œë“œ (python-dotenv ?¨í‚¤ì§€ ?„ìš”)
try:
    from dotenv import load_dotenv
    # ?„ë¡œ?íŠ¸ ë£¨íŠ¸?ì„œ .env ?Œì¼ ë¡œë“œ
    load_dotenv(dotenv_path=str(project_root / ".env"))
except ImportError:
    # python-dotenvê°€ ?¤ì¹˜?˜ì? ?Šì? ê²½ìš° ê²½ê³ ë§?ì¶œë ¥?˜ê³  ê³„ì† ì§„í–‰
    pass

# LangSmith ëª¨ë‹ˆ?°ë§ ?¤ì • (? íƒ ?¬í•­)
# .env ?Œì¼ ?ëŠ” ?˜ê²½ ë³€?˜ì—???¤ìŒ ë³€?˜ë“¤???¤ì •?˜ì„¸??
#
# ?„ìˆ˜ (LangChain ?œì? ?˜ê²½ë³€??:
#   LANGCHAIN_TRACING_V2=true          # LangSmith ?¸ë ˆ?´ì‹± ?œì„±??
#   LANGCHAIN_API_KEY=your-api-key     # LangSmith API ??
#   LANGCHAIN_PROJECT=LawFirmAI-Test   # LangSmith ?„ë¡œ?íŠ¸ ?´ë¦„ (? íƒ?¬í•­)
#
# ? íƒ (?˜ìœ„ ?¸í™˜??:
#   ENABLE_LANGSMITH=true              # ì¶”ê? ?œì„±???Œë˜ê·?(? íƒ?¬í•­)
#   LANGSMITH_API_KEY=...              # LANGCHAIN_API_KEY ?€???¬ìš© ê°€??
#   LANGSMITH_PROJECT=...              # LANGCHAIN_PROJECT ?€???¬ìš© ê°€??

# LangSmith ?¤ì • ?½ê¸° (.env ?Œì¼?ì„œ ?´ë? ë¡œë“œ??
# ?œì? LangChain ?˜ê²½ë³€???°ì„  ?¬ìš©
langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY", "") or os.environ.get("LANGSMITH_API_KEY", "")
langsmith_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower()
langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "") or os.environ.get("LANGSMITH_PROJECT", "LawFirmAI-Test")
enable_langsmith_flag = os.environ.get("ENABLE_LANGSMITH", "false").lower() == "true"

# LangSmith ?œì„±???¬ë? ?•ì¸
# LANGCHAIN_TRACING_V2ê°€ 'true'?´ê³  API ?¤ê? ?ˆìœ¼ë©??œì„±??
langsmith_enabled = (
    langsmith_tracing in ["true", "1", "yes"]
    and bool(langsmith_api_key)
)

# ENABLE_LANGSMITH ?Œë˜ê·¸ê? ?¤ì •??ê²½ìš°?ë„ ?œì„±??(?˜ìœ„ ?¸í™˜??
if enable_langsmith_flag and langsmith_tracing not in ["true", "1", "yes"]:
    # ?Œë˜ê·¸ë§Œ ?ˆê³  TRACING_V2ê°€ ?†ìœ¼ë©?ê²½ê³ 
    print("??ENABLE_LANGSMITH=trueê°€ ?¤ì •?˜ì—ˆì§€ë§?LANGCHAIN_TRACING_V2ê°€ ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
    print("  LangSmithë¥??œì„±?”í•˜?¤ë©´ LANGCHAIN_TRACING_V2=trueë¥??¤ì •?˜ì„¸??\n")
    langsmith_enabled = False

if langsmith_enabled:
    print("=" * 80)
    print("LangSmith ëª¨ë‹ˆ?°ë§ ?œì„±?”ë¨")
    print("=" * 80)
    if langsmith_api_key:
        # API ??ë¶€ë¶„ë§Œ ?œì‹œ (ë³´ì•ˆ)
        if len(langsmith_api_key) > 30:
            print(f"  API Key: {langsmith_api_key[:15]}...{langsmith_api_key[-10:]} (ë¶€ë¶??œì‹œ)")
        else:
            print(f"  API Key: {'*' * min(len(langsmith_api_key), 20)}... (?¤ì •??")
    print(f"  Project: {langsmith_project}")
    print(f"  Tracing: ?œì„±??(LANGCHAIN_TRACING_V2={langsmith_tracing})")
    if enable_langsmith_flag:
        print("  ENABLE_LANGSMITH: ?œì„±?”ë¨")
    print("=" * 80 + "\n")
else:
    # ë¹„í™œ?±í™”??ê²½ìš° ?ì„¸???ˆë‚´ ë©”ì‹œì§€ ì¶œë ¥
    print("??LangSmith ëª¨ë‹ˆ?°ë§ ë¹„í™œ?±í™”??(ê¸°ë³¸ê°?")

    missing_config = []
    if langsmith_tracing not in ["true", "1", "yes"]:
        missing_config.append("LANGCHAIN_TRACING_V2=true")
    if not langsmith_api_key:
        missing_config.append("LANGCHAIN_API_KEY=your-api-key")

    if missing_config:
        print("  LangSmithë¥??œì„±?”í•˜?¤ë©´ .env ?Œì¼???¤ìŒ ?˜ê²½ ë³€?˜ë? ?¤ì •?˜ì„¸??")
        for config in missing_config:
            print(f"    {config}")
        if not langsmith_project or langsmith_project == "LawFirmAI-Test":
            print("    LANGCHAIN_PROJECT=LawFirmAI-Test  # (? íƒ?¬í•­)")
        print()
    else:
        print("  (?¤ì •?€ ?˜ì–´ ?ˆì?ë§??œì„±??ì¡°ê±´??ë§Œì¡±?˜ì? ?ŠìŠµ?ˆë‹¤)\n")

from typing import Any, Dict

# ?µì‹¬ ëª¨ë“ˆ import
from source.agents.node_input_output_spec import (
    NODE_SPECS,
    get_all_node_names,
    get_node_spec,
    validate_node_input,
    validate_workflow_flow,
)
from source.agents.state_adapter import StateAdapter, adapt_state, flatten_state
from source.agents.state_reduction import StateReducer, reduce_state_for_node


def test_node_specs():
    """?¸ë“œ ?¤í™ ê²€ì¦?""
    print("=" * 80)
    print("1. ?¸ë“œ ?¤í™ ê²€ì¦?)
    print("=" * 80)

    all_nodes = get_all_node_names()
    print(f"??ì´?{len(all_nodes)}ê°??¸ë“œ ?¤í™ ?•ì˜??)

    for node_name in all_nodes:
        spec = get_node_spec(node_name)
        print(f"  - {node_name}: {len(spec.required_input)}ê°??…ë ¥, {len(spec.output)}ê°?ì¶œë ¥")

    return True


def test_state_adapter():
    """State Adapter ê²€ì¦?""
    print("\n" + "=" * 80)
    print("2. State Adapter ê²€ì¦?)
    print("=" * 80)

    # Flat State ?ì„±
    flat_state = {
        "query": "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­?€?",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85,
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
        "processing_steps": [],
        "errors": []
    }

    # Flat ??Nested ë³€??
    nested_state = adapt_state(flat_state)
    assert "input" in nested_state, "??Flat ??Nested ë³€???¤íŒ¨"
    assert nested_state["input"]["query"] == flat_state["query"]
    print("??Flat ??Nested ë³€???±ê³µ")

    # Nested ??Flat ë³€??
    flat_again = flatten_state(nested_state)
    assert flat_again["query"] == flat_state["query"]
    print("??Nested ??Flat ë³€???±ê³µ")

    # Round-trip ê²€ì¦?
    assert flat_again["query"] == flat_state["query"]
    assert flat_again["query_type"] == flat_state["query_type"]
    print("??Round-trip ë³€??ê²€ì¦??±ê³µ")

    return True


def test_state_reduction():
    """State Reduction ê²€ì¦?""
    print("\n" + "=" * 80)
    print("3. State Reduction ê²€ì¦?)
    print("=" * 80)

    # ?€?©ëŸ‰ State ?ì„±
    full_state = {
        "query": "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­?€?",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85,
        "retrieved_docs": [
            {"content": "test " * 100, "source": f"doc_{i}"}
            for i in range(20)
        ],
        "answer": "?µë? ?´ìš©?…ë‹ˆ??,
        "sources": [],
        "processing_steps": [f"step_{i}" for i in range(50)],
        "errors": []
    }

    reducer = StateReducer(aggressive_reduction=True)

    # ê°??¸ë“œë³„ë¡œ State Reduction
    nodes_to_test = [
        "classify_query",
        "assess_urgency",
        "retrieve_documents",
        "generate_answer_enhanced"
    ]

    for node_name in nodes_to_test:
        reduced = reducer.reduce_state_for_node(full_state, node_name)
        assert isinstance(reduced, dict), f"??{node_name} State Reduction ?¤íŒ¨"

        reduction_info = ""
        if "input" in reduced and "query" in reduced.get("input", {}):
            reduction_info = f" (reduced)"

        print(f"  ??{node_name} State Reduction ?±ê³µ{reduction_info}")

    return True


def test_workflow_validation():
    """?Œí¬?Œë¡œ??ê²€ì¦?""
    print("\n" + "=" * 80)
    print("4. ?Œí¬?Œë¡œ??ê²€ì¦?)
    print("=" * 80)

    result = validate_workflow_flow()

    print(f"ì´?{result['total_nodes']}ê°??¸ë“œ")
    print(f"ê²€ì¦?ê²°ê³¼: {'??Valid' if result['valid'] else '? ï¸ Issues found'}")

    if result['issues']:
        print(f"\n? ï¸ {len(result['issues'])}ê°œì˜ ?´ìŠˆ ë°œê²¬:")
        for issue in result['issues'][:5]:
            print(f"  - {issue}")

    return True


def test_node_input_validation():
    """?¸ë“œ Input ê²€ì¦?""
    print("\n" + "=" * 80)
    print("5. ?¸ë“œ Input ê²€ì¦?)
    print("=" * 80)

    # ? íš¨???…ë ¥
    valid_state = {
        "query": "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­?€?",
        "session_id": "test_123"
    }
    is_valid, error = validate_node_input("classify_query", valid_state)
    assert is_valid, f"??? íš¨???…ë ¥??ê±°ë??? {error}"
    print("??? íš¨???…ë ¥ ê²€ì¦??±ê³µ")

    # ? íš¨?˜ì? ?Šì? ?…ë ¥
    invalid_state = {
        "session_id": "test_123"
        # query ?„ë½
    }
    is_valid, error = validate_node_input("classify_query", invalid_state)
    assert not is_valid, "??? íš¨?˜ì? ?Šì? ?…ë ¥???µê³¼??
    print("??? íš¨?˜ì? ?Šì? ?…ë ¥ ê±°ë? ?±ê³µ")

    return True


def test_all_nodes():
    """ëª¨ë“  ?¸ë“œ ê²€ì¦?""
    print("\n" + "=" * 80)
    print("6. ëª¨ë“  ?¸ë“œ ê²€ì¦?)
    print("=" * 80)

    all_nodes = get_all_node_names()

    # ?„ìˆ˜ ?¸ë“œ ?•ì¸
    required_nodes = [
        "classify_query",
        "assess_urgency",
        "resolve_multi_turn",
        "route_expert",
        "retrieve_documents",
        "generate_answer_enhanced",
        "validate_answer_quality"
    ]

    for node in required_nodes:
        assert node in all_nodes, f"???„ìˆ˜ ?¸ë“œ {node}ê°€ ?†ìŠµ?ˆë‹¤"
        spec = get_node_spec(node)
        assert spec is not None, f"??{node}???€???¤í™???†ìŠµ?ˆë‹¤"
        print(f"  ??{node}")

    print(f"\n???„ì²´ {len(all_nodes)}ê°??¸ë“œ ëª¨ë‘ ?•ìƒ")

    return True


def main():
    """ë©”ì¸ ?ŒìŠ¤???¤í–‰"""
    print("\n" + "=" * 80)
    print("LangGraph State ?œìŠ¤???„ì²´ ?µí•© ?ŒìŠ¤??)
    print("=" * 80)

    tests = [
        ("?¸ë“œ ?¤í™ ê²€ì¦?, test_node_specs),
        ("State Adapter ê²€ì¦?, test_state_adapter),
        ("State Reduction ê²€ì¦?, test_state_reduction),
        ("?Œí¬?Œë¡œ??ê²€ì¦?, test_workflow_validation),
        ("?¸ë“œ Input ê²€ì¦?, test_node_input_validation),
        ("ëª¨ë“  ?¸ë“œ ê²€ì¦?, test_all_nodes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n??{test_name} ?¤íŒ¨: {e}")

    # ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("=" * 80)

    passed = sum(1 for _, result, _ in results if result)
    failed = sum(1 for _, result, _ in results if not result)

    for test_name, result, error in results:
        status = "???µê³¼" if result else "???¤íŒ¨"
        if error:
            print(f"  {status}: {test_name} ({error})")
        else:
            print(f"  {status}: {test_name}")

    print(f"\nì´?{len(results)}ê°??ŒìŠ¤??ì¤?{passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")

    if failed == 0:
        print("\n?‰ ëª¨ë“  ?ŒìŠ¤???µê³¼!")
    else:
        print(f"\n? ï¸ {failed}ê°??ŒìŠ¤???¤íŒ¨")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
