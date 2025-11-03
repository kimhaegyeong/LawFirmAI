# -*- coding: utf-8 -*-
"""
ëª¨ë‹ˆ?°ë§ ?„í™˜ ê¸°ë³¸ ?ŒìŠ¤??
?˜ê²½ë³€???„í™˜ ë¡œì§ë§??ŒìŠ¤??(?Œí¬?Œë¡œ???˜ì¡´???†ìŒ)
"""

import os
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch


def test_mode_set_and_restore():
    """?˜ê²½ë³€???¤ì • ë°?ë³µì› ?ŒìŠ¤??""
    print("="*80)
    print("?ŒìŠ¤??1: ?˜ê²½ë³€???¤ì • ë°?ë³µì›")
    print("="*80)

    # ?ë³¸ ê°??€??
    original_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    original_langfuse = os.environ.get("LANGFUSE_ENABLED")
    original_api_key = os.environ.get("LANGCHAIN_API_KEY")

    # LangSmith ëª¨ë“œë¡??¤ì • (API ?¤ë„ ?¨ê»˜ ?¤ì •)
    with MonitoringSwitch.set_mode(
        MonitoringMode.LANGSMITH,
        langsmith_api_key="test-api-key-for-testing"
    ):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LangSmith ?¸ë ˆ?´ì‹± ?œì„±???•ì¸"
        assert os.environ.get("LANGFUSE_ENABLED") == "false", "Langfuse ë¹„í™œ?±í™” ?•ì¸"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.LANGSMITH, f"?„ì¬ ëª¨ë“œ??LANGSMITH?¬ì•¼ ?? {current_mode}"
        print("??LangSmith ëª¨ë“œ ?¤ì • ?•ì¸")

    # ë³µì› ?•ì¸
    assert os.environ.get("LANGCHAIN_TRACING_V2") == original_tracing, "?˜ê²½ë³€??ë³µì› ?•ì¸"
    assert os.environ.get("LANGFUSE_ENABLED") == original_langfuse, "?˜ê²½ë³€??ë³µì› ?•ì¸"
    print("???˜ê²½ë³€??ë³µì› ?•ì¸")

    return True


def test_langfuse_mode():
    """Langfuse ëª¨ë“œ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ŒìŠ¤??2: Langfuse ëª¨ë“œ ?¤ì •")
    print("="*80)

    with MonitoringSwitch.set_mode(MonitoringMode.LANGFUSE):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false", "LangSmith ë¹„í™œ?±í™” ?•ì¸"
        assert os.environ.get("LANGFUSE_ENABLED") == "true", "Langfuse ?œì„±???•ì¸"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.LANGFUSE, f"?„ì¬ ëª¨ë“œ??LANGFUSE?¬ì•¼ ?? {current_mode}"
        print("??Langfuse ëª¨ë“œ ?¤ì • ?•ì¸")

    return True


def test_both_mode():
    """Both ëª¨ë“œ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ŒìŠ¤??3: Both ëª¨ë“œ ?¤ì •")
    print("="*80)

    with MonitoringSwitch.set_mode(
        MonitoringMode.BOTH,
        langsmith_api_key="test-api-key-for-testing"
    ):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LangSmith ?œì„±???•ì¸"
        assert os.environ.get("LANGFUSE_ENABLED") == "true", "Langfuse ?œì„±???•ì¸"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.BOTH, f"?„ì¬ ëª¨ë“œ??BOTH?¬ì•¼ ?? {current_mode}"
        print("??Both ëª¨ë“œ ?¤ì • ?•ì¸")

    return True


def test_none_mode():
    """None ëª¨ë“œ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ŒìŠ¤??4: None ëª¨ë“œ ?¤ì •")
    print("="*80)

    with MonitoringSwitch.set_mode(MonitoringMode.NONE):
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false", "LangSmith ë¹„í™œ?±í™” ?•ì¸"
        assert os.environ.get("LANGFUSE_ENABLED") == "false", "Langfuse ë¹„í™œ?±í™” ?•ì¸"
        current_mode = MonitoringSwitch.get_current_mode()
        assert current_mode == MonitoringMode.NONE, f"?„ì¬ ëª¨ë“œ??NONE?´ì–´???? {current_mode}"
        print("??None ëª¨ë“œ ?¤ì • ?•ì¸")

    return True


def test_mode_switching():
    """ëª¨ë“œ ?„í™˜ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ŒìŠ¤??5: ?œì°¨??ëª¨ë“œ ?„í™˜")
    print("="*80)

    modes_to_test = [
        (MonitoringMode.LANGSMITH, {"langsmith_api_key": "test-api-key"}),
        (MonitoringMode.LANGFUSE, {}),
        (MonitoringMode.BOTH, {"langsmith_api_key": "test-api-key"}),
        (MonitoringMode.NONE, {}),
    ]

    for mode, kwargs in modes_to_test:
        with MonitoringSwitch.set_mode(mode, **kwargs):
            current = MonitoringSwitch.get_current_mode()
            assert current == mode, f"ëª¨ë“œ ?„í™˜ ?¤íŒ¨: ?ˆìƒ={mode.value}, ?¤ì œ={current.value}"
            print(f"  ??{mode.value} ëª¨ë“œ ?•ì¸")

    print("??ëª¨ë“  ëª¨ë“œ ?„í™˜ ?±ê³µ")
    return True


def test_mode_from_string():
    """ë¬¸ì?´ì—??ëª¨ë“œ ?ì„± ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ŒìŠ¤??6: ë¬¸ì?´ì—??ëª¨ë“œ ?ì„±")
    print("="*80)

    test_cases = [
        ("langsmith", MonitoringMode.LANGSMITH),
        ("LANGSMITH", MonitoringMode.LANGSMITH),
        ("langfuse", MonitoringMode.LANGFUSE),
        ("both", MonitoringMode.BOTH),
        ("none", MonitoringMode.NONE),
    ]

    for string_value, expected_mode in test_cases:
        mode = MonitoringMode.from_string(string_value)
        assert mode == expected_mode, f"ë¬¸ì??'{string_value}' ?Œì‹± ?¤íŒ¨"
        print(f"  ??'{string_value}' -> {mode.value}")

    # ?˜ëª»??ê°??ŒìŠ¤??
    try:
        MonitoringMode.from_string("invalid")
        assert False, "?˜ëª»??ê°’ì— ?€???ˆì™¸ê°€ ë°œìƒ?´ì•¼ ??
    except ValueError:
        print("  ???˜ëª»??ê°’ì— ?€???ˆì™¸ ì²˜ë¦¬ ?•ì¸")

    return True


def main():
    """ëª¨ë“  ?ŒìŠ¤???¤í–‰"""
    print("\n" + "="*80)
    print("ëª¨ë‹ˆ?°ë§ ?„í™˜ ê¸°ë³¸ ?ŒìŠ¤???œì‘")
    print("="*80)

    tests = [
        ("?˜ê²½ë³€???¤ì • ë°?ë³µì›", test_mode_set_and_restore),
        ("Langfuse ëª¨ë“œ", test_langfuse_mode),
        ("Both ëª¨ë“œ", test_both_mode),
        ("None ëª¨ë“œ", test_none_mode),
        ("ëª¨ë“œ ?„í™˜", test_mode_switching),
        ("ë¬¸ì?´ì—??ëª¨ë“œ ?ì„±", test_mode_from_string),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n??{test_name} ?¤íŒ¨: {e}")
            import traceback
            traceback.print_exc()

    # ê²°ê³¼ ?”ì•½
    print("\n" + "="*80)
    print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("="*80)

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
