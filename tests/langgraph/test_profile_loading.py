# -*- coding: utf-8 -*-
"""
?„ë¡œ??ë¡œë“œ ê¸°ëŠ¥ ?ŒìŠ¤??
"""

import os
import sys
import tempfile
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringSwitch


def test_profile_loading():
    """?„ë¡œ??ë¡œë“œ ?ŒìŠ¤??""
    print("="*80)
    print("?„ë¡œ??ë¡œë“œ ê¸°ëŠ¥ ?ŒìŠ¤??)
    print("="*80)

    # ?„ì‹œ ?„ë¡œ???Œì¼ ?ì„±
    profiles_dir = project_root / ".env.profiles"
    profiles_dir.mkdir(exist_ok=True)

    test_profile_content = """
# ?ŒìŠ¤???„ë¡œ??
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=test-api-key-12345
LANGCHAIN_PROJECT=test-project
LANGFUSE_ENABLED=false
ENABLE_LANGSMITH=true
"""

    test_profile_path = profiles_dir / "test_profile.env"
    with open(test_profile_path, 'w', encoding='utf-8') as f:
        f.write(test_profile_content)

    # ?„ë¡œ??ë¡œë“œ
    env_vars = MonitoringSwitch.load_profile("test_profile")

    # ê²€ì¦?
    assert len(env_vars) == 5, f"?„ë¡œ?„ì—??5ê°??˜ê²½ë³€?˜ë? ?½ì–´???? {len(env_vars)}"
    assert env_vars.get("LANGCHAIN_TRACING_V2") == "true"
    assert env_vars.get("LANGCHAIN_API_KEY") == "test-api-key-12345"
    assert env_vars.get("LANGCHAIN_PROJECT") == "test-project"
    assert env_vars.get("LANGFUSE_ENABLED") == "false"
    assert env_vars.get("ENABLE_LANGSMITH") == "true"

    print("???„ë¡œ??ë¡œë“œ ?±ê³µ")
    print(f"   ë¡œë“œ???˜ê²½ë³€?? {list(env_vars.keys())}")

    # ì¡´ì¬?˜ì? ?ŠëŠ” ?„ë¡œ???ŒìŠ¤??
    non_existent = MonitoringSwitch.load_profile("non_existent_profile")
    assert len(non_existent) == 0, "ì¡´ì¬?˜ì? ?ŠëŠ” ?„ë¡œ?„ì? ë¹??•ì…”?ˆë¦¬ ë°˜í™˜"
    print("??ì¡´ì¬?˜ì? ?ŠëŠ” ?„ë¡œ??ì²˜ë¦¬ ?•ì¸")

    # ?•ë¦¬
    if test_profile_path.exists():
        test_profile_path.unlink()

    return True


if __name__ == "__main__":
    try:
        result = test_profile_loading()
        print("\n" + "="*80)
        if result:
            print("?‰ ?„ë¡œ??ë¡œë“œ ?ŒìŠ¤???µê³¼!")
        else:
            print("???„ë¡œ??ë¡œë“œ ?ŒìŠ¤???¤íŒ¨")
        print("="*80)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n???ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
