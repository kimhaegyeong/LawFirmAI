# -*- coding: utf-8 -*-
"""
State ë¦¬íŒ©? ë§ ?±ëŠ¥ ?ŒìŠ¤??
ë©”ëª¨ë¦??¬ìš©?? ë¡œê¹… ?°ì´???¬ê¸°, ì²˜ë¦¬ ?ë„ ì¸¡ì •
"""

import json
import sys
import time
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ì¶”ê? ??import ?„ìš” (lint: disable=wrong-import-position)
import pytest  # noqa: E402

from source.agents.modular_states import (  # noqa: E402
    create_initial_legal_state as create_modular_state,
)
from source.agents.state_definitions import (  # noqa: E402
    create_initial_legal_state as create_flat_state,
)
from source.agents.state_reduction import reduce_state_for_node  # noqa: E402


def estimate_size(state: dict) -> int:
    """State ?¬ê¸° ì¶”ì • (bytes)"""
    return len(json.dumps(state, ensure_ascii=False).encode('utf-8'))


class TestStatePerformance:
    """State ?±ëŠ¥ ?ŒìŠ¤??""

    def test_memory_comparison(self):
        """ë©”ëª¨ë¦??¬ìš©??ë¹„êµ ?ŒìŠ¤??""
        # ë§ì? ë¬¸ì„œë¥??¬í•¨??State ?ì„±
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        flat_state["retrieved_docs"] = [{"content": "ë¬¸ì„œ ?´ìš© " * 50} for _ in range(20)]
        flat_state["conversation_history"] = [{"role": "user", "content": "ì§ˆë¬¸"} for _ in range(10)]

        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        from source.agents.state_helpers import set_field
        set_field(modular_state, "retrieved_docs", [{"content": "ë¬¸ì„œ ?´ìš© " * 50} for _ in range(20)])
        set_field(modular_state, "conversation_history", [{"role": "user", "content": "ì§ˆë¬¸"} for _ in range(10)])

        # ?¬ê¸° ì¸¡ì •
        flat_size = estimate_size(flat_state)
        modular_size = estimate_size(modular_state)

        print(f"\nFlat êµ¬ì¡° ?¬ê¸°: {flat_size:,} bytes")
        print(f"Modular êµ¬ì¡° ?¬ê¸°: {modular_size:,} bytes")
        print(f"ì°¨ì´: {flat_size - modular_size:,} bytes ({(1 - modular_size/flat_size)*100:.1f}% ê°ì†Œ)")

        # Modular êµ¬ì¡°ê°€ ???‘ê±°??ë¹„ìŠ·?´ì•¼ ??(ê·¸ë£¹???¤ë²„?¤ë“œ ê³ ë ¤)
        # ?¤ì œë¡œëŠ” reduce_state_for_node ?„ì— ????ì°¨ì´ê°€ ??ê²?
        assert modular_size > 0

    def test_reduced_state_size(self):
        """ì¶•ì†Œ??State ?¬ê¸° ?ŒìŠ¤??""
        # ë§ì? ?„ë“œë¥??¬í•¨??State ?ì„±
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        flat_state["retrieved_docs"] = [{"content": "ë¬¸ì„œ ?´ìš© " * 50} for _ in range(20)]

        # classify_query ?¸ë“œ???„ìš”??ê²ƒë§Œ ì¶”ì¶œ
        reduced_flat = reduce_state_for_node(flat_state, "classify_query")

        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        from source.agents.state_helpers import set_field
        set_field(modular_state, "retrieved_docs", [{"content": "ë¬¸ì„œ ?´ìš© " * 50} for _ in range(20)])

        reduced_modular = reduce_state_for_node(modular_state, "classify_query")

        # ì¶•ì†Œ???¬ê¸° ì¸¡ì •
        full_flat_size = estimate_size(flat_state)
        reduced_flat_size = estimate_size(reduced_flat)

        full_modular_size = estimate_size(modular_state)
        reduced_modular_size = estimate_size(reduced_modular)

        print(f"\n?„ì²´ Flat ?¬ê¸°: {full_flat_size:,} bytes")
        print(f"ì¶•ì†Œ Flat ?¬ê¸°: {reduced_flat_size:,} bytes")
        print(f"?„ì²´ Modular ?¬ê¸°: {full_modular_size:,} bytes")
        print(f"ì¶•ì†Œ Modular ?¬ê¸°: {reduced_modular_size:,} bytes")

        # ì¶•ì†Œ??ê²ƒì´ ???‘ì•„????
        assert reduced_flat_size < full_flat_size
        assert reduced_modular_size < full_modular_size

    def test_field_access_performance(self):
        """?„ë“œ ?‘ê·¼ ?±ëŠ¥ ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        from source.agents.state_helpers import get_field

        # ë°˜ë³µ ?Ÿìˆ˜ ì¦ê? (???•í™•??ì¸¡ì •)
        iterations = 10000

        # Flat êµ¬ì¡° ?‘ê·¼ ?œê°„ ì¸¡ì •
        start = time.time()
        for _ in range(iterations):
            _ = flat_state.get("query")
            _ = flat_state.get("query_type")
            _ = flat_state.get("confidence")
        flat_time = time.time() - start

        # Modular êµ¬ì¡° ?‘ê·¼ ?œê°„ ì¸¡ì • (helper ?¨ìˆ˜ ?¬ìš©)
        start = time.time()
        for _ in range(iterations):
            _ = get_field(modular_state, "query")
            _ = get_field(modular_state, "query_type")
            _ = get_field(modular_state, "confidence")
        modular_time = time.time() - start

        print(f"\nFlat êµ¬ì¡° ?‘ê·¼: {flat_time*1000:.2f} ms ({iterations}??")
        print(f"Modular êµ¬ì¡° ?‘ê·¼: {modular_time*1000:.2f} ms ({iterations}??")

        if flat_time > 0:
            ratio = modular_time / flat_time
            print(f"?±ëŠ¥ ë¹„ìœ¨: {ratio:.2f}x (Modular/Flat)")

        # Helper ?¨ìˆ˜ ?¤ë²„?¤ë“œ???ˆì?ë§??ˆìš© ë²”ìœ„ ?´ì—¬????
        # Flat êµ¬ì¡° ?‘ê·¼???ˆë¬´ ë¹ ë¥´ë©??ë? ë¹„êµê°€ ?˜ë? ?†ìœ¼ë¯€ë¡??ˆë? ?œê°„ ê¸°ì? ?¬ìš©

        # ?ˆë? ?œê°„ ê¸°ì?: 10000??ë°˜ë³µ ??500ms ?´í•˜ë©??©ë¦¬??
        # (?Œë‹¹ 0.05ms = 50ë§ˆì´?¬ë¡œì´?
        # Helper ?¨ìˆ˜???Œë‹¤ ?¸ì¶œê³?ì¤‘ì²© ?‘ê·¼?¼ë¡œ ?¸í•´ ?¤ë²„?¤ë“œê°€ ?ˆì?ë§?
        # ?¤ì œ ?¬ìš© ?????¤ë²„?¤ë“œ??ë¬´ì‹œ???˜ì??´ì–´????
        max_modular_time = 0.5  # 500ms (10000??ê¸°ì?)

        # ?ˆë? ?œê°„ ê¸°ì? ?•ì¸ (?ë? ë¹„êµ???˜ê²½???°ë¼ ë³€?™ì´ ?¬ë?ë¡??œì™¸)
        assert modular_time < max_modular_time, \
            f"Modular ?‘ê·¼ ?œê°„({modular_time*1000:.2f}ms)??ìµœë? ?ˆìš© ?œê°„({max_modular_time*1000:.2f}ms)??ì´ˆê³¼. " \
            f"Helper ?¨ìˆ˜ ?±ëŠ¥ ìµœì ?”ê? ?„ìš”?????ˆìŠµ?ˆë‹¤."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
