# -*- coding: utf-8 -*-
"""
State Refactoring ?µí•© ?ŒìŠ¤??
Flat ??Modular ë³€??ë°??¸í™˜???ŒìŠ¤??
"""

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from source.agents.modular_states import (
    create_initial_legal_state as create_modular_state,
)
from source.agents.state_adapter import StateAdapter
from source.agents.state_definitions import (
    create_initial_legal_state as create_flat_state,
)
from source.agents.state_helpers import (
    get_field,
    is_modular_state,
    set_field,
)
from source.agents.state_reduction import (
    reduce_state_for_node,
    reduce_state_size,
)


class TestStateRefactoring:
    """State ë¦¬íŒ©? ë§ ?µí•© ?ŒìŠ¤??""

    def test_create_flat_state(self):
        """Flat êµ¬ì¡° ?ì„± ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        assert "query" in flat_state
        assert "session_id" in flat_state
        assert flat_state["query"] == "?ŒìŠ¤??ì§ˆë¬¸"
        assert not is_modular_state(flat_state)

    def test_create_modular_state(self):
        """Modular êµ¬ì¡° ?ì„± ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        assert "input" in modular_state
        assert "classification" in modular_state
        assert modular_state["input"]["query"] == "?ŒìŠ¤??ì§ˆë¬¸"
        assert is_modular_state(modular_state)

    def test_flat_to_modular_conversion(self):
        """Flat ??Modular ë³€???ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        flat_state["query_type"] = "legal_advice"
        flat_state["confidence"] = 0.9

        modular_state = StateAdapter.to_nested(flat_state)

        assert is_modular_state(modular_state)
        assert get_field(modular_state, "query") == "?ŒìŠ¤??ì§ˆë¬¸"
        assert get_field(modular_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "confidence") == 0.9

    def test_modular_to_flat_conversion(self):
        """Modular ??Flat ë³€???ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        set_field(modular_state, "query_type", "legal_advice")
        set_field(modular_state, "confidence", 0.9)

        flat_state = StateAdapter.to_flat(modular_state)

        assert not is_modular_state(flat_state)
        assert flat_state["query"] == "?ŒìŠ¤??ì§ˆë¬¸"
        assert flat_state["query_type"] == "legal_advice"
        assert flat_state["confidence"] == 0.9

    def test_field_access_compatibility(self):
        """?„ë“œ ?‘ê·¼ ?¸í™˜???ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # ê°™ì? ?„ë“œ??ê°™ì? ê°??¤ì •
        set_field(flat_state, "query_type", "legal_advice")
        set_field(modular_state, "query_type", "legal_advice")

        # ê°™ì? ê°?ê°€?¸ì˜¤ê¸?
        assert get_field(flat_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "query_type") == "legal_advice"
        assert get_field(flat_state, "query") == "?ŒìŠ¤??ì§ˆë¬¸"
        assert get_field(modular_state, "query") == "?ŒìŠ¤??ì§ˆë¬¸"

    def test_state_reduction_flat(self):
        """Flat êµ¬ì¡° State Reduction ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        flat_state["query_type"] = "legal_advice"
        flat_state["retrieved_docs"] = [{"content": "doc1"}, {"content": "doc2"}]

        # classify_query ?¸ë“œ???„ìš”???„ë“œë§?ì¶”ì¶œ
        reduced = reduce_state_for_node(flat_state, "classify_query")

        assert "query" in reduced or "input" in reduced
        assert "query_type" in reduced or "classification" in reduced

    def test_state_reduction_modular(self):
        """Modular êµ¬ì¡° State Reduction ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        set_field(modular_state, "query_type", "legal_advice")
        set_field(modular_state, "retrieved_docs", [{"content": "doc1"}, {"content": "doc2"}])

        # classify_query ?¸ë“œ???„ìš”??ê·¸ë£¹ë§?ì¶”ì¶œ
        reduced = reduce_state_for_node(modular_state, "classify_query")

        assert "input" in reduced
        assert "classification" in reduced
        assert "common" in reduced

    def test_state_size_reduction_modular(self):
        """Modular êµ¬ì¡°?ì„œ State ?¬ê¸° ?œí•œ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # ë§ì? ë¬¸ì„œ ì¶”ê?
        large_docs = [{"content": "doc" * 100} for _ in range(20)]
        set_field(modular_state, "retrieved_docs", large_docs)

        # ?¬ê¸° ?œí•œ ?ìš©
        reduced = reduce_state_size(modular_state, max_docs=10, max_content_per_doc=500)

        # Modular êµ¬ì¡° ? ì? ?•ì¸
        assert is_modular_state(reduced)

        # ë¬¸ì„œ ???œí•œ ?•ì¸
        final_docs = get_field(reduced, "retrieved_docs")
        assert len(final_docs) <= 10

    def test_state_size_reduction_flat(self):
        """Flat êµ¬ì¡°?ì„œ State ?¬ê¸° ?œí•œ ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # ë§ì? ë¬¸ì„œ ì¶”ê?
        large_docs = [{"content": "doc" * 100} for _ in range(20)]
        flat_state["retrieved_docs"] = large_docs

        # ?¬ê¸° ?œí•œ ?ìš©
        reduced = reduce_state_size(flat_state, max_docs=10, max_content_per_doc=500)

        # Flat êµ¬ì¡° ? ì? ?•ì¸
        assert not is_modular_state(reduced)

        # ë¬¸ì„œ ???œí•œ ?•ì¸
        assert len(reduced["retrieved_docs"]) <= 10

    def test_round_trip_conversion(self):
        """Round-trip ë³€???ŒìŠ¤??(Flat ??Modular ??Flat)"""
        original_flat = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        original_flat["query_type"] = "legal_advice"
        original_flat["confidence"] = 0.9
        original_flat["retrieved_docs"] = [{"content": "doc1"}]

        # Flat ??Modular
        modular = StateAdapter.to_nested(original_flat)
        assert is_modular_state(modular)

        # Modular ??Flat
        back_to_flat = StateAdapter.to_flat(modular)

        # ì£¼ìš” ?„ë“œ ?¼ì¹˜ ?•ì¸
        assert back_to_flat["query"] == original_flat["query"]
        assert back_to_flat["query_type"] == original_flat["query_type"]
        assert back_to_flat["confidence"] == original_flat["confidence"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
