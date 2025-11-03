# -*- coding: utf-8 -*-
"""
State Helpers ?ŒìŠ¤??
Flat ë°?Modular êµ¬ì¡° ì§€???ŒìŠ¤??
"""

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402

from source.agents.modular_states import (  # noqa: E402
    create_initial_legal_state as create_modular_state,
)
from source.agents.state_definitions import (  # noqa: E402
    create_flat_legal_state as create_flat_state,
)
from source.agents.state_helpers import (  # noqa: E402
    ensure_state_group,
    get_answer_text,
    get_classification,
    get_field,
    get_nested_value,
    get_query,
    is_modular_state,
    set_field,
)


class TestStateHelpers:
    """State Helper ?¨ìˆ˜ ?ŒìŠ¤??""

    def test_is_modular_state_flat(self):
        """Flat êµ¬ì¡° ê°ì? ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        assert not is_modular_state(flat_state)

    def test_is_modular_state_modular(self):
        """Modular êµ¬ì¡° ê°ì? ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        assert is_modular_state(modular_state)

    def test_get_field_flat(self):
        """Flat êµ¬ì¡°?ì„œ ?„ë“œ ?‘ê·¼ ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        assert get_field(flat_state, "query") == "?ŒìŠ¤??ì§ˆë¬¸"
        assert get_field(flat_state, "session_id") == "session_123"
        assert get_field(flat_state, "query_type") == ""
        assert get_field(flat_state, "confidence") == 0.0

    def test_get_field_modular(self):
        """Modular êµ¬ì¡°?ì„œ ?„ë“œ ?‘ê·¼ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        assert get_field(modular_state, "query") == "?ŒìŠ¤??ì§ˆë¬¸"
        assert get_field(modular_state, "session_id") == "session_123"
        assert get_field(modular_state, "query_type") == ""
        assert get_field(modular_state, "confidence") == 0.0

    def test_set_field_flat(self):
        """Flat êµ¬ì¡°?ì„œ ?„ë“œ ?¤ì • ?ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        set_field(flat_state, "query_type", "legal_advice")
        assert flat_state["query_type"] == "legal_advice"

        set_field(flat_state, "confidence", 0.9)
        assert flat_state["confidence"] == 0.9

    def test_set_field_modular(self):
        """Modular êµ¬ì¡°?ì„œ ?„ë“œ ?¤ì • ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        set_field(modular_state, "query_type", "legal_advice")
        assert get_field(modular_state, "query_type") == "legal_advice"

        set_field(modular_state, "confidence", 0.9)
        assert get_field(modular_state, "confidence") == 0.9

    def test_get_nested_value(self):
        """ì¤‘ì²© ê°??‘ê·¼ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # ?•ìƒ ê²½ë¡œ
        assert get_nested_value(modular_state, "input", "query") == "?ŒìŠ¤??ì§ˆë¬¸"
        assert get_nested_value(modular_state, "classification", "query_type") == ""

        # ì¡´ì¬?˜ì? ?ŠëŠ” ê²½ë¡œ
        assert get_nested_value(modular_state, "nonexistent", "field", default="default") == "default"

    def test_ensure_state_group(self):
        """State ê·¸ë£¹ ì´ˆê¸°???ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # None?¼ë¡œ ?¤ì •
        modular_state["classification"] = None
        ensure_state_group(modular_state, "classification")

        assert modular_state["classification"] is not None
        assert isinstance(modular_state["classification"], dict)
        assert "query_type" in modular_state["classification"]

    def test_get_query_modular(self):
        """Modular êµ¬ì¡°?ì„œ query ?‘ê·¼ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        assert get_query(modular_state) == "?ŒìŠ¤??ì§ˆë¬¸"

    def test_get_classification_modular(self):
        """Modular êµ¬ì¡°?ì„œ classification ?‘ê·¼ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        classification = get_classification(modular_state)

        assert isinstance(classification, dict)
        assert "query_type" in classification
        assert "confidence" in classification

    def test_answer_field_access(self):
        """Answer ?„ë“œ ?‘ê·¼ ?ŒìŠ¤??""
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # answer ?„ë“œ ?‘ê·¼
        answer = get_answer_text(modular_state)
        assert answer == ""

        # answer ?¤ì •
        set_field(modular_state, "answer", "?µë??…ë‹ˆ??)
        assert get_answer_text(modular_state) == "?µë??…ë‹ˆ??

    def test_compatibility_between_structures(self):
        """Flatê³?Modular êµ¬ì¡° ê°??¸í™˜???ŒìŠ¤??""
        flat_state = create_flat_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")
        modular_state = create_modular_state("?ŒìŠ¤??ì§ˆë¬¸", "session_123")

        # ê°™ì? ?„ë“œ??ê°™ì? ê°??¤ì •
        set_field(flat_state, "query_type", "legal_advice")
        set_field(modular_state, "query_type", "legal_advice")

        # ê°™ì? ê°?ê°€?¸ì˜¤ê¸?
        assert get_field(flat_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "query_type") == "legal_advice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
