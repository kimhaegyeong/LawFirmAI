# -*- coding: utf-8 -*-
"""
LangGraph ë©€?°í„´ ë¡œì§ ?µí•© ?ŒìŠ¤??
"""

import sys
from pathlib import Path

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # pytestê°€ ?†ìœ¼ë©?unittestë¡??€ì²?
    import unittest
    pytest = unittest

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from source.services.conversation_manager import ConversationManager, ConversationTurn
from source.agents.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.agents.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


def test_multi_turn_integration_direct():
    """LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤??(ì§ì ‘ ?¤í–‰)"""
    print("\n=== LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤???œì‘ (ì§ì ‘ ?¤í–‰) ===\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("???Œí¬?Œë¡œ??ì´ˆê¸°???±ê³µ")

        # ë©€?°í„´ ?¸ë“¤??ì´ˆê¸°???•ì¸
        if hasattr(workflow, 'multi_turn_handler') and workflow.multi_turn_handler:
            print("??ë©€?°í„´ ?¸ë“¤??ì´ˆê¸°???„ë£Œ")
        else:
            print("??ë©€?°í„´ ?¸ë“¤?¬ê? ì´ˆê¸°?”ë˜ì§€ ?ŠìŒ")

        if hasattr(workflow, 'conversation_manager') and workflow.conversation_manager:
            print("???€??ê´€ë¦¬ì ì´ˆê¸°???„ë£Œ")
        else:
            print("???€??ê´€ë¦¬ìê°€ ì´ˆê¸°?”ë˜ì§€ ?ŠìŒ")

        # ?íƒœ ?•ì˜ ë©€?°í„´ ?„ë“œ ?•ì¸
        state = create_initial_legal_state("?ŒìŠ¤??ì§ˆë¬¸", "test_session")
        multi_turn_fields = ["is_multi_turn", "original_query", "resolved_query",
                            "multi_turn_confidence", "multi_turn_reasoning",
                            "conversation_history", "conversation_context"]
        missing_fields = [f for f in multi_turn_fields if f not in state]
        if missing_fields:
            print(f"???„ë½??ë©€?°í„´ ?„ë“œ: {missing_fields}")
        else:
            print("??ë©€?°í„´ ?„ë“œ ëª¨ë‘ ì¡´ì¬")

        # ?Œí¬?Œë¡œ???¸ë“œ ?•ì¸
        if hasattr(workflow, 'graph'):
            nodes = list(workflow.graph.nodes.keys())
            if "resolve_multi_turn" in nodes:
                print("??ë©€?°í„´ ?¸ë“œê°€ ?Œí¬?Œë¡œ?°ì— ?µí•©??)
            else:
                print("??ë©€?°í„´ ?¸ë“œê°€ ?Œí¬?Œë¡œ?°ì— ?†ìŒ")

        print("\n=== LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤???„ë£Œ ===\n")
        return True

    except Exception as e:
        print(f"???ŒìŠ¤???¤í–‰ ?¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        return False


class TestLangGraphMultiTurnIntegration:
    """LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤??(pytest??"""

    def __init__(self):
        """pytest fixture ?†ì´ ?¤í–‰ ê°€?¥í•˜?„ë¡ ì´ˆê¸°??""
        # pytest ?†ì´ ì§ì ‘ ?¤í–‰ ê°€?¥í•˜?„ë¡ ??ƒ ì´ˆê¸°??
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.conversation_manager = self._create_conversation_manager()

    def _create_conversation_manager(self):
        """?€??ê´€ë¦¬ì ?ì„±"""
        manager = ConversationManager()

        # ?ŒìŠ¤???€??ì¶”ê?
        session_id = "test_session_001"

        turn1 = ConversationTurn(
            user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            timestamp=datetime.now(),
            question_type="legal_advice",
            entities={"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?], "legal_terms": ["?í•´ë°°ìƒ"]}
        )

        turn2 = ConversationTurn(
            user_query="ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            bot_response="ê³„ì•½ ?´ì? ?ˆì°¨???¤ìŒê³?ê°™ìŠµ?ˆë‹¤...",
            timestamp=datetime.now(),
            question_type="procedure_guide",
            entities={"legal_terms": ["ê³„ì•½", "?´ì?"]}
        )

        context = manager.add_turn(session_id, turn1.user_query, turn1.bot_response, turn1.question_type)
        context = manager.add_turn(session_id, turn2.user_query, turn2.bot_response, turn2.question_type)

        return manager

    # pytestê°€ ?ˆì„ ?Œë§Œ fixture ?•ì˜
    if PYTEST_AVAILABLE:
        @pytest.fixture
        def config(self):
            """LangGraph ?¤ì •"""
            return LangGraphConfig.from_env()

        @pytest.fixture
        def workflow(self, config):
            """?Œí¬?Œë¡œ??ì´ˆê¸°??""
            return EnhancedLegalQuestionWorkflow(config)

        @pytest.fixture
        def conversation_manager(self):
            """?€??ê´€ë¦¬ì ì´ˆê¸°??""
            manager = ConversationManager()

            # ?ŒìŠ¤???€??ì¶”ê?
            session_id = "test_session_001"

            turn1 = ConversationTurn(
                user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?], "legal_terms": ["?í•´ë°°ìƒ"]}
            )

            turn2 = ConversationTurn(
                user_query="ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                bot_response="ê³„ì•½ ?´ì? ?ˆì°¨???¤ìŒê³?ê°™ìŠµ?ˆë‹¤...",
                timestamp=datetime.now(),
                question_type="procedure_guide",
                entities={"legal_terms": ["ê³„ì•½", "?´ì?"]}
            )

            context = manager.add_turn(session_id, turn1.user_query, turn1.bot_response, turn1.question_type)
            context = manager.add_turn(session_id, turn2.user_query, turn2.bot_response, turn2.question_type)

            return manager

    def test_multi_turn_handler_initialization(self, workflow=None):
        """ë©€?°í„´ ?¸ë“¤??ì´ˆê¸°???ŒìŠ¤??""
        # pytest fixtureê°€ ì£¼ì…?˜ì? ?Šìœ¼ë©??¸ìŠ¤?´ìŠ¤ ë³€???¬ìš©
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        assert workflow.multi_turn_handler is not None, "MultiTurnQuestionHandlerê°€ ì´ˆê¸°?”ë˜ì§€ ?Šì•˜?µë‹ˆ??
        assert workflow.conversation_manager is not None, "ConversationManagerê°€ ì´ˆê¸°?”ë˜ì§€ ?Šì•˜?µë‹ˆ??
        print("??ë©€?°í„´ ?¸ë“¤??ì´ˆê¸°???±ê³µ")

    def test_state_definitions_multi_turn_fields(self):
        """?íƒœ ?•ì˜??ë©€?°í„´ ?„ë“œê°€ ?ˆëŠ”ì§€ ?ŒìŠ¤??""
        state = create_initial_legal_state("?ŒìŠ¤??ì§ˆë¬¸", "test_session")

        # ë©€?°í„´ ê´€???„ë“œ ?•ì¸
        assert "is_multi_turn" in state, "is_multi_turn ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "original_query" in state, "original_query ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "resolved_query" in state, "resolved_query ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "multi_turn_confidence" in state, "multi_turn_confidence ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "multi_turn_reasoning" in state, "multi_turn_reasoning ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "conversation_history" in state, "conversation_history ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"
        assert "conversation_context" in state, "conversation_context ?„ë“œê°€ ?†ìŠµ?ˆë‹¤"

        print("??ë©€?°í„´ ?„ë“œ ëª¨ë‘ ì¡´ì¬")

    def test_resolve_multi_turn_node(self, workflow=None):
        """ë©€?°í„´ ?´ê²° ?¸ë“œ ?ŒìŠ¤??""
        # pytest fixtureê°€ ì£¼ì…?˜ì? ?Šìœ¼ë©??¸ìŠ¤?´ìŠ¤ ë³€???¬ìš©
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        # ?ŒìŠ¤???íƒœ ?ì„±
        state = create_initial_legal_state("ê·¸ê²ƒ???€?????ì„¸???Œë ¤ì£¼ì„¸??, "test_session_001")

        # ?€??ë§¥ë½ ?œë??ˆì´?˜ì„ ?„í•´ conversation_manager??ì§ì ‘ ?‘ì„¸??
        if workflow.conversation_manager:
            workflow.conversation_manager.sessions = self._create_test_context()

        # ë©€?°í„´ ?´ê²° ?¸ë“œ ?¤í–‰
        result_state = workflow.resolve_multi_turn(state)

        # ê²°ê³¼ ?•ì¸
        assert "is_multi_turn" in result_state
        assert "resolved_query" in result_state
        assert "original_query" in result_state

        print(f"??ë©€?°í„´ ?¸ë“œ ?¤í–‰: is_multi_turn={result_state.get('is_multi_turn')}")
        print(f"  Original: {result_state.get('original_query')}")
        print(f"  Resolved: {result_state.get('resolved_query')}")

    def test_single_turn_question(self, workflow=None):
        """?¨ì¼ ??ì§ˆë¬¸ ì²˜ë¦¬ ?ŒìŠ¤??""
        # pytest fixtureê°€ ì£¼ì…?˜ì? ?Šìœ¼ë©??¸ìŠ¤?´ìŠ¤ ë³€???¬ìš©
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        state = create_initial_legal_state("?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??, "test_session_001")

        result_state = workflow.resolve_multi_turn(state)

        # ?¨ì¼ ??ì§ˆë¬¸?´ë?ë¡?is_multi_turn?€ False?¬ì•¼ ??
        assert result_state.get("is_multi_turn") == False
        assert result_state.get("resolved_query") == state["query"]

        print("???¨ì¼ ??ì§ˆë¬¸ ì²˜ë¦¬ ?±ê³µ")

    def test_workflow_graph_includes_multi_turn_node(self, workflow=None):
        """?Œí¬?Œë¡œ??ê·¸ë˜?„ì— ë©€?°í„´ ?¸ë“œê°€ ?¬í•¨?˜ì–´ ?ˆëŠ”ì§€ ?ŒìŠ¤??""
        # pytest fixtureê°€ ì£¼ì…?˜ì? ?Šìœ¼ë©??¸ìŠ¤?´ìŠ¤ ë³€???¬ìš©
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        nodes = workflow.graph.nodes.keys()

        assert "resolve_multi_turn" in nodes, "resolve_multi_turn ?¸ë“œê°€ ê·¸ë˜?„ì— ?†ìŠµ?ˆë‹¤"

        print(f"???Œí¬?Œë¡œ???¸ë“œ: {list(nodes)}")

    def _create_test_context(self):
        """?ŒìŠ¤?¸ìš© ?€??ë§¥ë½ ?ì„±"""
        from datetime import datetime

        from source.services.conversation_manager import (
            ConversationContext,
            ConversationManager,
        )

        manager = ConversationManager()
        session_id = "test_session_001"

        manager.add_turn(
            session_id,
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            "legal_advice"
        )

        manager.add_turn(
            session_id,
            "ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            "ê³„ì•½ ?´ì? ?ˆì°¨???¤ìŒê³?ê°™ìŠµ?ˆë‹¤...",
            "procedure_guide"
        )

        return manager.sessions


def test_multi_turn_integration():
    """ë©€?°í„´ ?µí•© ?ŒìŠ¤???¤í–‰"""
    print("\n=== LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤???œì‘ ===\n")

    # ?¤ì • ë¡œë“œ
    try:
        config = LangGraphConfig.from_env()
        print("???¤ì • ë¡œë“œ ?±ê³µ")
    except Exception as e:
        print(f"???¤ì • ë¡œë“œ ?¤íŒ¨: {e}")
        return

    # ?Œí¬?Œë¡œ??ì´ˆê¸°??
    try:
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("???Œí¬?Œë¡œ??ì´ˆê¸°???±ê³µ")

        # ë©€?°í„´ ?¸ë“¤???•ì¸
        if workflow.multi_turn_handler:
            print("??ë©€?°í„´ ?¸ë“¤??ì´ˆê¸°???„ë£Œ")
        else:
            print("??ë©€?°í„´ ?¸ë“¤?¬ê? ì´ˆê¸°?”ë˜ì§€ ?ŠìŒ")

        if workflow.conversation_manager:
            print("???€??ê´€ë¦¬ì ì´ˆê¸°???„ë£Œ")
        else:
            print("???€??ê´€ë¦¬ìê°€ ì´ˆê¸°?”ë˜ì§€ ?ŠìŒ")

    except Exception as e:
        print(f"???Œí¬?Œë¡œ??ì´ˆê¸°???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        return

    # ?Œí¬?Œë¡œ???¸ë“œ ?•ì¸
    try:
        nodes = list(workflow.graph.nodes.keys())
        print(f"\n?Œí¬?Œë¡œ???¸ë“œ ëª©ë¡: {nodes}")

        if "resolve_multi_turn" in nodes:
            print("??ë©€?°í„´ ?¸ë“œê°€ ?Œí¬?Œë¡œ?°ì— ?µí•©??)
        else:
            print("??ë©€?°í„´ ?¸ë“œê°€ ?Œí¬?Œë¡œ?°ì— ?†ìŒ")

    except Exception as e:
        print(f"???¸ë“œ ?•ì¸ ?¤íŒ¨: {e}")

    print("\n=== LangGraph ë©€?°í„´ ?µí•© ?ŒìŠ¤???„ë£Œ ===\n")


if __name__ == "__main__":
    # pytestê°€ ?†ìœ¼ë©?ì§ì ‘ ?¤í–‰ ê°€?¥í•œ ?¨ìˆ˜ ?¸ì¶œ
    if PYTEST_AVAILABLE:
        test_multi_turn_integration()
    else:
        test_multi_turn_integration_direct()
