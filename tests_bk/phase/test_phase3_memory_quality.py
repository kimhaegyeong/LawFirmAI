# -*- coding: utf-8 -*-
"""
Phase 3: ?¥ê¸° ê¸°ì–µ ë°??ˆì§ˆ ëª¨ë‹ˆ?°ë§ ê¸°ëŠ¥ ?ŒìŠ¤???¤í¬ë¦½íŠ¸
- ContextualMemoryManager
- ConversationQualityMonitor
"""

import json
import os
import shutil
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.conversation_store import ConversationStore
from source.services.contextual_memory_manager import (
    ContextualMemoryManager,
    MemoryItem,
    MemoryType,
)
from source.services.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationTurn,
)
from source.services.conversation_quality_monitor import (
    ConversationQualityMonitor,
    QualityMetrics,
)


class TestPhase3MemoryAndQuality(unittest.TestCase):

    def setUp(self):
        self.test_db_path = "test_conversations_phase3.db"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.conversation_store = ConversationStore(self.test_db_path)
        self.memory_manager = ContextualMemoryManager(self.conversation_store)
        self.quality_monitor = ConversationQualityMonitor(self.conversation_store)

        self.user_id = "test_user_phase3"
        self.session_id = "test_session_phase3"

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_contextual_memory_manager_store_facts(self):
        print("\n--- Test ContextualMemoryManager: Store Facts ---")

        # ì¤‘ìš”???¬ì‹¤ ?€??
        facts = {
            "legal_knowledge": [
                "ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??,
                "ê³„ì•½ ?´ì œ ???„ì•½ê¸ˆì? ?í•´ë°°ìƒ?¡ì„ ì´ˆê³¼?????†ìŠµ?ˆë‹¤"
            ],
            "user_context": [
                "?¬ìš©?ëŠ” ë¶€?™ì‚° ê´€??ë¬¸ì œë¥??ì£¼ ë¬¸ì˜?©ë‹ˆ??,
                "?¬ìš©?ëŠ” ë²•ë¥  ì´ˆë³´?ì…?ˆë‹¤"
            ],
            "preference": [
                "?¬ìš©?ëŠ” ?ì„¸???¤ëª…??? í˜¸?©ë‹ˆ??,
                "?¬ìš©?ëŠ” ?ë? ì¤‘ì‹¬???µë????í•©?ˆë‹¤"
            ]
        }

        success = self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)
        self.assertTrue(success)

        # ?€?¥ëœ ë©”ëª¨ë¦??•ì¸
        memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        self.assertGreater(len(memories), 0)

    def test_contextual_memory_manager_retrieve_memory(self):
        print("\n--- Test ContextualMemoryManager: Retrieve Memory ---")

        # ?ŒìŠ¤??ë©”ëª¨ë¦??€??
        facts = {
            "legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??]
        }
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # ê´€??ë©”ëª¨ë¦?ê²€??
        query = "?í•´ë°°ìƒ ê´€??ì§ˆë¬¸?…ë‹ˆ??
        relevant_memories = self.memory_manager.retrieve_relevant_memory(self.session_id, query, self.user_id)

        self.assertIsNotNone(relevant_memories)
        self.assertIsInstance(relevant_memories, list)
        if relevant_memories:
            self.assertGreater(relevant_memories[0].relevance_score, 0)

    def test_contextual_memory_manager_extract_facts(self):
        print("\n--- Test ContextualMemoryManager: Extract Facts ---")

        # ?ŒìŠ¤???€????
        turn = ConversationTurn(
            user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
            bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ??..",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # ?¬ì‹¤ ì¶”ì¶œ
        extracted_facts = self.memory_manager.extract_facts_from_conversation(turn)

        self.assertIsNotNone(extracted_facts)
        self.assertIsInstance(extracted_facts, list)
        self.assertGreater(len(extracted_facts), 0)

    def test_contextual_memory_manager_consolidate_memories(self):
        print("\n--- Test ContextualMemoryManager: Consolidate Memories ---")

        # ì¤‘ë³µ ë©”ëª¨ë¦??€??
        facts1 = {"legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??]}
        facts2 = {"legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??]}  # ? ì‚¬???´ìš©

        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts1)
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts2)

        # ë©”ëª¨ë¦??µí•©
        consolidated_count = self.memory_manager.consolidate_memories(self.user_id)
        self.assertGreaterEqual(consolidated_count, 0)

    def test_contextual_memory_manager_statistics(self):
        print("\n--- Test ContextualMemoryManager: Statistics ---")

        # ?ŒìŠ¤??ë©”ëª¨ë¦??€??
        facts = {
            "legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??],
            "user_context": ["?¬ìš©?ëŠ” ë²•ë¥  ì´ˆë³´?ì…?ˆë‹¤"]
        }
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # ?µê³„ ì¡°íšŒ
        stats = self.memory_manager.get_memory_statistics(self.user_id)

        self.assertIsNotNone(stats)
        self.assertIn("user_id", stats)
        self.assertIn("total_memories", stats)
        self.assertGreater(stats["total_memories"], 0)

    def test_contextual_memory_manager_importance_update(self):
        print("\n--- Test ContextualMemoryManager: Importance Update ---")

        # ?ŒìŠ¤??ë©”ëª¨ë¦??€??
        facts = {"legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??]}
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # ë©”ëª¨ë¦?ê²€??
        memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        self.assertGreater(len(memories), 0)

        # ì¤‘ìš”???…ë°?´íŠ¸
        memory_id = memories[0].memory_id
        success = self.memory_manager.update_memory_importance(memory_id, 0.9)
        self.assertTrue(success)

    def test_contextual_memory_manager_cleanup(self):
        print("\n--- Test ContextualMemoryManager: Cleanup ---")

        # ?ŒìŠ¤??ë©”ëª¨ë¦??€??
        facts = {"legal_knowledge": ["ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??]}
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # ?¤ë˜??ë©”ëª¨ë¦??•ë¦¬ (?ŒìŠ¤?¸ë? ?„í•´ 1?¼ë¡œ ?¤ì •)
        cleaned_count = self.memory_manager.cleanup_old_memories(days=1)
        self.assertGreaterEqual(cleaned_count, 0)

    def test_conversation_quality_monitor_assess_quality(self):
        print("\n--- Test ConversationQualityMonitor: Assess Quality ---")

        # ?ŒìŠ¤???€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ?? ??ì¡°ë¬¸???°ë¥´ë©?..",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="ê°ì‚¬?©ë‹ˆ?? ?•ë§ ?„ì????˜ì—ˆ?´ìš”!",
                    bot_response="ì²œë§Œ?ìš”. ì¶”ê?ë¡?ê¶ê¸ˆ??ê²ƒì´ ?ˆìœ¼?œë©´ ?¸ì œ? ì? ë¬¸ì˜?´ì£¼?¸ìš”.",
                    timestamp=datetime.now(),
                    question_type="thanks"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["ë¯¼ë²•", "?í•´ë°°ìƒ"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?ˆì§ˆ ?‰ê?
        quality_assessment = self.quality_monitor.assess_conversation_quality(context)

        self.assertIsNotNone(quality_assessment)
        self.assertIn("overall_score", quality_assessment)
        self.assertIn("completeness_score", quality_assessment)
        self.assertIn("satisfaction_score", quality_assessment)
        self.assertIn("accuracy_score", quality_assessment)
        self.assertGreaterEqual(quality_assessment["overall_score"], 0.0)
        self.assertLessEqual(quality_assessment["overall_score"], 1.0)

    def test_conversation_quality_monitor_detect_issues(self):
        print("\n--- Test ConversationQualityMonitor: Detect Issues ---")

        # ë¬¸ì œê°€ ?ˆëŠ” ?€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="???ì„¸???¤ëª…?´ì£¼?¸ìš”. ?´í•´ê°€ ???¼ìš”",
                    bot_response="?? ?¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤.",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ë¬¸ì œ??ê°ì?
        issues = self.quality_monitor.detect_conversation_issues(context)

        self.assertIsNotNone(issues)
        self.assertIsInstance(issues, list)

    def test_conversation_quality_monitor_suggest_improvements(self):
        print("\n--- Test ConversationQualityMonitor: Suggest Improvements ---")

        # ?ŒìŠ¤???€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="ë¯¼ë²• ??50ì¡°ì— ?€???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["ë¯¼ë²•"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ê°œì„  ?œì•ˆ ?ì„±
        suggestions = self.quality_monitor.suggest_improvements(context)

        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)

    def test_conversation_quality_monitor_calculate_turn_quality(self):
        print("\n--- Test ConversationQualityMonitor: Calculate Turn Quality ---")

        # ?ŒìŠ¤???€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["ë¯¼ë²•"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?ŒìŠ¤????
        turn = ConversationTurn(
            user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
            bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ?? ??ì¡°ë¬¸???°ë¥´ë©?..",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # ???ˆì§ˆ ê³„ì‚°
        turn_quality = self.quality_monitor.calculate_turn_quality(turn, context)

        self.assertIsNotNone(turn_quality)
        self.assertIn("completeness", turn_quality)
        self.assertIn("satisfaction", turn_quality)
        self.assertIn("accuracy", turn_quality)
        self.assertGreaterEqual(turn_quality["completeness"], 0.0)
        self.assertLessEqual(turn_quality["completeness"], 1.0)

    def test_conversation_quality_monitor_dashboard_data(self):
        print("\n--- Test ConversationQualityMonitor: Dashboard Data ---")

        # ?ˆì§ˆ ?€?œë³´???°ì´???ì„±
        dashboard_data = self.quality_monitor.get_quality_dashboard_data(self.user_id)

        self.assertIsNotNone(dashboard_data)
        self.assertIsInstance(dashboard_data, dict)

    def test_conversation_quality_monitor_trend_analysis(self):
        print("\n--- Test ConversationQualityMonitor: Trend Analysis ---")

        # ?ˆì§ˆ ?¸ë Œ??ë¶„ì„
        trend_analysis = self.quality_monitor.analyze_quality_trends([self.session_id])

        self.assertIsNotNone(trend_analysis)
        self.assertIsInstance(trend_analysis, dict)

    def test_integrated_phase3_components(self):
        print("\n--- Test Integrated Phase 3 Components ---")

        # ?ŒìŠ¤???€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ?? ??ì¡°ë¬¸???°ë¥´ë©?..",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="ê°ì‚¬?©ë‹ˆ?? ?•ë§ ?„ì????˜ì—ˆ?´ìš”!",
                    bot_response="ì²œë§Œ?ìš”. ì¶”ê?ë¡?ê¶ê¸ˆ??ê²ƒì´ ?ˆìœ¼?œë©´ ?¸ì œ? ì? ë¬¸ì˜?´ì£¼?¸ìš”.",
                    timestamp=datetime.now(),
                    question_type="thanks"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["ë¯¼ë²•", "?í•´ë°°ìƒ"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?µí•© ?ŒìŠ¤?? ë©”ëª¨ë¦?ê´€ë¦?+ ?ˆì§ˆ ëª¨ë‹ˆ?°ë§

        # 1. ?€?”ì—???¬ì‹¤ ì¶”ì¶œ ë°??€??
        facts = {}
        for turn in context.turns:
            extracted_facts = self.memory_manager.extract_facts_from_conversation(turn)
            for fact in extracted_facts:
                fact_type = fact["type"]
                if fact_type not in facts:
                    facts[fact_type] = []
                facts[fact_type].append(fact["content"])

        memory_storage_success = self.memory_manager.store_important_facts(
            self.session_id, self.user_id, facts
        )
        self.assertTrue(memory_storage_success)

        # 2. ê´€??ë©”ëª¨ë¦?ê²€??
        query = "?í•´ë°°ìƒ ê´€??ì§ˆë¬¸?…ë‹ˆ??
        relevant_memories = self.memory_manager.retrieve_relevant_memory(
            self.session_id, query, self.user_id
        )

        # ?”ë²„ê¹? ?€?¥ëœ ë©”ëª¨ë¦??•ì¸
        all_memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        print(f"   ?”ë²„ê¹? ?€?¥ëœ ë©”ëª¨ë¦??? {len(all_memories)}")
        for i, mem in enumerate(all_memories):
            print(f"   ë©”ëª¨ë¦?{i+1}: {mem.content[:50]}... (?ìˆ˜: {mem.importance_score})")
            # ê´€?¨ì„± ?ìˆ˜ ê³„ì‚°
            score = self.memory_manager._calculate_relevance_score(mem, query)
            print(f"   ê´€?¨ì„± ?ìˆ˜: {score}")

        self.assertIsNotNone(relevant_memories)

        # 3. ?€???ˆì§ˆ ?‰ê?
        quality_assessment = self.quality_monitor.assess_conversation_quality(context)
        self.assertIsNotNone(quality_assessment)
        self.assertGreaterEqual(quality_assessment["overall_score"], 0.0)

        # 4. ë¬¸ì œ??ê°ì? ë°?ê°œì„  ?œì•ˆ
        issues = self.quality_monitor.detect_conversation_issues(context)
        suggestions = self.quality_monitor.suggest_improvements(context)

        self.assertIsNotNone(issues)
        self.assertIsNotNone(suggestions)

        # 5. ë©”ëª¨ë¦??µê³„ ë°??ˆì§ˆ ?€?œë³´??
        memory_stats = self.memory_manager.get_memory_statistics(self.user_id)
        quality_dashboard = self.quality_monitor.get_quality_dashboard_data(self.user_id)

        self.assertIsNotNone(memory_stats)
        self.assertIsNotNone(quality_dashboard)

        # ?µí•© ê²°ê³¼ ê²€ì¦?
        self.assertGreater(len(relevant_memories), 0)
        self.assertGreater(quality_assessment["overall_score"], 0.0)
        self.assertGreater(memory_stats["total_memories"], 0)

        print(f"   ?µí•© ?ŒìŠ¤???„ë£Œ:")
        print(f"   - ?€?¥ëœ ë©”ëª¨ë¦??? {memory_stats['total_memories']}")
        print(f"   - ê²€?‰ëœ ê´€??ë©”ëª¨ë¦??? {len(relevant_memories)}")
        print(f"   - ?€???ˆì§ˆ ?ìˆ˜: {quality_assessment['overall_score']:.2f}")
        print(f"   - ê°ì???ë¬¸ì œ???? {len(issues)}")
        print(f"   - ê°œì„  ?œì•ˆ ?? {len(suggestions)}")

    def test_memory_quality_integration(self):
        print("\n--- Test Memory-Quality Integration ---")

        # ?ˆì§ˆ???’ì? ?€?”ì? ??? ?€???ì„±
        high_quality_context = ConversationContext(
            session_id=f"{self.session_id}_high",
            turns=[
                ConversationTurn(
                    user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ?? ??ì¡°ë¬¸???°ë¥´ë©?ê³ ì˜ ?ëŠ” ê³¼ì‹¤ë¡??¸í•˜???€?¸ì—ê²??í•´ë¥?ê°€???ëŠ” ê·??í•´ë¥?ë°°ìƒ??ì±…ì„???ˆìŠµ?ˆë‹¤. ?´ëŠ” ë¯¼ë²•???µì‹¬ ?ì¹™ ì¤??˜ë‚˜ë¡?..",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["ë¯¼ë²•"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        low_quality_context = ConversationContext(
            session_id=f"{self.session_id}_low",
            turns=[
                ConversationTurn(
                    user_query="???ì„¸???¤ëª…?´ì£¼?¸ìš”. ?´í•´ê°€ ???¼ìš”",
                    bot_response="?? ?¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤.",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ê°??€?”ì˜ ?ˆì§ˆ ?‰ê?
        high_quality_assessment = self.quality_monitor.assess_conversation_quality(high_quality_context)
        low_quality_assessment = self.quality_monitor.assess_conversation_quality(low_quality_context)

        # ?ˆì§ˆ???’ì? ?€?”ì—????ë§ì? ?¬ì‹¤ ì¶”ì¶œ
        high_quality_facts = {}
        for turn in high_quality_context.turns:
            extracted_facts = self.memory_manager.extract_facts_from_conversation(turn)
            for fact in extracted_facts:
                fact_type = fact["type"]
                if fact_type not in high_quality_facts:
                    high_quality_facts[fact_type] = []
                high_quality_facts[fact_type].append(fact["content"])

        low_quality_facts = {}
        for turn in low_quality_context.turns:
            extracted_facts = self.memory_manager.extract_facts_from_conversation(turn)
            for fact in extracted_facts:
                fact_type = fact["type"]
                if fact_type not in low_quality_facts:
                    low_quality_facts[fact_type] = []
                low_quality_facts[fact_type].append(fact["content"])

        # ë©”ëª¨ë¦??€??
        self.memory_manager.store_important_facts(
            high_quality_context.session_id, self.user_id, high_quality_facts
        )
        self.memory_manager.store_important_facts(
            low_quality_context.session_id, self.user_id, low_quality_facts
        )

        # ?µê³„ ?•ì¸
        memory_stats = self.memory_manager.get_memory_statistics(self.user_id)

        # ê²€ì¦?
        self.assertGreater(high_quality_assessment["overall_score"], low_quality_assessment["overall_score"])
        self.assertGreater(memory_stats["total_memories"], 0)

        print(f"   ë©”ëª¨ë¦??ˆì§ˆ ?µí•© ?ŒìŠ¤???„ë£Œ:")
        print(f"   - ê³ í’ˆì§??€???ìˆ˜: {high_quality_assessment['overall_score']:.2f}")
        print(f"   - ?€?ˆì§ˆ ?€???ìˆ˜: {low_quality_assessment['overall_score']:.2f}")
        print(f"   - ì´?ë©”ëª¨ë¦??? {memory_stats['total_memories']}")


if __name__ == '__main__':
    unittest.main()
