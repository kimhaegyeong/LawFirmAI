# -*- coding: utf-8 -*-
"""
Phase 1 ?¨ìœ„ ë°??µí•© ?ŒìŠ¤??
ConversationStore ?•ì¥, IntegratedSessionManager, MultiTurnQuestionHandler, ContextCompressor ?ŒìŠ¤??
"""

import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.conversation_store import ConversationStore
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor
from source.services.conversation_manager import ConversationContext, ConversationTurn


class TestConversationStoreExtensions(unittest.TestCase):
    """ConversationStore ?•ì¥ ê¸°ëŠ¥ ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_conversations.db")
        self.store = ConversationStore(self.db_path)
    
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        shutil.rmtree(self.temp_dir)
    
    def test_user_sessions(self):
        """?¬ìš©?ë³„ ?¸ì…˜ ì¡°íšŒ ?ŒìŠ¤??""
        # ?ŒìŠ¤???¸ì…˜ ?°ì´???ì„±
        test_session = {
            "session_id": "test_user_session_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["?í•´ë°°ìƒ", "ê³„ì•½"],
            "metadata": {"user_id": "test_user_001"},
            "turns": [
                {
                    "user_query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                    "bot_response": "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?]}
                }
            ],
            "entities": {
                "laws": ["ë¯¼ë²•"],
                "articles": ["??50ì¡?],
                "precedents": [],
                "legal_terms": ["?í•´ë°°ìƒ"]
            }
        }
        
        # ?¸ì…˜ ?€??
        self.assertTrue(self.store.save_session(test_session))
        
        # ?¬ìš©???¸ì…˜ ì¡°íšŒ
        user_sessions = self.store.get_user_sessions("test_user_001")
        self.assertEqual(len(user_sessions), 1)
        self.assertEqual(user_sessions[0]["session_id"], "test_user_session_001")
    
    def test_session_search(self):
        """?¸ì…˜ ê²€???ŒìŠ¤??""
        # ?ŒìŠ¤???¸ì…˜???ì„±
        sessions = [
            {
                "session_id": "search_test_001",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "topic_stack": ["?í•´ë°°ìƒ"],
                "metadata": {"user_id": "user_001"},
                "turns": [
                    {
                        "user_query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•",
                        "bot_response": "?í•´ë°°ìƒ ê´€???µë?",
                        "timestamp": datetime.now().isoformat(),
                        "question_type": "legal_advice",
                        "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
                    }
                ],
                "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
            },
            {
                "session_id": "search_test_002",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "topic_stack": ["ê³„ì•½"],
                "metadata": {"user_id": "user_002"},
                "turns": [
                    {
                        "user_query": "ê³„ì•½ ?´ì? ?ˆì°¨",
                        "bot_response": "ê³„ì•½ ê´€???µë?",
                        "timestamp": datetime.now().isoformat(),
                        "question_type": "procedure_guide",
                        "entities": {"legal_terms": ["ê³„ì•½"]}
                    }
                ],
                "entities": {"legal_terms": ["ê³„ì•½"]}
            }
        ]
        
        # ?¸ì…˜???€??
        for session in sessions:
            self.assertTrue(self.store.save_session(session))
        
        # ?¤ì›Œ??ê²€??
        results = self.store.search_sessions("?í•´ë°°ìƒ", {"limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "search_test_001")
        
        # ?¬ìš©???„í„° ê²€??
        results = self.store.search_sessions("", {"user_id": "user_001", "limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "search_test_001")
    
    def test_session_backup_restore(self):
        """?¸ì…˜ ë°±ì—… ë°?ë³µì› ?ŒìŠ¤??""
        # ?ŒìŠ¤???¸ì…˜ ?ì„±
        test_session = {
            "session_id": "backup_test_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["?í•´ë°°ìƒ"],
            "metadata": {},
            "turns": [
                {
                    "user_query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•",
                    "bot_response": "?í•´ë°°ìƒ ê´€???µë?",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
                }
            ],
            "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
        }
        
        # ?¸ì…˜ ?€??
        self.assertTrue(self.store.save_session(test_session))
        
        # ë°±ì—…
        backup_dir = os.path.join(self.temp_dir, "backup")
        self.assertTrue(self.store.backup_session("backup_test_001", backup_dir))
        
        # ë°±ì—… ?Œì¼ ?•ì¸
        backup_file = os.path.join(backup_dir, "backup_test_001_backup.json")
        self.assertTrue(os.path.exists(backup_file))
        
        # ë³µì›
        restored_session_id = self.store.restore_session(backup_file)
        self.assertIsNotNone(restored_session_id)
        self.assertTrue(restored_session_id.startswith("backup_test_001_restored_"))
    
    def test_statistics(self):
        """?µê³„ ì¡°íšŒ ?ŒìŠ¤??""
        # ?ŒìŠ¤???°ì´???ì„±
        test_session = {
            "session_id": "stats_test_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["?í•´ë°°ìƒ"],
            "metadata": {},
            "turns": [
                {
                    "user_query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•",
                    "bot_response": "?í•´ë°°ìƒ ê´€???µë?",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
                }
            ],
            "entities": {"legal_terms": ["?í•´ë°°ìƒ"]}
        }
        
        # ?¸ì…˜ ?€??
        self.assertTrue(self.store.save_session(test_session))
        
        # ?µê³„ ì¡°íšŒ
        stats = self.store.get_statistics()
        self.assertIn("session_count", stats)
        self.assertIn("turn_count", stats)
        self.assertIn("entity_count", stats)
        self.assertIn("user_count", stats)
        self.assertIn("memory_count", stats)
        self.assertIn("expertise_stats", stats)
        
        self.assertEqual(stats["session_count"], 1)
        self.assertEqual(stats["turn_count"], 1)
        self.assertEqual(stats["entity_count"], 1)


class TestIntegratedSessionManager(unittest.TestCase):
    """IntegratedSessionManager ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integrated_conversations.db")
        self.manager = IntegratedSessionManager(self.db_path)
    
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        shutil.rmtree(self.temp_dir)
    
    def test_add_turn(self):
        """??ì¶”ê? ?ŒìŠ¤??""
        session_id = "test_session_001"
        user_id = "test_user_001"
        
        # ??ì¶”ê?
        context = self.manager.add_turn(
            session_id, 
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            "legal_advice",
            user_id
        )
        
        self.assertIsNotNone(context)
        self.assertEqual(context.session_id, session_id)
        self.assertEqual(len(context.turns), 1)
        self.assertEqual(context.turns[0].user_query, "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??)
    
    def test_session_persistence(self):
        """?¸ì…˜ ì§€?ì„± ?ŒìŠ¤??""
        session_id = "test_persistence_001"
        user_id = "test_user_001"
        
        # ??ì¶”ê?
        context1 = self.manager.add_turn(
            session_id,
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            "legal_advice",
            user_id
        )
        
        # ê°•ì œ ?™ê¸°??
        self.assertTrue(self.manager.sync_to_database(session_id))
        
        # ?ˆë¡œ??ë§¤ë‹ˆ?€ ?¸ìŠ¤?´ìŠ¤ë¡??¸ì…˜ ë¡œë“œ
        new_manager = IntegratedSessionManager(self.db_path)
        context2 = new_manager.get_or_create_session(session_id, user_id)
        
        self.assertIsNotNone(context2)
        self.assertEqual(len(context2.turns), 1)
        self.assertEqual(context2.turns[0].user_query, "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??)
    
    def test_user_sessions(self):
        """?¬ìš©???¸ì…˜ ì¡°íšŒ ?ŒìŠ¤??""
        user_id = "test_user_002"
        
        # ?¬ëŸ¬ ?¸ì…˜ ?ì„±
        for i in range(3):
            session_id = f"test_user_sessions_{i:03d}"
            self.manager.add_turn(
                session_id,
                f"ì§ˆë¬¸ {i}",
                f"?µë? {i}",
                "legal_advice",
                user_id
            )
            self.manager.sync_to_database(session_id)
        
        # ?¬ìš©???¸ì…˜ ì¡°íšŒ
        user_sessions = self.manager.get_user_sessions(user_id)
        self.assertEqual(len(user_sessions), 3)
    
    def test_session_search(self):
        """?¸ì…˜ ê²€???ŒìŠ¤??""
        user_id = "test_user_003"
        
        # ê²€?‰ìš© ?¸ì…˜ ?ì„±
        session_id = "test_search_001"
        self.manager.add_turn(
            session_id,
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•",
            "?í•´ë°°ìƒ ê´€???µë?",
            "legal_advice",
            user_id
        )
        self.manager.sync_to_database(session_id)
        
        # ê²€???¤í–‰
        results = self.manager.search_sessions("?í•´ë°°ìƒ", {"limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], session_id)
    
    def test_cleanup(self):
        """?¸ì…˜ ?•ë¦¬ ?ŒìŠ¤??""
        # ?¤ë˜???¸ì…˜ ?ì„±
        old_session_id = "test_cleanup_001"
        self.manager.add_turn(
            old_session_id,
            "?¤ë˜??ì§ˆë¬¸",
            "?¤ë˜???µë?",
            "legal_advice"
        )
        
        # ?•ë¦¬ ?¤í–‰
        cleaned_count = self.manager.cleanup_old_sessions(days=0)  # ì¦‰ì‹œ ?•ë¦¬
        self.assertGreaterEqual(cleaned_count, 0)


class TestMultiTurnQuestionHandler(unittest.TestCase):
    """MultiTurnQuestionHandler ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.handler = MultiTurnQuestionHandler()
        
        # ?ŒìŠ¤?¸ìš© ?€??ë§¥ë½ ?ì„±
        self.test_turns = [
            ConversationTurn(
                user_query="ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                bot_response="ê³„ì•½ ?´ì? ?ˆì°¨???¤ìŒê³?ê°™ìŠµ?ˆë‹¤...",
                timestamp=datetime.now(),
                question_type="procedure_guide",
                entities={"legal_terms": ["ê³„ì•½", "?´ì?"]}
            ),
            ConversationTurn(
                user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?], "legal_terms": ["?í•´ë°°ìƒ"]}
            )
        ]
        
        self.context = ConversationContext(
            session_id="test_session",
            turns=self.test_turns,
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, 
                     "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ", "ê³„ì•½", "?´ì?"}},
            topic_stack=["?í•´ë°°ìƒ", "ê³„ì•½"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def test_detect_multi_turn_question(self):
        """?¤ì¤‘ ??ì§ˆë¬¸ ê°ì? ?ŒìŠ¤??""
        # ?¤ì¤‘ ??ì§ˆë¬¸??
        multi_turn_queries = [
            "ê·¸ê²ƒ???€?????ì„¸???Œë ¤ì£¼ì„¸??,
            "?„ì˜ ?í•´ë°°ìƒ ?¬ê±´?ì„œ ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²??•í•´ì§€?˜ìš”?",
            "ê·??ë????´ë–¤ ?¬ê±´?´ì—ˆ?˜ìš”?",
            "?´ê²ƒ??ë²•ì  ê·¼ê±°??ë¬´ì—‡?¸ê???"
        ]
        
        for query in multi_turn_queries:
            with self.subTest(query=query):
                is_multi_turn = self.handler.detect_multi_turn_question(query, self.context)
                self.assertTrue(is_multi_turn, f"'{query}' should be detected as multi-turn")
        
        # ?¼ë°˜ ì§ˆë¬¸
        normal_query = "?¼ë°˜?ì¸ ì§ˆë¬¸?…ë‹ˆ??
        is_multi_turn = self.handler.detect_multi_turn_question(normal_query, self.context)
        self.assertFalse(is_multi_turn, f"'{normal_query}' should not be detected as multi-turn")
    
    def test_resolve_pronouns(self):
        """?€ëª…ì‚¬ ?´ê²° ?ŒìŠ¤??""
        # ?€ëª…ì‚¬ê°€ ?¬í•¨??ì§ˆë¬¸
        pronoun_query = "ê·¸ê²ƒ???€?????ì„¸???Œë ¤ì£¼ì„¸??
        resolved_query = self.handler.resolve_pronouns(pronoun_query, self.context)
        
        self.assertNotEqual(pronoun_query, resolved_query)
        # ?€ëª…ì‚¬ê°€ ?´ê²°?˜ì–´????(?í•´ë°°ìƒ, ë¯¼ë²•, ê³„ì•½, ?´ì? ì¤??˜ë‚˜)
        self.assertTrue(
            any(keyword in resolved_query for keyword in ["?í•´ë°°ìƒ", "ë¯¼ë²•", "ê³„ì•½", "?´ì?"]),
            f"Expected one of ['?í•´ë°°ìƒ', 'ë¯¼ë²•', 'ê³„ì•½', '?´ì?'] in '{resolved_query}'"
        )
    
    def test_build_complete_query(self):
        """?„ì „??ì§ˆë¬¸ êµ¬ì„± ?ŒìŠ¤??""
        query = "ê·¸ê²ƒ???€?????ì„¸???Œë ¤ì£¼ì„¸??
        result = self.handler.build_complete_query(query, self.context)
        
        self.assertIn("original_query", result)
        self.assertIn("resolved_query", result)
        self.assertIn("referenced_entities", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning", result)
        
        self.assertEqual(result["original_query"], query)
        self.assertNotEqual(result["resolved_query"], query)
        self.assertGreater(result["confidence"], 0.0)
    
    def test_extract_reference_entities(self):
        """ì°¸ì¡° ?”í‹°??ì¶”ì¶œ ?ŒìŠ¤??""
        # ?”í‹°??ì°¸ì¡°ê°€ ?¬í•¨??ì§ˆë¬¸
        query = "ê·??ë????´ë–¤ ?¬ê±´?´ì—ˆ?˜ìš”?"
        entities = self.handler.extract_reference_entities(query)
        
        self.assertIsInstance(entities, list)
        # ?ë? ì°¸ì¡°ê°€ ê°ì??˜ì–´????
        self.assertTrue(any("?ë?" in entity for entity in entities))


class TestContextCompressor(unittest.TestCase):
    """ContextCompressor ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.compressor = ContextCompressor(max_tokens=1000)
        
        # ?ŒìŠ¤?¸ìš© ?€??ë§¥ë½ ?ì„±
        self.test_turns = [
            ConversationTurn(
                user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤. ?í•´ë°°ìƒ?€ ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë¥?ë°°ìƒë°›ëŠ” ?œë„?…ë‹ˆ?? ?í•´??ë°œìƒ, ê°€?´ì??ê³ ì˜ ?ëŠ” ê³¼ì‹¤, ?¸ê³¼ê´€ê³? ?í•´??ë°œìƒ???„ìš”?©ë‹ˆ??",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?], "legal_terms": ["?í•´ë°°ìƒ", "ë¶ˆë²•?‰ìœ„"]}
            ),
            ConversationTurn(
                user_query="ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                bot_response="ê³„ì•½ ?´ì? ?ˆì°¨???¤ìŒê³?ê°™ìŠµ?ˆë‹¤. 1) ?´ì? ?¬ìœ  ?•ì¸ 2) ?´ì? ?µì? 3) ?í•´ë°°ìƒ ì²?µ¬ 4) ?Œì†¡ ?œê¸° ?±ì´ ?ˆìŠµ?ˆë‹¤. ë¯¼ë²• ??43ì¡°ì— ?°ë¼ ê³„ì•½?€ ?¹ì‚¬???¼ë°©???˜ì‚¬?œì‹œë¡??´ì??????ˆìŠµ?ˆë‹¤.",
                timestamp=datetime.now(),
                question_type="procedure_guide",
                entities={"laws": ["ë¯¼ë²•"], "articles": ["??43ì¡?], "legal_terms": ["ê³„ì•½", "?´ì?"]}
            ),
            ConversationTurn(
                user_query="?„ì˜ ?í•´ë°°ìƒ ?¬ê±´?ì„œ ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²??•í•´ì§€?˜ìš”?",
                bot_response="ê³¼ì‹¤ë¹„ìœ¨?€ êµí†µ?¬ê³ ??ê²½ìš° ë³´í—˜?Œì‚¬?ì„œ ?•í•œ ê¸°ì??œë? ì°¸ê³ ?˜ì—¬ ê²°ì •?©ë‹ˆ?? ?€ë²•ì› 2023??2345 ?ë????°ë¥´ë©?ê³¼ì‹¤ë¹„ìœ¨?€ ?¬ê³  ?í™©, ?„ë¡œ ?í™©, ì°¨ëŸ‰ ?íƒœ ?±ì„ ì¢…í•©?ìœ¼ë¡?ê³ ë ¤?˜ì—¬ ê²°ì •?©ë‹ˆ??",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"precedents": ["2023??2345"], "legal_terms": ["ê³¼ì‹¤ë¹„ìœ¨", "êµí†µ?¬ê³ "]}
            )
        ]
        
        self.context = ConversationContext(
            session_id="test_session",
            turns=self.test_turns,
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?, "??43ì¡?}, 
                     "precedents": {"2023??2345"}, 
                     "legal_terms": {"?í•´ë°°ìƒ", "ë¶ˆë²•?‰ìœ„", "ê³„ì•½", "?´ì?", "ê³¼ì‹¤ë¹„ìœ¨", "êµí†µ?¬ê³ "}},
            topic_stack=["?í•´ë°°ìƒ", "ê³„ì•½", "êµí†µ?¬ê³ "],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def test_calculate_tokens(self):
        """? í° ??ê³„ì‚° ?ŒìŠ¤??""
        tokens = self.compressor.calculate_tokens(self.context)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_calculate_tokens_from_text(self):
        """?ìŠ¤??? í° ??ê³„ì‚° ?ŒìŠ¤??""
        text = "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??
        tokens = self.compressor.calculate_tokens_from_text(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_extract_key_information(self):
        """?µì‹¬ ?•ë³´ ì¶”ì¶œ ?ŒìŠ¤??""
        key_info = self.compressor.extract_key_information(self.context.turns)
        
        self.assertIn("entities", key_info)
        self.assertIn("topics", key_info)
        self.assertIn("question_types", key_info)
        self.assertIn("important_queries", key_info)
        self.assertIn("key_responses", key_info)
        
        self.assertIsInstance(key_info["entities"], list)
        self.assertIsInstance(key_info["question_types"], list)
        self.assertIsInstance(key_info["important_queries"], list)
    
    def test_maintain_relevant_context(self):
        """ê´€??ì»¨í…?¤íŠ¸ ? ì? ?ŒìŠ¤??""
        relevant_turns = self.compressor.maintain_relevant_context(self.context, "")
        
        self.assertIsInstance(relevant_turns, list)
        self.assertLessEqual(len(relevant_turns), len(self.context.turns))
        # ìµœê·¼ ?´ì? ??ƒ ?¬í•¨?˜ì–´????
        self.assertIn(self.context.turns[-1], relevant_turns)
    
    def test_compress_long_conversation(self):
        """ê¸??€???•ì¶• ?ŒìŠ¤??""
        # ?‘ì? ? í° ?œí•œ?¼ë¡œ ?•ì¶• ê°•ì œ
        result = self.compressor.compress_long_conversation(self.context, max_tokens=100)
        
        self.assertIn("original_tokens", result.__dict__)
        self.assertIn("compressed_tokens", result.__dict__)
        self.assertIn("compression_ratio", result.__dict__)
        self.assertIn("compressed_text", result.__dict__)
        self.assertIn("preserved_entities", result.__dict__)
        self.assertIn("preserved_topics", result.__dict__)
        self.assertIn("summary", result.__dict__)
        
        self.assertGreater(result.original_tokens, 0)
        self.assertGreater(result.compressed_tokens, 0)
        self.assertLessEqual(result.compression_ratio, 1.0)
        self.assertIsInstance(result.compressed_text, str)
        self.assertIsInstance(result.preserved_entities, list)
        self.assertIsInstance(result.preserved_topics, list)


class TestPhase1Integration(unittest.TestCase):
    """Phase 1 ?µí•© ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        
        # ëª¨ë“  ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.multi_turn_handler = MultiTurnQuestionHandler()
        self.context_compressor = ContextCompressor(max_tokens=500)
    
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_conversation(self):
        """?„ì²´ ?€???ë¦„ ?ŒìŠ¤??""
        session_id = "integration_test_001"
        user_id = "test_user_001"
        
        # 1. ì²?ë²ˆì§¸ ì§ˆë¬¸
        context1 = self.session_manager.add_turn(
            session_id,
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            "legal_advice",
            user_id
        )
        
        self.assertEqual(len(context1.turns), 1)
        
        # 2. ?¤ì¤‘ ??ì§ˆë¬¸ ì²˜ë¦¬
        multi_turn_query = "ê·¸ê²ƒ???€?????ì„¸???Œë ¤ì£¼ì„¸??
        is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(multi_turn_query, context1)
        self.assertTrue(is_multi_turn)
        
        resolved_result = self.multi_turn_handler.build_complete_query(multi_turn_query, context1)
        resolved_query = resolved_result["resolved_query"]
        
        # 3. ?´ê²°??ì§ˆë¬¸?¼ë¡œ ??ë²ˆì§¸ ??ì¶”ê?
        context2 = self.session_manager.add_turn(
            session_id,
            resolved_query,
            "?í•´ë°°ìƒ??êµ¬ì²´?ì¸ ?”ê±´???€???ì„¸???¤ëª…?œë¦¬ê² ìŠµ?ˆë‹¤...",
            "legal_advice",
            user_id
        )
        
        self.assertEqual(len(context2.turns), 2)
        
        # 4. ì»¨í…?¤íŠ¸ ?•ì¶• ?ŒìŠ¤??
        compression_result = self.context_compressor.compress_long_conversation(context2)
        self.assertLessEqual(compression_result.compressed_tokens, 500)
        
        # 5. ?¸ì…˜ ì§€?ì„± ?•ì¸
        self.session_manager.sync_to_database(session_id)
        
        # ?ˆë¡œ??ë§¤ë‹ˆ?€ë¡??¸ì…˜ ë¡œë“œ
        new_manager = IntegratedSessionManager(self.db_path)
        loaded_context = new_manager.get_or_create_session(session_id, user_id)
        
        self.assertEqual(len(loaded_context.turns), 2)
        self.assertEqual(loaded_context.turns[0].user_query, "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??)
    
    def test_performance_metrics(self):
        """?±ëŠ¥ ë©”íŠ¸ë¦??ŒìŠ¤??""
        session_id = "performance_test_001"
        user_id = "test_user_001"
        
        # ?¬ëŸ¬ ??ì¶”ê?
        for i in range(5):
            self.session_manager.add_turn(
                session_id,
                f"ì§ˆë¬¸ {i}",
                f"?µë? {i}",
                "legal_advice",
                user_id
            )
        
        # ?µê³„ ì¡°íšŒ
        stats = self.session_manager.get_session_stats()
        self.assertIn("memory_stats", stats)
        self.assertIn("database_stats", stats)
        self.assertIn("sync_stats", stats)
        
        # ?±ëŠ¥ ?•ì¸
        memory_stats = stats["memory_stats"]
        self.assertEqual(memory_stats["total_sessions"], 1)
        self.assertEqual(memory_stats["total_turns"], 5)


def run_phase1_tests():
    """Phase 1 ?ŒìŠ¤???¤í–‰"""
    print("=== Phase 1 ?¨ìœ„ ë°??µí•© ?ŒìŠ¤???¤í–‰ ===")
    
    # ?ŒìŠ¤???¤ìœ„???ì„±
    test_suite = unittest.TestSuite()
    
    # ?¨ìœ„ ?ŒìŠ¤??ì¶”ê?
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestConversationStoreExtensions))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegratedSessionManager))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMultiTurnQuestionHandler))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestContextCompressor))
    
    # ?µí•© ?ŒìŠ¤??ì¶”ê?
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPhase1Integration))
    
    # ?ŒìŠ¤???¤í–‰
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # ê²°ê³¼ ?”ì•½
    print(f"\n=== ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
    print(f"?¤í–‰???ŒìŠ¤?? {result.testsRun}")
    print(f"?±ê³µ: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"?¤íŒ¨: {len(result.failures)}")
    print(f"?¤ë¥˜: {len(result.errors)}")
    
    if result.failures:
        print(f"\n?¤íŒ¨???ŒìŠ¤??")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n?¤ë¥˜ê°€ ë°œìƒ???ŒìŠ¤??")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase1_tests()
    sys.exit(0 if success else 1)

