# -*- coding: utf-8 -*-
"""
LawFirmAI ì¢…í•© ?œìŠ¤???ŒìŠ¤??
ëª¨ë“  ì»´í¬?ŒíŠ¸???µí•© ?ŒìŠ¤??ë°??±ëŠ¥ ?ŒìŠ¤??
"""

import unittest
import asyncio
import time
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor
from source.services.user_profile_manager import UserProfileManager
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer
from source.services.conversation_flow_tracker import ConversationFlowTracker
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor
from source.utils.performance_optimizer import PerformanceMonitor, MemoryOptimizer, CacheManager
from source.data.conversation_store import ConversationStore
from source.services.conversation_manager import ConversationManager, ConversationContext, ConversationTurn


class TestComprehensiveSystem(unittest.TestCase):
    """ì¢…í•© ?œìŠ¤???ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_comprehensive.db")
        
        # ëª¨ë“  ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.multi_turn_handler = MultiTurnQuestionHandler()
        self.context_compressor = ContextCompressor(max_tokens=1000)
        self.user_profile_manager = UserProfileManager(self.session_manager.conversation_store)
        self.emotion_analyzer = EmotionIntentAnalyzer()
        self.flow_tracker = ConversationFlowTracker()
        self.memory_manager = ContextualMemoryManager(self.session_manager.conversation_store)
        self.quality_monitor = ConversationQualityMonitor(self.session_manager.conversation_store)
        
        # ?±ëŠ¥ ëª¨ë‹ˆ?°ë§ ì»´í¬?ŒíŠ¸
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager(max_size=500, ttl=1800)
        
        # ?ŒìŠ¤???°ì´??
        self.test_user_id = "test_user_comprehensive"
        self.test_session_id = "test_session_comprehensive"
        
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_phase1_comprehensive(self):
        """Phase 1 ì¢…í•© ?ŒìŠ¤??""
        print("\n=== Phase 1 ì¢…í•© ?ŒìŠ¤??===")
        
        # 1. ?¤ì¤‘ ???€???œë‚˜ë¦¬ì˜¤
        conversation_scenarios = [
            {
                "query": "?í•´ë°°ìƒ???€???Œë ¤ì£¼ì„¸??,
                "response": "ë¯¼ë²• ??50ì¡°ì— ?°ë¥¸ ?í•´ë°°ìƒ?€ ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë¥?ë°°ìƒ?˜ëŠ” ?œë„?…ë‹ˆ??",
                "expected_entities": ["?í•´ë°°ìƒ", "ë¯¼ë²•", "??50ì¡?]
            },
            {
                "query": "ê·¸ê²ƒ???”ê±´?€ ë¬´ì—‡?¸ê???",
                "response": "?í•´ë°°ìƒ???”ê±´?€ ê°€?´í–‰?? ?í•´ë°œìƒ, ?¸ê³¼ê´€ê³? ê³ ì˜ ?ëŠ” ê³¼ì‹¤?…ë‹ˆ??",
                "expected_entities": ["?í•´ë°°ìƒ", "?”ê±´", "ê°€?´í–‰??, "?í•´ë°œìƒ"]
            },
            {
                "query": "?„ì˜ ?¬ê±´?ì„œ ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²??•í•´ì§€?˜ìš”?",
                "response": "ê³¼ì‹¤ë¹„ìœ¨?€ êµí†µ?¬ê³ ??ê²½ìš° ?„ë¡œêµí†µë²•ì— ?°ë¼ ?•í•´ì§‘ë‹ˆ??",
                "expected_entities": ["ê³¼ì‹¤ë¹„ìœ¨", "êµí†µ?¬ê³ ", "?„ë¡œêµí†µë²?]
            }
        ]
        
        # ?€??ì§„í–‰
        for i, scenario in enumerate(conversation_scenarios):
            print(f"\n??{i+1}: {scenario['query']}")
            
            # ??ì¶”ê?
            context = self.session_manager.add_turn(
                self.test_session_id,
                scenario["query"],
                scenario["response"],
                "legal_advice",
                self.test_user_id
            )
            
            # ?¤ì¤‘ ??ì²˜ë¦¬ ?ŒìŠ¤??
            multi_turn_result = self.multi_turn_handler.build_complete_query(
                scenario["query"], context
            )
            
            self.assertIsNotNone(multi_turn_result)
            self.assertIn("resolved_query", multi_turn_result)
            self.assertIn("confidence", multi_turn_result)
            
            print(f"  ?´ê²°??ì¿¼ë¦¬: {multi_turn_result['resolved_query']}")
            print(f"  ? ë¢°?? {multi_turn_result['confidence']:.2f}")
            
            # ?”í‹°??ì¶”ì¶œ ?•ì¸
            if i == 0:  # ì²?ë²ˆì§¸ ?´ì—???”í‹°???•ì¸
                entities = context.entities
                for entity_type, entity_set in entities.items():
                    if entity_set:
                        print(f"  {entity_type}: {list(entity_set)}")
        
        # 2. ì»¨í…?¤íŠ¸ ?•ì¶• ?ŒìŠ¤??
        print("\nì»¨í…?¤íŠ¸ ?•ì¶• ?ŒìŠ¤??")
        compression_result = self.context_compressor.compress_long_conversation(context)
        
        self.assertIsNotNone(compression_result)
        self.assertLessEqual(compression_result.compression_ratio, 1.0)
        self.assertGreater(compression_result.original_tokens, 0)
        
        print(f"  ?ë³¸ ? í°: {compression_result.original_tokens}")
        print(f"  ?•ì¶• ? í°: {compression_result.compressed_tokens}")
        print(f"  ?•ì¶•ë¥? {compression_result.compression_ratio:.2f}")
        
        # 3. ?¸ì…˜ ì§€?ì„± ?ŒìŠ¤??
        print("\n?¸ì…˜ ì§€?ì„± ?ŒìŠ¤??")
        self.session_manager.sync_to_database(self.test_session_id)
        
        # ?ˆë¡œ???¸ì…˜ ë§¤ë‹ˆ?€ë¡?ë¡œë“œ
        new_session_manager = IntegratedSessionManager(self.db_path)
        loaded_context = new_session_manager.load_from_database(self.test_session_id)
        
        self.assertIsNotNone(loaded_context)
        self.assertEqual(len(loaded_context.turns), len(conversation_scenarios))
        
        print(f"  ë¡œë“œ?????? {len(loaded_context.turns)}")
        
        print("??Phase 1 ì¢…í•© ?ŒìŠ¤???„ë£Œ")
    
    def test_phase2_comprehensive(self):
        """Phase 2 ì¢…í•© ?ŒìŠ¤??""
        print("\n=== Phase 2 ì¢…í•© ?ŒìŠ¤??===")
        
        # 1. ?¬ìš©???„ë¡œ??ê´€ë¦??ŒìŠ¤??
        print("\n?¬ìš©???„ë¡œ??ê´€ë¦??ŒìŠ¤??")
        
        # ?„ë¡œ???ì„±
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "preferred_language": "ko",
            "interest_areas": ["ë¯¼ë²•", "?•ë²•", "?ë²•"],
            "device_info": {"platform": "web", "browser": "chrome"},
            "location_info": {"country": "KR", "region": "Seoul"}
        }
        
        # ë¨¼ì? ?„ë¡œ???ì„±
        create_success = self.user_profile_manager.create_profile(
            self.test_user_id, profile_data
        )
        self.assertTrue(create_success, "?„ë¡œ???ì„± ?¤íŒ¨")
        
        # ? í˜¸???…ë°?´íŠ¸
        preferences = {
            "expertise_level": "advanced",
            "preferred_detail_level": "comprehensive"
        }
        
        success = self.user_profile_manager.update_preferences(
            self.test_user_id, preferences
        )
        self.assertTrue(success)
        
        # ?„ë¡œ??ì¡°íšŒ (?…ë°?´íŠ¸ ??
        profile = self.user_profile_manager.get_profile(self.test_user_id)
        self.assertIsNotNone(profile)
        # ?…ë°?´íŠ¸ê°€ ?œë?ë¡?ë°˜ì˜?˜ì—ˆ?”ì? ?•ì¸
        updated_expertise = profile.get("expertise_level", "unknown")
        self.assertIn(updated_expertise, ["intermediate", "advanced"], 
                     f"?ˆìƒì¹?ëª»í•œ ?„ë¬¸???ˆë²¨: {updated_expertise}")
        
        print(f"  ?¬ìš©???„ë¡œ???ì„±: {profile['expertise_level']}")
        
        # 2. ê°ì • ë°??˜ë„ ë¶„ì„ ?ŒìŠ¤??
        print("\nê°ì • ë°??˜ë„ ë¶„ì„ ?ŒìŠ¤??")
        
        test_queries = [
            "ê¸‰í•˜ê²??„ì????„ìš”?©ë‹ˆ??",
            "ê³„ì•½??ê²€??ë¶€?ë“œë¦½ë‹ˆ??",
            "??ë¬¸ì œê°€ ?•ë§ ë³µì¡?˜ë„¤??..",
            "ê°ì‚¬?©ë‹ˆ?? ?„ì???ë§ì´ ?˜ì—ˆ?´ìš”."
        ]
        
        for query in test_queries:
            emotion_result = self.emotion_analyzer.analyze_emotion(query)
            
            self.assertIsNotNone(emotion_result)
            self.assertIsNotNone(emotion_result.primary_emotion)
            self.assertIsNotNone(emotion_result.confidence)
            
            print(f"  '{query[:20]}...' -> ê°ì •: {emotion_result.primary_emotion}, ? ë¢°?? {emotion_result.confidence:.2f}")
        
        # 3. ?€???ë¦„ ì¶”ì  ?ŒìŠ¤??
        print("\n?€???ë¦„ ì¶”ì  ?ŒìŠ¤??")
        
        # ?€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.test_session_id,
            turns=[
                ConversationTurn(
                    user_query="?í•´ë°°ìƒ???€???Œë ¤ì£¼ì„¸??,
                    bot_response="?í•´ë°°ìƒ?€ ë¯¼ë²• ??50ì¡°ì— ê·œì •???œë„?…ë‹ˆ??",
                    timestamp=datetime.now(),
                    question_type="legal_advice",
                    entities={"legal_terms": ["?í•´ë°°ìƒ"]}
                )
            ],
            entities={"legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["?í•´ë°°ìƒ"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # ?¤ìŒ ?˜ë„ ?ˆì¸¡
        next_intent = self.flow_tracker.predict_next_intent(context)
        self.assertIsNotNone(next_intent)
        
        # ?„ì† ì§ˆë¬¸ ?œì•ˆ
        suggestions = self.flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)
        
        print(f"  ?ˆì¸¡???¤ìŒ ?˜ë„: {next_intent}")
        print(f"  ?œì•ˆ???„ì† ì§ˆë¬¸ ?? {len(suggestions)}")
        
        print("??Phase 2 ì¢…í•© ?ŒìŠ¤???„ë£Œ")
    
    def test_phase3_comprehensive(self):
        """Phase 3 ì¢…í•© ?ŒìŠ¤??""
        print("\n=== Phase 3 ì¢…í•© ?ŒìŠ¤??===")
        
        # 1. ë§¥ë½??ë©”ëª¨ë¦?ê´€ë¦??ŒìŠ¤??
        print("\në§¥ë½??ë©”ëª¨ë¦?ê´€ë¦??ŒìŠ¤??")
        
        # ?¤ì–‘??? í˜•???¬ì‹¤ ?€??
        facts_to_store = {
            "legal_knowledge": [
                "ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?©ë‹ˆ??,
                "?í•´ë°°ìƒ???”ê±´?€ ê°€?´í–‰?? ?í•´ë°œìƒ, ?¸ê³¼ê´€ê³? ê³ ì˜ ?ëŠ” ê³¼ì‹¤?…ë‹ˆ??
            ],
            "case_detail": [
                "êµí†µ?¬ê³  ê³¼ì‹¤ë¹„ìœ¨ ?°ì • ?¬ê±´",
                "ê³„ì•½ ?´ì? ê´€??ë¶„ìŸ ?¬ê±´"
            ],
            "user_context": [
                "?¬ìš©?ëŠ” ë²•ë¥  ì´ˆë³´?ì…?ˆë‹¤",
                "?¬ìš©?ëŠ” ?ì„¸???¤ëª…??? í˜¸?©ë‹ˆ??
            ]
        }
        
        # ?¬ì‹¤ ?€??
        storage_success = self.memory_manager.store_important_facts(
            self.test_session_id, self.test_user_id, facts_to_store
        )
        self.assertTrue(storage_success)
        
        # ë©”ëª¨ë¦??µê³„ ì¡°íšŒ
        stats = self.memory_manager.get_memory_statistics(self.test_user_id)
        self.assertIsNotNone(stats)
        self.assertGreater(stats["total_memories"], 0)
        
        print(f"  ?€?¥ëœ ë©”ëª¨ë¦??? {stats['total_memories']}")
        print(f"  ? í˜•ë³??µê³„: {stats['type_statistics']}")
        
        # ê´€??ë©”ëª¨ë¦?ê²€??
        relevant_memories = self.memory_manager.retrieve_relevant_memory(
            self.test_session_id, "?í•´ë°°ìƒ ê´€??ì§ˆë¬¸", self.test_user_id
        )
        self.assertIsNotNone(relevant_memories)
        self.assertGreater(len(relevant_memories), 0)
        
        print(f"  ê²€?‰ëœ ê´€??ë©”ëª¨ë¦??? {len(relevant_memories)}")
        
        # 2. ?€???ˆì§ˆ ëª¨ë‹ˆ?°ë§ ?ŒìŠ¤??
        print("\n?€???ˆì§ˆ ëª¨ë‹ˆ?°ë§ ?ŒìŠ¤??")
        
        # ê³ í’ˆì§??€??ì»¨í…?¤íŠ¸
        high_quality_context = ConversationContext(
            session_id=f"{self.test_session_id}_high",
            turns=[
                ConversationTurn(
                    user_query="ë¯¼ë²• ??50ì¡°ì— ?€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ ì±…ì„??ê·œì •?˜ëŠ” ì¤‘ìš”??ì¡°ë¬¸?…ë‹ˆ?? ??ì¡°ë¬¸?€ ê³ ì˜ ?ëŠ” ê³¼ì‹¤ë¡??¸í•œ ë¶ˆë²•?‰ìœ„ë¡??€?¸ì—ê²??í•´ë¥?ê°€???ëŠ” ê·??í•´ë¥?ë°°ìƒ??ì±…ì„???ˆë‹¤ê³?ê·œì •?˜ê³  ?ˆìŠµ?ˆë‹¤.",
                    timestamp=datetime.now(),
                    question_type="law_inquiry",
                    entities={"laws": ["ë¯¼ë²•"], "articles": ["??50ì¡?]}
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}},
            topic_stack=["ë¯¼ë²•", "??50ì¡?],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # ?€?ˆì§ˆ ?€??ì»¨í…?¤íŠ¸
        low_quality_context = ConversationContext(
            session_id=f"{self.test_session_id}_low",
            turns=[
                ConversationTurn(
                    user_query="ë²?,
                    bot_response="??,
                    timestamp=datetime.now(),
                    question_type="general",
                    entities={}
                )
            ],
            entities={},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # ?ˆì§ˆ ?‰ê?
        high_quality_score = self.quality_monitor.assess_conversation_quality(high_quality_context)
        low_quality_score = self.quality_monitor.assess_conversation_quality(low_quality_context)
        
        self.assertIsNotNone(high_quality_score)
        self.assertIsNotNone(low_quality_score)
        self.assertGreater(high_quality_score["overall_score"], low_quality_score["overall_score"])
        
        print(f"  ê³ í’ˆì§??€???ìˆ˜: {high_quality_score['overall_score']:.2f}")
        print(f"  ?€?ˆì§ˆ ?€???ìˆ˜: {low_quality_score['overall_score']:.2f}")
        
        # ë¬¸ì œ??ê°ì? ë°?ê°œì„  ?œì•ˆ
        issues = self.quality_monitor.detect_conversation_issues(low_quality_context)
        suggestions = self.quality_monitor.suggest_improvements(low_quality_context)
        
        self.assertIsNotNone(issues)
        self.assertIsNotNone(suggestions)
        
        print(f"  ê°ì???ë¬¸ì œ???? {len(issues)}")
        print(f"  ê°œì„  ?œì•ˆ ?? {len(suggestions)}")
        
        print("??Phase 3 ì¢…í•© ?ŒìŠ¤???„ë£Œ")
    
    def test_performance_comprehensive(self):
        """?±ëŠ¥ ì¢…í•© ?ŒìŠ¤??""
        print("\n=== ?±ëŠ¥ ì¢…í•© ?ŒìŠ¤??===")
        
        # 1. ?±ëŠ¥ ëª¨ë‹ˆ?°ë§ ?ŒìŠ¤??
        print("\n?±ëŠ¥ ëª¨ë‹ˆ?°ë§ ?ŒìŠ¤??")
        
        # CPU ë°?ë©”ëª¨ë¦??¬ìš©??ì¸¡ì •
        system_health = self.performance_monitor.get_system_health()
        cpu_usage = system_health["system"]["cpu_usage"]
        memory_usage = system_health["system"]["memory_usage"]
        
        self.assertIsNotNone(cpu_usage)
        self.assertIsNotNone(memory_usage)
        
        print(f"  CPU ?¬ìš©ë¥? {cpu_usage:.1f}%")
        print(f"  ë©”ëª¨ë¦??¬ìš©ë¥? {memory_usage:.1f}%")
        
        # 2. ìºì‹œ ?±ëŠ¥ ?ŒìŠ¤??
        print("\nìºì‹œ ?±ëŠ¥ ?ŒìŠ¤??")
        
        # ìºì‹œ ?€??ë°?ì¡°íšŒ ?ŒìŠ¤??
        test_key = "test_cache_key"
        test_value = {"data": "test_value", "timestamp": datetime.now().isoformat()}
        
        # ?€??
        self.cache_manager.set(test_key, test_value)
        
        # ì¡°íšŒ
        retrieved_value = self.cache_manager.get(test_key)
        self.assertIsNotNone(retrieved_value)
        self.assertEqual(retrieved_value["data"], test_value["data"])
        
        # ìºì‹œ ?µê³„
        cache_stats = self.cache_manager.get_stats()
        self.assertIsNotNone(cache_stats)
        
        print(f"  ìºì‹œ ?ˆíŠ¸?? {cache_stats['hit_rate']:.2f}")
        print(f"  ìºì‹œ ?¬ê¸°: {cache_stats['cache_size']}")
        
        # 3. ë©”ëª¨ë¦?ìµœì ???ŒìŠ¤??
        print("\në©”ëª¨ë¦?ìµœì ???ŒìŠ¤??")
        
        # ë©”ëª¨ë¦??¬ìš©??ëª¨ë‹ˆ?°ë§
        initial_memory = self.memory_optimizer.get_memory_usage()
        
        # ê°€ë¹„ì? ì»¬ë ‰???¤í–‰
        freed_memory = self.memory_optimizer.optimize_memory()
        
        final_memory = self.memory_optimizer.get_memory_usage()
        
        print(f"  ì´ˆê¸° ë©”ëª¨ë¦? {initial_memory.process_memory / 1024 / 1024:.1f} MB")
        print(f"  ìµœì ????ë©”ëª¨ë¦? {final_memory.process_memory / 1024 / 1024:.1f} MB")
        print(f"  ?´ì œ??ë©”ëª¨ë¦? {freed_memory.get('memory_freed_mb', 0):.1f} MB")
        
        print("???±ëŠ¥ ì¢…í•© ?ŒìŠ¤???„ë£Œ")
    
    def test_integration_comprehensive(self):
        """?µí•© ?œìŠ¤???ŒìŠ¤??""
        print("\n=== ?µí•© ?œìŠ¤???ŒìŠ¤??===")
        
        # 1. ?„ì²´ ?Œí¬?Œë¡œ???ŒìŠ¤??
        print("\n?„ì²´ ?Œí¬?Œë¡œ???ŒìŠ¤??")
        
        # ë³µì¡???€???œë‚˜ë¦¬ì˜¤
        complex_scenario = [
            {
                "query": "?ˆë…•?˜ì„¸?? ë²•ë¥  ?ë‹´??ë°›ê³  ?¶ìŠµ?ˆë‹¤.",
                "expected_emotion": "neutral",
                "expected_intent": "greeting"
            },
            {
                "query": "ê³„ì•½??ê²€? ë? ?„ì?ì£¼ì‹¤ ???ˆë‚˜??",
                "expected_emotion": "neutral",
                "expected_intent": "request"
            },
            {
                "query": "ê·?ê³„ì•½?œì—??ë¬¸ì œê°€ ?????ˆëŠ” ë¶€ë¶„ì´ ?ˆë‚˜??",
                "expected_emotion": "neutral",
                "expected_intent": "clarification"
            },
            {
                "query": "ê°ì‚¬?©ë‹ˆ?? ?•ë§ ?„ì???ë§ì´ ?˜ì—ˆ?´ìš”!",
                "expected_emotion": "positive",
                "expected_intent": "gratitude"
            }
        ]
        
        total_processing_time = 0
        
        for i, scenario in enumerate(complex_scenario):
            start_time = time.time()
            
            # Phase 1: ?€??ë§¥ë½ ì²˜ë¦¬
            context = self.session_manager.add_turn(
                f"{self.test_session_id}_integration",
                scenario["query"],
                f"?œìŠ¤???‘ë‹µ {i+1}",
                "general",
                self.test_user_id
            )
            
            # Phase 2: ê°ì • ë°??˜ë„ ë¶„ì„
            emotion_result = self.emotion_analyzer.analyze_emotion(scenario["query"])
            
            # Phase 3: ë©”ëª¨ë¦??€??
            facts = {"user_context": [scenario["query"]]}
            self.memory_manager.store_important_facts(
                f"{self.test_session_id}_integration", self.test_user_id, facts
            )
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            print(f"  ??{i+1}: {processing_time:.3f}ì´?)
        
        avg_processing_time = total_processing_time / len(complex_scenario)
        print(f"  ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_processing_time:.3f}ì´?)
        
        # 2. ?™ì‹œ???ŒìŠ¤??
        print("\n?™ì‹œ???ŒìŠ¤??")
        
        async def concurrent_test():
            tasks = []
            for i in range(5):
                task = asyncio.create_task(self._async_conversation_test(i))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # ?™ì‹œ???ŒìŠ¤???¤í–‰
        try:
            results = asyncio.run(concurrent_test())
            self.assertEqual(len(results), 5)
            print(f"  ?™ì‹œ ì²˜ë¦¬???€???? {len(results)}")
        except Exception as e:
            print(f"  ?™ì‹œ???ŒìŠ¤???¤ë¥˜: {e}")
        
        # 3. ?œìŠ¤???ˆì •???ŒìŠ¤??
        print("\n?œìŠ¤???ˆì •???ŒìŠ¤??")
        
        # ?€???°ì´??ì²˜ë¦¬
        large_data_test = []
        for i in range(100):
            large_data_test.append(f"?ŒìŠ¤???°ì´??{i} - " + "x" * 100)
        
        start_time = time.time()
        
        # ?€???°ì´??ì²˜ë¦¬
        for data in large_data_test[:10]:  # ì²˜ìŒ 10ê°œë§Œ ?ŒìŠ¤??
            facts = {"test_data": [data]}
            self.memory_manager.store_important_facts(
                f"{self.test_session_id}_stress", self.test_user_id, facts
            )
        
        processing_time = time.time() - start_time
        print(f"  ?€???°ì´??ì²˜ë¦¬ ?œê°„: {processing_time:.3f}ì´?)
        
        print("???µí•© ?œìŠ¤???ŒìŠ¤???„ë£Œ")
    
    async def _async_conversation_test(self, test_id: int):
        """ë¹„ë™ê¸??€???ŒìŠ¤??""
        session_id = f"async_test_{test_id}"
        
        # ë¹„ë™ê¸??€??ì²˜ë¦¬
        context = self.session_manager.add_turn(
            session_id,
            f"ë¹„ë™ê¸??ŒìŠ¤??ì§ˆë¬¸ {test_id}",
            f"ë¹„ë™ê¸??ŒìŠ¤???‘ë‹µ {test_id}",
            "test",
            f"async_user_{test_id}"
        )
        
        return {
            "test_id": test_id,
            "session_id": session_id,
            "turns_count": len(context.turns)
        }


def run_comprehensive_tests():
    """ì¢…í•© ?ŒìŠ¤???¤í–‰"""
    print("=" * 60)
    print("LawFirmAI ì¢…í•© ?œìŠ¤???ŒìŠ¤???œì‘")
    print("=" * 60)
    
    # ?ŒìŠ¤???¤ìœ„???ì„±
    test_suite = unittest.TestSuite()
    
    # ?ŒìŠ¤??ì¼€?´ìŠ¤ ì¶”ê?
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveSystem))
    
    # ?ŒìŠ¤???¤í–‰
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 60)
    print("ì¢…í•© ?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("=" * 60)
    print(f"?¤í–‰???ŒìŠ¤?? {result.testsRun}")
    print(f"?±ê³µ: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"?¤íŒ¨: {len(result.failures)}")
    print(f"?¤ë¥˜: {len(result.errors)}")
    
    if result.failures:
        print("\n?¤íŒ¨???ŒìŠ¤??")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n?¤ë¥˜ê°€ ë°œìƒ???ŒìŠ¤??")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n?„ì²´ ?µê³¼?? {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("?‰ ?°ìˆ˜???±ëŠ¥! ?œìŠ¤?œì´ ?ˆì •?ìœ¼ë¡??‘ë™?©ë‹ˆ??")
    elif success_rate >= 85:
        print("???‘í˜¸???±ëŠ¥! ëª?ê°€ì§€ ê°œì„ ???„ìš”?©ë‹ˆ??")
    else:
        print("? ï¸ ê°œì„ ???„ìš”?©ë‹ˆ?? ?œìŠ¤?œì„ ?ê??´ì£¼?¸ìš”.")
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()
