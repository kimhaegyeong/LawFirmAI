# -*- coding: utf-8 -*-
"""
LawFirmAI ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??
?€???°ì´??ì²˜ë¦¬ ë°??™ì‹œ???ŒìŠ¤??
"""

import unittest
import asyncio
import time
import tempfile
import os
import sys
import threading
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor
from source.utils.performance_optimizer import PerformanceMonitor, MemoryOptimizer, CacheManager
from source.data.conversation_store import ConversationStore


class TestStressSystem(unittest.TestCase):
    """?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stress.db")
        
        # ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.memory_manager = ContextualMemoryManager(self.session_manager.conversation_store)
        self.quality_monitor = ConversationQualityMonitor(self.session_manager.conversation_store)
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager(max_size=1000, ttl=3600)
        
        # ?ŒìŠ¤???°ì´??
        self.test_users = [f"stress_user_{i}" for i in range(50)]
        self.test_sessions = [f"stress_session_{i}" for i in range(100)]
        
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_scale_conversations(self):
        """?€ê·œëª¨ ?€??ì²˜ë¦¬ ?ŒìŠ¤??""
        print("\n=== ?€ê·œëª¨ ?€??ì²˜ë¦¬ ?ŒìŠ¤??===")
        
        # ?€???€???°ì´???ì„±
        conversation_data = []
        for i in range(500):  # 500ê°œì˜ ?€???ì„±
            conversation_data.append({
                "session_id": f"large_scale_session_{i}",
                "user_id": f"large_scale_user_{i % 50}",
                "query": f"ë²•ë¥  ì§ˆë¬¸ {i} - ?í•´ë°°ìƒ, ê³„ì•½, ?•ë²• ê´€??ë¬¸ì˜",
                "response": f"ë²•ë¥  ?µë? {i} - ?ì„¸??ë²•ë¥  ?¤ëª…ê³??ë? ?¸ìš©",
                "question_type": random.choice(["legal_advice", "law_inquiry", "procedure_guide"])
            })
        
        start_time = time.time()
        
        # ?€???€??ì²˜ë¦¬
        processed_count = 0
        for data in conversation_data:
            try:
                # ??ì¶”ê?
                context = self.session_manager.add_turn(
                    data["session_id"],
                    data["query"],
                    data["response"],
                    data["question_type"],
                    data["user_id"]
                )
                
                # ë©”ëª¨ë¦??€??
                facts = {
                    "legal_knowledge": [data["response"]],
                    "user_context": [data["query"]]
                }
                self.memory_manager.store_important_facts(
                    data["session_id"], data["user_id"], facts
                )
                
                processed_count += 1
                
                # ì§„í–‰ë¥??œì‹œ
                if processed_count % 100 == 0:
                    print(f"  ì²˜ë¦¬???€?? {processed_count}/500")
                    
            except Exception as e:
                print(f"  ?¤ë¥˜ ë°œìƒ (?€??{processed_count}): {e}")
        
        processing_time = time.time() - start_time
        
        print(f"\n?€ê·œëª¨ ?€??ì²˜ë¦¬ ê²°ê³¼:")
        print(f"  ì²˜ë¦¬???€???? {processed_count}")
        print(f"  ì´?ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)
        print(f"  ?‰ê·  ì²˜ë¦¬ ?œê°„: {processing_time/processed_count:.3f}ì´??€??)
        print(f"  ì²˜ë¦¬ ?ë„: {processed_count/processing_time:.1f}?€??ì´?)
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertGreaterEqual(processed_count, 450)  # ìµœì†Œ 90% ?±ê³µ
        self.assertLess(processing_time, 60)  # 60ì´??´ë‚´ ?„ë£Œ
        
        print("???€ê·œëª¨ ?€??ì²˜ë¦¬ ?ŒìŠ¤???„ë£Œ")
    
    def test_concurrent_sessions(self):
        """?™ì‹œ ?¸ì…˜ ì²˜ë¦¬ ?ŒìŠ¤??""
        print("\n=== ?™ì‹œ ?¸ì…˜ ì²˜ë¦¬ ?ŒìŠ¤??===")
        
        def process_session(session_data):
            """?¸ì…˜ ì²˜ë¦¬ ?¨ìˆ˜"""
            try:
                # ??ì¶”ê?
                context = self.session_manager.add_turn(
                    session_data["session_id"],
                    session_data["query"],
                    session_data["response"],
                    session_data["question_type"],
                    session_data["user_id"]
                )
                
                # ë©”ëª¨ë¦??€??
                facts = {"test_data": [session_data["query"]]}
                self.memory_manager.store_important_facts(
                    session_data["session_id"], session_data["user_id"], facts
                )
                
                return {
                    "session_id": session_data["session_id"],
                    "success": True,
                    "turns_count": len(context.turns)
                }
                
            except Exception as e:
                return {
                    "session_id": session_data["session_id"],
                    "success": False,
                    "error": str(e)
                }
        
        # ?™ì‹œ ì²˜ë¦¬???¸ì…˜ ?°ì´???ì„±
        concurrent_sessions = []
        for i in range(100):
            concurrent_sessions.append({
                "session_id": f"concurrent_session_{i}",
                "user_id": f"concurrent_user_{i % 20}",
                "query": f"?™ì‹œ ì²˜ë¦¬ ?ŒìŠ¤??ì§ˆë¬¸ {i}",
                "response": f"?™ì‹œ ì²˜ë¦¬ ?ŒìŠ¤???‘ë‹µ {i}",
                "question_type": "test"
            })
        
        start_time = time.time()
        
        # ThreadPoolExecutorë¥??¬ìš©???™ì‹œ ì²˜ë¦¬
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_session, session) for session in concurrent_sessions]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        processing_time = time.time() - start_time
        
        # ê²°ê³¼ ë¶„ì„
        successful_sessions = [r for r in results if r["success"]]
        failed_sessions = [r for r in results if not r["success"]]
        
        print(f"\n?™ì‹œ ?¸ì…˜ ì²˜ë¦¬ ê²°ê³¼:")
        print(f"  ì´??¸ì…˜ ?? {len(concurrent_sessions)}")
        print(f"  ?±ê³µ???¸ì…˜: {len(successful_sessions)}")
        print(f"  ?¤íŒ¨???¸ì…˜: {len(failed_sessions)}")
        print(f"  ?±ê³µë¥? {len(successful_sessions)/len(concurrent_sessions)*100:.1f}%")
        print(f"  ì´?ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)
        print(f"  ?‰ê·  ì²˜ë¦¬ ?œê°„: {processing_time/len(concurrent_sessions):.3f}ì´??¸ì…˜")
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertGreaterEqual(len(successful_sessions), 90)  # ìµœì†Œ 90% ?±ê³µ
        self.assertLess(processing_time, 30)  # 30ì´??´ë‚´ ?„ë£Œ
        
        if failed_sessions:
            print(f"\n?¤íŒ¨???¸ì…˜??")
            for failed in failed_sessions[:5]:  # ì²˜ìŒ 5ê°œë§Œ ?œì‹œ
                print(f"  - {failed['session_id']}: {failed['error']}")
        
        print("???™ì‹œ ?¸ì…˜ ì²˜ë¦¬ ?ŒìŠ¤???„ë£Œ")
    
    def test_memory_stress(self):
        """ë©”ëª¨ë¦??¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??""
        print("\n=== ë©”ëª¨ë¦??¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??===")
        
        # ì´ˆê¸° ë©”ëª¨ë¦??¬ìš©??
        initial_memory_obj = self.memory_optimizer.get_memory_usage()
        initial_memory = initial_memory_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
        print(f"ì´ˆê¸° ë©”ëª¨ë¦??¬ìš©?? {initial_memory:.1f} MB")
        
        # ?€??ë©”ëª¨ë¦?? ë‹¹
        large_data_sets = []
        for i in range(100):
            # ê°ê° 1MB ?•ë„???°ì´???ì„±
            large_data = {
                "id": i,
                "content": "x" * 1000000,  # 1MB ë¬¸ì??
                "metadata": {"created_at": datetime.now().isoformat()},
                "entities": [f"entity_{j}" for j in range(100)]
            }
            large_data_sets.append(large_data)
        
        # ë©”ëª¨ë¦??¬ìš©??ëª¨ë‹ˆ?°ë§
        peak_memory = initial_memory
        for i, data in enumerate(large_data_sets):
            # ?°ì´??ì²˜ë¦¬ ?œë??ˆì´??
            facts = {"large_data": [data["content"]]}
            self.memory_manager.store_important_facts(
                f"memory_stress_session_{i}", f"memory_stress_user_{i}", facts
            )
            
            # ë©”ëª¨ë¦??¬ìš©??ì²´í¬
            current_memory_obj = self.memory_optimizer.get_memory_usage()
            current_memory = current_memory_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
            peak_memory = max(peak_memory, current_memory)
            
            if i % 20 == 0:
                print(f"  ì²˜ë¦¬???°ì´?? {i+1}/100, ?„ì¬ ë©”ëª¨ë¦? {current_memory:.1f} MB")
        
        # ë©”ëª¨ë¦?ìµœì ??
        print("\në©”ëª¨ë¦?ìµœì ???¤í–‰...")
        freed_memory_result = self.memory_optimizer.optimize_memory()
        freed_memory = freed_memory_result["memory_freed_mb"]
        
        final_memory_obj = self.memory_optimizer.get_memory_usage()
        final_memory = final_memory_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
        
        print(f"\në©”ëª¨ë¦??¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??ê²°ê³¼:")
        print(f"  ì´ˆê¸° ë©”ëª¨ë¦? {initial_memory:.1f} MB")
        print(f"  ìµœë? ë©”ëª¨ë¦? {peak_memory:.1f} MB")
        print(f"  ìµœì¢… ë©”ëª¨ë¦? {final_memory:.1f} MB")
        print(f"  ?´ì œ??ë©”ëª¨ë¦? {freed_memory:.1f} MB")
        print(f"  ë©”ëª¨ë¦?ì¦ê??? {peak_memory - initial_memory:.1f} MB")
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertLess(peak_memory - initial_memory, 500)  # 500MB ?´í•˜ ì¦ê?
        self.assertGreater(freed_memory, 0)  # ë©”ëª¨ë¦??´ì œ ?•ì¸
        
        print("??ë©”ëª¨ë¦??¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???„ë£Œ")
    
    def test_cache_stress(self):
        """ìºì‹œ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??""
        print("\n=== ìºì‹œ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??===")
        
        # ?€??ìºì‹œ ?°ì´???ì„± ë°??€??
        cache_data_count = 1000
        start_time = time.time()
        
        print(f"ìºì‹œ ?°ì´??{cache_data_count}ê°??ì„± ì¤?..")
        
        for i in range(cache_data_count):
            cache_key = f"stress_cache_key_{i}"
            cache_value = {
                "id": i,
                "data": f"?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???°ì´??{i}",
                "timestamp": datetime.now().isoformat(),
                "large_content": "x" * 1000  # 1KB ?°ì´??
            }
            
            self.cache_manager.set(cache_key, cache_value)
            
            if i % 200 == 0:
                print(f"  ?€?¥ëœ ìºì‹œ: {i+1}/{cache_data_count}")
        
        storage_time = time.time() - start_time
        
        # ìºì‹œ ì¡°íšŒ ?ŒìŠ¤??
        print(f"\nìºì‹œ ì¡°íšŒ ?ŒìŠ¤???œì‘...")
        start_time = time.time()
        
        hit_count = 0
        miss_count = 0
        
        for i in range(cache_data_count):
            cache_key = f"stress_cache_key_{i}"
            retrieved_value = self.cache_manager.get(cache_key)
            
            if retrieved_value:
                hit_count += 1
            else:
                miss_count += 1
            
            if i % 200 == 0:
                print(f"  ì¡°íšŒ??ìºì‹œ: {i+1}/{cache_data_count}")
        
        retrieval_time = time.time() - start_time
        
        # ìºì‹œ ?µê³„
        cache_stats = self.cache_manager.get_stats()
        
        print(f"\nìºì‹œ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??ê²°ê³¼:")
        print(f"  ?€?¥ëœ ìºì‹œ ?? {cache_data_count}")
        print(f"  ?€???œê°„: {storage_time:.2f}ì´?)
        print(f"  ì¡°íšŒ ?œê°„: {retrieval_time:.2f}ì´?)
        print(f"  ?ˆíŠ¸ ?? {hit_count}")
        print(f"  ë¯¸ìŠ¤ ?? {miss_count}")
        print(f"  ?ˆíŠ¸?? {hit_count/(hit_count+miss_count)*100:.1f}%")
        print(f"  ìºì‹œ ?¬ê¸°: {cache_stats['cache_size']}")
        print(f"  ?‰ê·  ?€???œê°„: {storage_time/cache_data_count*1000:.2f}ms/??ª©")
        print(f"  ?‰ê·  ì¡°íšŒ ?œê°„: {retrieval_time/cache_data_count*1000:.2f}ms/??ª©")
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertGreater(hit_count, cache_data_count * 0.8)  # 80% ?´ìƒ ?ˆíŠ¸??
        self.assertLess(storage_time, 10)  # 10ì´??´ë‚´ ?€??
        self.assertLess(retrieval_time, 5)  # 5ì´??´ë‚´ ì¡°íšŒ
        
        print("??ìºì‹œ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???„ë£Œ")
    
    def test_database_stress(self):
        """?°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??""
        print("\n=== ?°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??===")
        
        # ?€???°ì´?°ë² ?´ìŠ¤ ?‘ì—…
        operations_count = 1000
        start_time = time.time()
        
        print(f"?°ì´?°ë² ?´ìŠ¤ ?‘ì—… {operations_count}ê°??¤í–‰ ì¤?..")
        
        successful_operations = 0
        
        for i in range(operations_count):
            try:
                # ?¸ì…˜ ?ì„± ë°??€??
                session_data = {
                    "session_id": f"db_stress_session_{i}",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "topic_stack": [f"topic_{i}"],
                    "metadata": {"user_id": f"db_stress_user_{i % 100}"},
                    "turns": [
                        {
                            "user_query": f"?°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??ì§ˆë¬¸ {i}",
                            "bot_response": f"?°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???‘ë‹µ {i}",
                            "timestamp": datetime.now().isoformat(),
                            "question_type": "test",
                            "entities": {"test_entities": [f"entity_{i}"]}
                        }
                    ],
                    "entities": {"test_entities": [f"entity_{i}"]}
                }
                
                # ?¸ì…˜ ?€??
                success = self.session_manager.conversation_store.save_session(session_data)
                if success:
                    successful_operations += 1
                
                # ì§„í–‰ë¥??œì‹œ
                if i % 200 == 0:
                    print(f"  ?„ë£Œ???‘ì—…: {i+1}/{operations_count}")
                    
            except Exception as e:
                print(f"  ?¤ë¥˜ ë°œìƒ (?‘ì—… {i}): {e}")
        
        processing_time = time.time() - start_time
        
        # ?°ì´?°ë² ?´ìŠ¤ ?µê³„ ì¡°íšŒ
        try:
            stats = self.session_manager.conversation_store.get_statistics()
            print(f"\n?°ì´?°ë² ?´ìŠ¤ ?µê³„:")
            print(f"  ì´??¸ì…˜ ?? {stats.get('total_sessions', 0)}")
            print(f"  ì´????? {stats.get('total_turns', 0)}")
            print(f"  ì´??”í‹°???? {stats.get('total_entities', 0)}")
        except Exception as e:
            print(f"  ?µê³„ ì¡°íšŒ ?¤ë¥˜: {e}")
        
        print(f"\n?°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??ê²°ê³¼:")
        print(f"  ì´??‘ì—… ?? {operations_count}")
        print(f"  ?±ê³µ???‘ì—…: {successful_operations}")
        print(f"  ?¤íŒ¨???‘ì—…: {operations_count - successful_operations}")
        print(f"  ?±ê³µë¥? {successful_operations/operations_count*100:.1f}%")
        print(f"  ì´?ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)
        print(f"  ?‰ê·  ì²˜ë¦¬ ?œê°„: {processing_time/operations_count*1000:.2f}ms/?‘ì—…")
        print(f"  ì²˜ë¦¬ ?ë„: {operations_count/processing_time:.1f}?‘ì—…/ì´?)
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertGreaterEqual(successful_operations, operations_count * 0.9)  # 90% ?´ìƒ ?±ê³µ
        self.assertLess(processing_time, 60)  # 60ì´??´ë‚´ ?„ë£Œ
        
        print("???°ì´?°ë² ?´ìŠ¤ ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???„ë£Œ")
    
    def test_error_recovery(self):
        """?¤ë¥˜ ë³µêµ¬ ?ŒìŠ¤??""
        print("\n=== ?¤ë¥˜ ë³µêµ¬ ?ŒìŠ¤??===")
        
        # ?¤ì–‘???¤ë¥˜ ?í™© ?œë??ˆì´??
        error_scenarios = [
            {
                "name": "?˜ëª»???¸ì…˜ ID",
                "test": lambda: self.session_manager.add_turn("", "test", "test", "test", "test")
            },
            {
                "name": "?˜ëª»???¬ìš©??ID",
                "test": lambda: self.session_manager.add_turn("test", "test", "test", "test", "")
            },
            {
                "name": "?˜ëª»??ë©”ëª¨ë¦??°ì´??,
                "test": lambda: self.memory_manager.store_important_facts("test", "test", None)
            },
            {
                "name": "?˜ëª»??ìºì‹œ ??,
                "test": lambda: self.cache_manager.get(None)
            }
        ]
        
        recovery_success_count = 0
        
        for scenario in error_scenarios:
            print(f"\n?¤ë¥˜ ?œë‚˜ë¦¬ì˜¤: {scenario['name']}")
            
            try:
                # ?¤ë¥˜ ë°œìƒ ?œë„
                result = scenario['test']()
                print(f"  ?ˆìƒì¹?ëª»í•œ ?±ê³µ: {result}")
            except Exception as e:
                print(f"  ?ˆìƒ???¤ë¥˜: {type(e).__name__}")
                
                # ?œìŠ¤?œì´ ?¬ì „???‘ë™?˜ëŠ”ì§€ ?•ì¸
                try:
                    # ?•ìƒ?ì¸ ?‘ì—… ?˜í–‰
                    context = self.session_manager.add_turn(
                        "recovery_test_session",
                        "ë³µêµ¬ ?ŒìŠ¤??ì§ˆë¬¸",
                        "ë³µêµ¬ ?ŒìŠ¤???‘ë‹µ",
                        "test",
                        "recovery_test_user"
                    )
                    
                    if context and len(context.turns) > 0:
                        print(f"  ???œìŠ¤??ë³µêµ¬ ?±ê³µ")
                        recovery_success_count += 1
                    else:
                        print(f"  ???œìŠ¤??ë³µêµ¬ ?¤íŒ¨")
                        
                except Exception as recovery_error:
                    print(f"  ??ë³µêµ¬ ì¤??¤ë¥˜: {recovery_error}")
        
        print(f"\n?¤ë¥˜ ë³µêµ¬ ?ŒìŠ¤??ê²°ê³¼:")
        print(f"  ?ŒìŠ¤?¸ëœ ?œë‚˜ë¦¬ì˜¤: {len(error_scenarios)}")
        print(f"  ?±ê³µ??ë³µêµ¬: {recovery_success_count}")
        print(f"  ë³µêµ¬ ?±ê³µë¥? {recovery_success_count/len(error_scenarios)*100:.1f}%")
        
        # ê²°ê³¼ ê²€ì¦?
        self.assertGreaterEqual(recovery_success_count, len(error_scenarios) * 0.8)  # 80% ?´ìƒ ë³µêµ¬ ?±ê³µ
        
        print("???¤ë¥˜ ë³µêµ¬ ?ŒìŠ¤???„ë£Œ")


def run_stress_tests():
    """?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???¤í–‰"""
    print("=" * 60)
    print("LawFirmAI ?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤???œì‘")
    print("=" * 60)
    
    # ?ŒìŠ¤???¤ìœ„???ì„±
    test_suite = unittest.TestSuite()
    
    # ?ŒìŠ¤??ì¼€?´ìŠ¤ ì¶”ê?
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStressSystem))
    
    # ?ŒìŠ¤???¤í–‰
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 60)
    print("?¤íŠ¸?ˆìŠ¤ ?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
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
    
    if success_rate >= 90:
        print("?‰ ?°ìˆ˜???¤íŠ¸?ˆìŠ¤ ?€??„±! ?œìŠ¤?œì´ ?ˆì •?ìœ¼ë¡??‘ë™?©ë‹ˆ??")
    elif success_rate >= 75:
        print("???‘í˜¸???¤íŠ¸?ˆìŠ¤ ?€??„±! ?¼ë? ê°œì„ ???„ìš”?©ë‹ˆ??")
    else:
        print("? ï¸ ?¤íŠ¸?ˆìŠ¤ ?€??„± ê°œì„ ???„ìš”?©ë‹ˆ?? ?œìŠ¤?œì„ ?ê??´ì£¼?¸ìš”.")
    
    return result


if __name__ == "__main__":
    run_stress_tests()
