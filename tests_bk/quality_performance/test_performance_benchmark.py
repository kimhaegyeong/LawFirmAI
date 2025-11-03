# -*- coding: utf-8 -*-
"""
LawFirmAI ?±ëŠ¥ ë²¤ì¹˜ë§ˆí¬ ?ŒìŠ¤??
?‘ë‹µ ?œê°„, ì²˜ë¦¬?? ë¦¬ì†Œ???¬ìš©??ì¸¡ì •
"""

import unittest
import time
import tempfile
import os
import sys
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

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


class TestPerformanceBenchmark(unittest.TestCase):
    """?±ëŠ¥ ë²¤ì¹˜ë§ˆí¬ ?ŒìŠ¤??""
    
    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_benchmark.db")
        
        # ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
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
        self.cache_manager = CacheManager(max_size=1000, ttl=3600)
        
        # ë²¤ì¹˜ë§ˆí¬ ?°ì´??
        self.benchmark_queries = [
            "?í•´ë°°ìƒ???€???ì„¸???¤ëª…?´ì£¼?¸ìš”",
            "ê³„ì•½ ?´ì? ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            "ë¯¼ë²• ??50ì¡°ì˜ ?”ê±´???Œë ¤ì£¼ì„¸??,
            "ë¶ˆë²•?‰ìœ„???±ë¦½?”ê±´?€ ë¬´ì—‡?¸ê???",
            "ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²??°ì •?˜ë‚˜??"
        ]
        
    def tearDown(self):
        """?ŒìŠ¤???•ë¦¬"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_response_time_benchmark(self):
        """?‘ë‹µ ?œê°„ ë²¤ì¹˜ë§ˆí¬"""
        print("\n=== ?‘ë‹µ ?œê°„ ë²¤ì¹˜ë§ˆí¬ ===")
        
        response_times = []
        
        for i, query in enumerate(self.benchmark_queries):
            # ê°?ì¿¼ë¦¬ë³??‘ë‹µ ?œê°„ ì¸¡ì •
            start_time = time.time()
            
            # Phase 1: ?€??ë§¥ë½ ì²˜ë¦¬
            context = self.session_manager.add_turn(
                f"benchmark_session_{i}",
                query,
                f"ë²¤ì¹˜ë§ˆí¬ ?‘ë‹µ {i}",
                "legal_advice",
                f"benchmark_user_{i}"
            )
            
            # Phase 2: ê°ì • ë¶„ì„
            emotion_result = self.emotion_analyzer.analyze_emotion(query)
            
            # Phase 3: ë©”ëª¨ë¦??€??
            facts = {"legal_knowledge": [f"ë²¤ì¹˜ë§ˆí¬ ?¬ì‹¤ {i}"]}
            self.memory_manager.store_important_facts(
                f"benchmark_session_{i}", f"benchmark_user_{i}", facts
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"ì¿¼ë¦¬ {i+1}: {response_time:.3f}ì´?)
        
        # ?µê³„ ê³„ì‚°
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        print(f"\n?‘ë‹µ ?œê°„ ?µê³„:")
        print(f"  ?‰ê· : {avg_response_time:.3f}ì´?)
        print(f"  ì¤‘ê°„ê°? {median_response_time:.3f}ì´?)
        print(f"  ìµœì†Œ: {min_response_time:.3f}ì´?)
        print(f"  ìµœë?: {max_response_time:.3f}ì´?)
        print(f"  ?œì??¸ì°¨: {std_response_time:.3f}ì´?)
        
        # ?±ëŠ¥ ê¸°ì? ê²€ì¦?
        self.assertLess(avg_response_time, 2.0)  # ?‰ê·  2ì´??´ë‚´
        self.assertLess(max_response_time, 5.0)  # ìµœë? 5ì´??´ë‚´
        
        print("???‘ë‹µ ?œê°„ ë²¤ì¹˜ë§ˆí¬ ?„ë£Œ")
    
    def test_throughput_benchmark(self):
        """ì²˜ë¦¬??ë²¤ì¹˜ë§ˆí¬"""
        print("\n=== ì²˜ë¦¬??ë²¤ì¹˜ë§ˆí¬ ===")
        
        # ?¤ì–‘??ë¶€???˜ì??ì„œ ì²˜ë¦¬??ì¸¡ì •
        load_levels = [10, 50, 100, 200]
        throughput_results = []
        
        for load in load_levels:
            print(f"\në¶€???˜ì?: {load}ê°??”ì²­")
            
            start_time = time.time()
            successful_requests = 0
            
            for i in range(load):
                try:
                    # ?”ì²­ ì²˜ë¦¬
                    context = self.session_manager.add_turn(
                        f"throughput_session_{load}_{i}",
                        f"ì²˜ë¦¬???ŒìŠ¤??ì§ˆë¬¸ {i}",
                        f"ì²˜ë¦¬???ŒìŠ¤???‘ë‹µ {i}",
                        "test",
                        f"throughput_user_{i % 10}"
                    )
                    
                    # ë©”ëª¨ë¦??€??
                    facts = {"test_data": [f"ì²˜ë¦¬???ŒìŠ¤???°ì´??{i}"]}
                    self.memory_manager.store_important_facts(
                        f"throughput_session_{load}_{i}", f"throughput_user_{i % 10}", facts
                    )
                    
                    successful_requests += 1
                    
                except Exception as e:
                    print(f"  ?”ì²­ {i} ?¤íŒ¨: {e}")
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = successful_requests / total_time if total_time > 0 else 0
            
            throughput_results.append({
                "load": load,
                "successful_requests": successful_requests,
                "total_time": total_time,
                "throughput": throughput
            })
            
            print(f"  ?±ê³µ???”ì²­: {successful_requests}/{load}")
            print(f"  ì´?ì²˜ë¦¬ ?œê°„: {total_time:.2f}ì´?)
            print(f"  ì²˜ë¦¬?? {throughput:.1f}?”ì²­/ì´?)
        
        # ì²˜ë¦¬??ë¶„ì„
        print(f"\nì²˜ë¦¬??ë¶„ì„:")
        for result in throughput_results:
            print(f"  ë¶€??{result['load']}: {result['throughput']:.1f}?”ì²­/ì´?)
        
        # ?±ëŠ¥ ê¸°ì? ê²€ì¦?
        max_throughput = max(r["throughput"] for r in throughput_results)
        self.assertGreater(max_throughput, 10)  # ìµœì†Œ 10?”ì²­/ì´?
        
        print("??ì²˜ë¦¬??ë²¤ì¹˜ë§ˆí¬ ?„ë£Œ")
    
    def test_memory_usage_benchmark(self):
        """ë©”ëª¨ë¦??¬ìš©??ë²¤ì¹˜ë§ˆí¬"""
        print("\n=== ë©”ëª¨ë¦??¬ìš©??ë²¤ì¹˜ë§ˆí¬ ===")
        
        # ì´ˆê¸° ë©”ëª¨ë¦??¬ìš©??
        initial_memory_obj = self.memory_optimizer.get_memory_usage()
        initial_memory = initial_memory_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
        print(f"ì´ˆê¸° ë©”ëª¨ë¦??¬ìš©?? {initial_memory:.1f} MB")
        
        memory_usage_history = [initial_memory]
        
        # ?¨ê³„ë³?ë©”ëª¨ë¦??¬ìš©??ì¸¡ì •
        stages = [
            ("?¸ì…˜ ê´€ë¦¬ì ì´ˆê¸°??, lambda: None),
            ("100ê°??¸ì…˜ ?ì„±", self._create_100_sessions),
            ("1000ê°?ë©”ëª¨ë¦??€??, self._store_1000_memories),
            ("ìºì‹œ 500ê°???ª©", self._cache_500_items),
            ("ë©”ëª¨ë¦?ìµœì ??, self._optimize_memory)
        ]
        
        for stage_name, stage_func in stages:
            print(f"\n{stage_name}...")
            
            # ë©”ëª¨ë¦??¬ìš©??ì¸¡ì •
            memory_before_obj = self.memory_optimizer.get_memory_usage()
            memory_before = memory_before_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
            
            # ?‘ì—… ?¤í–‰
            stage_func()
            
            memory_after_obj = self.memory_optimizer.get_memory_usage()
            memory_after = memory_after_obj.process_memory / 1024 / 1024  # MBë¡?ë³€??
            memory_increase = memory_after - memory_before
            
            memory_usage_history.append(memory_after)
            
            print(f"  ë©”ëª¨ë¦??¬ìš©?? {memory_after:.1f} MB")
            print(f"  ë©”ëª¨ë¦?ì¦ê??? {memory_increase:.1f} MB")
        
        # ë©”ëª¨ë¦??¬ìš©??ë¶„ì„
        peak_memory = max(memory_usage_history)
        final_memory = memory_usage_history[-1]
        total_memory_increase = final_memory - initial_memory
        
        print(f"\në©”ëª¨ë¦??¬ìš©??ë¶„ì„:")
        print(f"  ì´ˆê¸° ë©”ëª¨ë¦? {initial_memory:.1f} MB")
        print(f"  ìµœë? ë©”ëª¨ë¦? {peak_memory:.1f} MB")
        print(f"  ìµœì¢… ë©”ëª¨ë¦? {final_memory:.1f} MB")
        print(f"  ì´?ë©”ëª¨ë¦?ì¦ê??? {total_memory_increase:.1f} MB")
        
        # ë©”ëª¨ë¦??¨ìœ¨??ê²€ì¦?
        self.assertLess(total_memory_increase, 200)  # 200MB ?´í•˜ ì¦ê?
        self.assertLess(peak_memory, 1000)  # 1000MB ?´í•˜ ìµœë? ?¬ìš©??
        
        print("??ë©”ëª¨ë¦??¬ìš©??ë²¤ì¹˜ë§ˆí¬ ?„ë£Œ")
    
    def test_cpu_usage_benchmark(self):
        """CPU ?¬ìš©??ë²¤ì¹˜ë§ˆí¬"""
        print("\n=== CPU ?¬ìš©??ë²¤ì¹˜ë§ˆí¬ ===")
        
        # CPU ì§‘ì•½???‘ì—…??
        cpu_intensive_tasks = [
            ("?¤ì¤‘ ??ì²˜ë¦¬", self._cpu_intensive_multi_turn),
            ("ì»¨í…?¤íŠ¸ ?•ì¶•", self._cpu_intensive_compression),
            ("ê°ì • ë¶„ì„", self._cpu_intensive_emotion_analysis),
            ("ë©”ëª¨ë¦?ê²€??, self._cpu_intensive_memory_search)
        ]
        
        cpu_usage_results = []
        
        for task_name, task_func in cpu_intensive_tasks:
            print(f"\n{task_name} ?ŒìŠ¤??..")
            
            # CPU ?¬ìš©??ëª¨ë‹ˆ?°ë§ ?œì‘
            cpu_monitor = threading.Thread(target=self._monitor_cpu_usage)
            cpu_monitor.daemon = True
            cpu_monitor.start()
            
            # ?‘ì—… ?¤í–‰
            start_time = time.time()
            task_func()
            end_time = time.time()
            
            # CPU ?¬ìš©??ì¸¡ì •
            cpu_usage = psutil.cpu_percent(interval=1)
            processing_time = end_time - start_time
            
            cpu_usage_results.append({
                "task": task_name,
                "cpu_usage": cpu_usage,
                "processing_time": processing_time
            })
            
            print(f"  CPU ?¬ìš©ë¥? {cpu_usage:.1f}%")
            print(f"  ì²˜ë¦¬ ?œê°„: {processing_time:.3f}ì´?)
        
        # CPU ?¬ìš©??ë¶„ì„
        avg_cpu_usage = statistics.mean([r["cpu_usage"] for r in cpu_usage_results])
        max_cpu_usage = max([r["cpu_usage"] for r in cpu_usage_results])
        
        print(f"\nCPU ?¬ìš©??ë¶„ì„:")
        print(f"  ?‰ê·  CPU ?¬ìš©ë¥? {avg_cpu_usage:.1f}%")
        print(f"  ìµœë? CPU ?¬ìš©ë¥? {max_cpu_usage:.1f}%")
        
        # CPU ?¨ìœ¨??ê²€ì¦?
        self.assertLess(avg_cpu_usage, 80)  # ?‰ê·  80% ?´í•˜
        self.assertLess(max_cpu_usage, 95)  # ìµœë? 95% ?´í•˜
        
        print("??CPU ?¬ìš©??ë²¤ì¹˜ë§ˆí¬ ?„ë£Œ")
    
    def test_scalability_benchmark(self):
        """?•ì¥??ë²¤ì¹˜ë§ˆí¬"""
        print("\n=== ?•ì¥??ë²¤ì¹˜ë§ˆí¬ ===")
        
        # ?¤ì–‘??ê·œëª¨?ì„œ ?±ëŠ¥ ì¸¡ì •
        scales = [1, 5, 10, 20, 50]
        scalability_results = []
        
        for scale in scales:
            print(f"\nê·œëª¨ {scale} ?ŒìŠ¤??..")
            
            start_time = time.time()
            
            # ê·œëª¨???°ë¥¸ ?‘ì—… ?˜í–‰
            for i in range(scale):
                # ?¸ì…˜ ?ì„±
                context = self.session_manager.add_turn(
                    f"scale_session_{scale}_{i}",
                    f"?•ì¥???ŒìŠ¤??ì§ˆë¬¸ {i}",
                    f"?•ì¥???ŒìŠ¤???‘ë‹µ {i}",
                    "test",
                    f"scale_user_{i % 10}"
                )
                
                # ë©”ëª¨ë¦??€??
                facts = {"scale_data": [f"?•ì¥???ŒìŠ¤???°ì´??{i}"]}
                self.memory_manager.store_important_facts(
                    f"scale_session_{scale}_{i}", f"scale_user_{i % 10}", facts
                )
                
                # ìºì‹œ ?€??
                self.cache_manager.set(f"scale_cache_{scale}_{i}", {
                    "data": f"?•ì¥??ìºì‹œ ?°ì´??{i}",
                    "timestamp": datetime.now().isoformat()
                })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # ?±ëŠ¥ ë©”íŠ¸ë¦?ê³„ì‚°
            throughput = scale / processing_time if processing_time > 0 else 0
            avg_time_per_item = processing_time / scale if scale > 0 else 0
            
            scalability_results.append({
                "scale": scale,
                "processing_time": processing_time,
                "throughput": throughput,
                "avg_time_per_item": avg_time_per_item
            })
            
            print(f"  ì²˜ë¦¬ ?œê°„: {processing_time:.3f}ì´?)
            print(f"  ì²˜ë¦¬?? {throughput:.1f}??ª©/ì´?)
            print(f"  ??ª©???‰ê·  ?œê°„: {avg_time_per_item:.3f}ì´?)
        
        # ?•ì¥??ë¶„ì„
        print(f"\n?•ì¥??ë¶„ì„:")
        for result in scalability_results:
            print(f"  ê·œëª¨ {result['scale']}: {result['throughput']:.1f}??ª©/ì´?)
        
        # ?•ì¥??ê²€ì¦?(ì²˜ë¦¬?‰ì´ ê·œëª¨??ë¹„ë??´ì„œ ì¦ê??˜ëŠ”ì§€)
        throughputs = [r["throughput"] for r in scalability_results]
        if len(throughputs) > 1:
            # ì²˜ë¦¬?‰ì´ ì¦ê??˜ëŠ”ì§€ ?•ì¸
            self.assertGreaterEqual(throughputs[-1], throughputs[0] * 0.5)  # ìµœì†Œ 50% ? ì?
        
        print("???•ì¥??ë²¤ì¹˜ë§ˆí¬ ?„ë£Œ")
    
    def _create_100_sessions(self):
        """100ê°??¸ì…˜ ?ì„±"""
        for i in range(100):
            self.session_manager.add_turn(
                f"memory_test_session_{i}",
                f"ë©”ëª¨ë¦??ŒìŠ¤??ì§ˆë¬¸ {i}",
                f"ë©”ëª¨ë¦??ŒìŠ¤???‘ë‹µ {i}",
                "test",
                f"memory_test_user_{i % 20}"
            )
    
    def _store_1000_memories(self):
        """1000ê°?ë©”ëª¨ë¦??€??""
        for i in range(1000):
            facts = {"memory_test": [f"ë©”ëª¨ë¦??ŒìŠ¤???°ì´??{i}"]}
            self.memory_manager.store_important_facts(
                f"memory_test_session_{i % 100}", f"memory_test_user_{i % 20}", facts
            )
    
    def _cache_500_items(self):
        """500ê°?ìºì‹œ ??ª© ?€??""
        for i in range(500):
            self.cache_manager.set(f"cache_test_{i}", {
                "data": f"ìºì‹œ ?ŒìŠ¤???°ì´??{i}",
                "timestamp": datetime.now().isoformat()
            })
    
    def _optimize_memory(self):
        """ë©”ëª¨ë¦?ìµœì ??""
        self.memory_optimizer.optimize_memory()
    
    def _cpu_intensive_multi_turn(self):
        """CPU ì§‘ì•½???¤ì¤‘ ??ì²˜ë¦¬"""
        for i in range(50):
            context = self.session_manager.add_turn(
                f"cpu_test_session_{i}",
                f"CPU ?ŒìŠ¤??ì§ˆë¬¸ {i}",
                f"CPU ?ŒìŠ¤???‘ë‹µ {i}",
                "test",
                f"cpu_test_user_{i % 10}"
            )
            self.multi_turn_handler.build_complete_query(f"?„ì† ì§ˆë¬¸ {i}", context)
    
    def _cpu_intensive_compression(self):
        """CPU ì§‘ì•½??ì»¨í…?¤íŠ¸ ?•ì¶•"""
        # ê¸??€??ì»¨í…?¤íŠ¸ ?ì„±
        long_context = self.session_manager.add_turn(
            "cpu_compression_session",
            "ê¸?ì§ˆë¬¸ " + "x" * 1000,
            "ê¸??‘ë‹µ " + "y" * 1000,
            "test",
            "cpu_compression_user"
        )
        
        for i in range(20):
            self.context_compressor.compress_long_conversation(long_context)
    
    def _cpu_intensive_emotion_analysis(self):
        """CPU ì§‘ì•½??ê°ì • ë¶„ì„"""
        test_queries = [f"ê°ì • ë¶„ì„ ?ŒìŠ¤??ì§ˆë¬¸ {i} " + "ê°ì •" * 10 for i in range(100)]
        
        for query in test_queries:
            self.emotion_analyzer.analyze_emotion(query)
    
    def _cpu_intensive_memory_search(self):
        """CPU ì§‘ì•½??ë©”ëª¨ë¦?ê²€??""
        for i in range(100):
            self.memory_manager.retrieve_relevant_memory(
                f"cpu_search_session_{i % 10}", f"ê²€???ŒìŠ¤??ì§ˆë¬¸ {i}", f"cpu_search_user_{i % 5}"
            )
    
    def _monitor_cpu_usage(self):
        """CPU ?¬ìš©??ëª¨ë‹ˆ?°ë§"""
        while True:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 90:
                print(f"  ?’ì? CPU ?¬ìš©ë¥?ê°ì?: {cpu_usage:.1f}%")
            time.sleep(0.5)


def run_performance_benchmark():
    """?±ëŠ¥ ë²¤ì¹˜ë§ˆí¬ ?¤í–‰"""
    print("=" * 60)
    print("LawFirmAI ?±ëŠ¥ ë²¤ì¹˜ë§ˆí¬ ?œì‘")
    print("=" * 60)
    
    # ?ŒìŠ¤???¤ìœ„???ì„±
    test_suite = unittest.TestSuite()
    
    # ?ŒìŠ¤??ì¼€?´ìŠ¤ ì¶”ê?
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmark))
    
    # ?ŒìŠ¤???¤í–‰
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 60)
    print("?±ëŠ¥ ë²¤ì¹˜ë§ˆí¬ ê²°ê³¼ ?”ì•½")
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
        print("?‰ ?°ìˆ˜???±ëŠ¥! ?œìŠ¤?œì´ ?¨ìœ¨?ìœ¼ë¡??‘ë™?©ë‹ˆ??")
    elif success_rate >= 75:
        print("???‘í˜¸???±ëŠ¥! ?¼ë? ìµœì ?”ê? ?„ìš”?©ë‹ˆ??")
    else:
        print("? ï¸ ?±ëŠ¥ ê°œì„ ???„ìš”?©ë‹ˆ?? ?œìŠ¤?œì„ ìµœì ?”í•´ì£¼ì„¸??")
    
    return result


if __name__ == "__main__":
    run_performance_benchmark()
