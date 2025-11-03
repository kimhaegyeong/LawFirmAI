# -*- coding: utf-8 -*-
"""
LawFirmAI 성능 벤치마크 테스트
응답 시간, 처리량, 리소스 사용량 측정
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

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.services.integrated_session_manager import IntegratedSessionManager
from lawfirm_langgraph.langgraph_core.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor
from source.services.user_profile_manager import UserProfileManager
from lawfirm_langgraph.langgraph_core.services.emotion_intent_analyzer import EmotionIntentAnalyzer
from source.services.conversation_flow_tracker import ConversationFlowTracker
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor
from source.utils.performance_optimizer import PerformanceMonitor, MemoryOptimizer, CacheManager


class TestPerformanceBenchmark(unittest.TestCase):
    """성능 벤치마크 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_benchmark.db")
        
        # 컴포넌트 초기화
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.multi_turn_handler = MultiTurnQuestionHandler()
        self.context_compressor = ContextCompressor(max_tokens=1000)
        self.user_profile_manager = UserProfileManager(self.session_manager.conversation_store)
        self.emotion_analyzer = EmotionIntentAnalyzer()
        self.flow_tracker = ConversationFlowTracker()
        self.memory_manager = ContextualMemoryManager(self.session_manager.conversation_store)
        self.quality_monitor = ConversationQualityMonitor(self.session_manager.conversation_store)
        
        # 성능 모니터링 컴포넌트
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager(max_size=1000, ttl=3600)
        
        # 벤치마크 데이터
        self.benchmark_queries = [
            "손해배상에 대해 자세히 설명해주세요",
            "계약 해지 절차는 어떻게 되나요?",
            "민법 제750조의 요건을 알려주세요",
            "불법행위의 성립요건은 무엇인가요?",
            "과실비율은 어떻게 산정되나요?"
        ]
        
    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_response_time_benchmark(self):
        """응답 시간 벤치마크"""
        print("\n=== 응답 시간 벤치마크 ===")
        
        response_times = []
        
        for i, query in enumerate(self.benchmark_queries):
            # 각 쿼리별 응답 시간 측정
            start_time = time.time()
            
            # Phase 1: 대화 맥락 처리
            context = self.session_manager.add_turn(
                f"benchmark_session_{i}",
                query,
                f"벤치마크 응답 {i}",
                "legal_advice",
                f"benchmark_user_{i}"
            )
            
            # Phase 2: 감정 분석
            emotion_result = self.emotion_analyzer.analyze_emotion(query)
            
            # Phase 3: 메모리 저장
            facts = {"legal_knowledge": [f"벤치마크 사실 {i}"]}
            self.memory_manager.store_important_facts(
                f"benchmark_session_{i}", f"benchmark_user_{i}", facts
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"쿼리 {i+1}: {response_time:.3f}초")
        
        # 통계 계산
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        print(f"\n응답 시간 통계:")
        print(f"  평균: {avg_response_time:.3f}초")
        print(f"  중간값: {median_response_time:.3f}초")
        print(f"  최소: {min_response_time:.3f}초")
        print(f"  최대: {max_response_time:.3f}초")
        print(f"  표준편차: {std_response_time:.3f}초")
        
        # 성능 기준 검증
        self.assertLess(avg_response_time, 2.0)  # 평균 2초 이내
        self.assertLess(max_response_time, 5.0)  # 최대 5초 이내
        
        print("✅ 응답 시간 벤치마크 완료")
    
    def test_throughput_benchmark(self):
        """처리량 벤치마크"""
        print("\n=== 처리량 벤치마크 ===")
        
        # 다양한 부하 수준에서 처리량 측정
        load_levels = [10, 50, 100, 200]
        throughput_results = []
        
        for load in load_levels:
            print(f"\n부하 수준: {load}개 요청")
            
            start_time = time.time()
            successful_requests = 0
            
            for i in range(load):
                try:
                    # 요청 처리
                    context = self.session_manager.add_turn(
                        f"throughput_session_{load}_{i}",
                        f"처리량 테스트 질문 {i}",
                        f"처리량 테스트 응답 {i}",
                        "test",
                        f"throughput_user_{i % 10}"
                    )
                    
                    # 메모리 저장
                    facts = {"test_data": [f"처리량 테스트 데이터 {i}"]}
                    self.memory_manager.store_important_facts(
                        f"throughput_session_{load}_{i}", f"throughput_user_{i % 10}", facts
                    )
                    
                    successful_requests += 1
                    
                except Exception as e:
                    print(f"  요청 {i} 실패: {e}")
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = successful_requests / total_time if total_time > 0 else 0
            
            throughput_results.append({
                "load": load,
                "successful_requests": successful_requests,
                "total_time": total_time,
                "throughput": throughput
            })
            
            print(f"  성공한 요청: {successful_requests}/{load}")
            print(f"  총 처리 시간: {total_time:.2f}초")
            print(f"  처리량: {throughput:.1f}요청/초")
        
        # 처리량 분석
        print(f"\n처리량 분석:")
        for result in throughput_results:
            print(f"  부하 {result['load']}: {result['throughput']:.1f}요청/초")
        
        # 성능 기준 검증
        max_throughput = max(r["throughput"] for r in throughput_results)
        self.assertGreater(max_throughput, 10)  # 최소 10요청/초
        
        print("✅ 처리량 벤치마크 완료")
    
    def test_memory_usage_benchmark(self):
        """메모리 사용량 벤치마크"""
        print("\n=== 메모리 사용량 벤치마크 ===")
        
        # 초기 메모리 사용량
        initial_memory_obj = self.memory_optimizer.get_memory_usage()
        initial_memory = initial_memory_obj.process_memory / 1024 / 1024  # MB로 변환
        print(f"초기 메모리 사용량: {initial_memory:.1f} MB")
        
        memory_usage_history = [initial_memory]
        
        # 단계별 메모리 사용량 측정
        stages = [
            ("세션 관리자 초기화", lambda: None),
            ("100개 세션 생성", self._create_100_sessions),
            ("1000개 메모리 저장", self._store_1000_memories),
            ("캐시 500개 항목", self._cache_500_items),
            ("메모리 최적화", self._optimize_memory)
        ]
        
        for stage_name, stage_func in stages:
            print(f"\n{stage_name}...")
            
            # 메모리 사용량 측정
            memory_before_obj = self.memory_optimizer.get_memory_usage()
            memory_before = memory_before_obj.process_memory / 1024 / 1024  # MB로 변환
            
            # 작업 실행
            stage_func()
            
            memory_after_obj = self.memory_optimizer.get_memory_usage()
            memory_after = memory_after_obj.process_memory / 1024 / 1024  # MB로 변환
            memory_increase = memory_after - memory_before
            
            memory_usage_history.append(memory_after)
            
            print(f"  메모리 사용량: {memory_after:.1f} MB")
            print(f"  메모리 증가량: {memory_increase:.1f} MB")
        
        # 메모리 사용량 분석
        peak_memory = max(memory_usage_history)
        final_memory = memory_usage_history[-1]
        total_memory_increase = final_memory - initial_memory
        
        print(f"\n메모리 사용량 분석:")
        print(f"  초기 메모리: {initial_memory:.1f} MB")
        print(f"  최대 메모리: {peak_memory:.1f} MB")
        print(f"  최종 메모리: {final_memory:.1f} MB")
        print(f"  총 메모리 증가량: {total_memory_increase:.1f} MB")
        
        # 메모리 효율성 검증
        self.assertLess(total_memory_increase, 200)  # 200MB 이하 증가
        self.assertLess(peak_memory, 1000)  # 1000MB 이하 최대 사용량
        
        print("✅ 메모리 사용량 벤치마크 완료")
    
    def test_cpu_usage_benchmark(self):
        """CPU 사용량 벤치마크"""
        print("\n=== CPU 사용량 벤치마크 ===")
        
        # CPU 집약적 작업들
        cpu_intensive_tasks = [
            ("다중 턴 처리", self._cpu_intensive_multi_turn),
            ("컨텍스트 압축", self._cpu_intensive_compression),
            ("감정 분석", self._cpu_intensive_emotion_analysis),
            ("메모리 검색", self._cpu_intensive_memory_search)
        ]
        
        cpu_usage_results = []
        
        for task_name, task_func in cpu_intensive_tasks:
            print(f"\n{task_name} 테스트...")
            
            # CPU 사용량 모니터링 시작
            cpu_monitor = threading.Thread(target=self._monitor_cpu_usage)
            cpu_monitor.daemon = True
            cpu_monitor.start()
            
            # 작업 실행
            start_time = time.time()
            task_func()
            end_time = time.time()
            
            # CPU 사용량 측정
            cpu_usage = psutil.cpu_percent(interval=1)
            processing_time = end_time - start_time
            
            cpu_usage_results.append({
                "task": task_name,
                "cpu_usage": cpu_usage,
                "processing_time": processing_time
            })
            
            print(f"  CPU 사용률: {cpu_usage:.1f}%")
            print(f"  처리 시간: {processing_time:.3f}초")
        
        # CPU 사용량 분석
        avg_cpu_usage = statistics.mean([r["cpu_usage"] for r in cpu_usage_results])
        max_cpu_usage = max([r["cpu_usage"] for r in cpu_usage_results])
        
        print(f"\nCPU 사용량 분석:")
        print(f"  평균 CPU 사용률: {avg_cpu_usage:.1f}%")
        print(f"  최대 CPU 사용률: {max_cpu_usage:.1f}%")
        
        # CPU 효율성 검증
        self.assertLess(avg_cpu_usage, 80)  # 평균 80% 이하
        self.assertLess(max_cpu_usage, 95)  # 최대 95% 이하
        
        print("✅ CPU 사용량 벤치마크 완료")
    
    def test_scalability_benchmark(self):
        """확장성 벤치마크"""
        print("\n=== 확장성 벤치마크 ===")
        
        # 다양한 규모에서 성능 측정
        scales = [1, 5, 10, 20, 50]
        scalability_results = []
        
        for scale in scales:
            print(f"\n규모 {scale} 테스트...")
            
            start_time = time.time()
            
            # 규모에 따른 작업 수행
            for i in range(scale):
                # 세션 생성
                context = self.session_manager.add_turn(
                    f"scale_session_{scale}_{i}",
                    f"확장성 테스트 질문 {i}",
                    f"확장성 테스트 응답 {i}",
                    "test",
                    f"scale_user_{i % 10}"
                )
                
                # 메모리 저장
                facts = {"scale_data": [f"확장성 테스트 데이터 {i}"]}
                self.memory_manager.store_important_facts(
                    f"scale_session_{scale}_{i}", f"scale_user_{i % 10}", facts
                )
                
                # 캐시 저장
                self.cache_manager.set(f"scale_cache_{scale}_{i}", {
                    "data": f"확장성 캐시 데이터 {i}",
                    "timestamp": datetime.now().isoformat()
                })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 성능 메트릭 계산
            throughput = scale / processing_time if processing_time > 0 else 0
            avg_time_per_item = processing_time / scale if scale > 0 else 0
            
            scalability_results.append({
                "scale": scale,
                "processing_time": processing_time,
                "throughput": throughput,
                "avg_time_per_item": avg_time_per_item
            })
            
            print(f"  처리 시간: {processing_time:.3f}초")
            print(f"  처리량: {throughput:.1f}항목/초")
            print(f"  항목당 평균 시간: {avg_time_per_item:.3f}초")
        
        # 확장성 분석
        print(f"\n확장성 분석:")
        for result in scalability_results:
            print(f"  규모 {result['scale']}: {result['throughput']:.1f}항목/초")
        
        # 확장성 검증 (처리량이 규모에 비례해서 증가하는지)
        throughputs = [r["throughput"] for r in scalability_results]
        if len(throughputs) > 1:
            # 처리량이 증가하는지 확인
            self.assertGreaterEqual(throughputs[-1], throughputs[0] * 0.5)  # 최소 50% 유지
        
        print("✅ 확장성 벤치마크 완료")
    
    def _create_100_sessions(self):
        """100개 세션 생성"""
        for i in range(100):
            self.session_manager.add_turn(
                f"memory_test_session_{i}",
                f"메모리 테스트 질문 {i}",
                f"메모리 테스트 응답 {i}",
                "test",
                f"memory_test_user_{i % 20}"
            )
    
    def _store_1000_memories(self):
        """1000개 메모리 저장"""
        for i in range(1000):
            facts = {"memory_test": [f"메모리 테스트 데이터 {i}"]}
            self.memory_manager.store_important_facts(
                f"memory_test_session_{i % 100}", f"memory_test_user_{i % 20}", facts
            )
    
    def _cache_500_items(self):
        """500개 캐시 항목 저장"""
        for i in range(500):
            self.cache_manager.set(f"cache_test_{i}", {
                "data": f"캐시 테스트 데이터 {i}",
                "timestamp": datetime.now().isoformat()
            })
    
    def _optimize_memory(self):
        """메모리 최적화"""
        self.memory_optimizer.optimize_memory()
    
    def _cpu_intensive_multi_turn(self):
        """CPU 집약적 다중 턴 처리"""
        for i in range(50):
            context = self.session_manager.add_turn(
                f"cpu_test_session_{i}",
                f"CPU 테스트 질문 {i}",
                f"CPU 테스트 응답 {i}",
                "test",
                f"cpu_test_user_{i % 10}"
            )
            self.multi_turn_handler.build_complete_query(f"후속 질문 {i}", context)
    
    def _cpu_intensive_compression(self):
        """CPU 집약적 컨텍스트 압축"""
        # 긴 대화 컨텍스트 생성
        long_context = self.session_manager.add_turn(
            "cpu_compression_session",
            "긴 질문 " + "x" * 1000,
            "긴 응답 " + "y" * 1000,
            "test",
            "cpu_compression_user"
        )
        
        for i in range(20):
            self.context_compressor.compress_long_conversation(long_context)
    
    def _cpu_intensive_emotion_analysis(self):
        """CPU 집약적 감정 분석"""
        test_queries = [f"감정 분석 테스트 질문 {i} " + "감정" * 10 for i in range(100)]
        
        for query in test_queries:
            self.emotion_analyzer.analyze_emotion(query)
    
    def _cpu_intensive_memory_search(self):
        """CPU 집약적 메모리 검색"""
        for i in range(100):
            self.memory_manager.retrieve_relevant_memory(
                f"cpu_search_session_{i % 10}", f"검색 테스트 질문 {i}", f"cpu_search_user_{i % 5}"
            )
    
    def _monitor_cpu_usage(self):
        """CPU 사용량 모니터링"""
        while True:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 90:
                print(f"  높은 CPU 사용률 감지: {cpu_usage:.1f}%")
            time.sleep(0.5)


def run_performance_benchmark():
    """성능 벤치마크 실행"""
    print("=" * 60)
    print("LawFirmAI 성능 벤치마크 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmark))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("성능 벤치마크 결과 요약")
    print("=" * 60)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n전체 통과율: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 우수한 성능! 시스템이 효율적으로 작동합니다.")
    elif success_rate >= 75:
        print("✅ 양호한 성능! 일부 최적화가 필요합니다.")
    else:
        print("⚠️ 성능 개선이 필요합니다. 시스템을 최적화해주세요.")
    
    return result


if __name__ == "__main__":
    run_performance_benchmark()
