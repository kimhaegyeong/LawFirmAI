# -*- coding: utf-8 -*-
"""
LawFirmAI 스트레스 테스트
대량 데이터 처리 및 동시성 테스트
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

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.chat_service import ChatService
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor
from source.utils.performance_optimizer import PerformanceMonitor, MemoryOptimizer, CacheManager
from source.data.conversation_store import ConversationStore


class TestStressSystem(unittest.TestCase):
    """스트레스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stress.db")
        
        # 컴포넌트 초기화
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.memory_manager = ContextualMemoryManager(self.session_manager.conversation_store)
        self.quality_monitor = ConversationQualityMonitor(self.session_manager.conversation_store)
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager(max_size=1000, ttl=3600)
        
        # 테스트 데이터
        self.test_users = [f"stress_user_{i}" for i in range(50)]
        self.test_sessions = [f"stress_session_{i}" for i in range(100)]
        
    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_scale_conversations(self):
        """대규모 대화 처리 테스트"""
        print("\n=== 대규모 대화 처리 테스트 ===")
        
        # 대량 대화 데이터 생성
        conversation_data = []
        for i in range(500):  # 500개의 대화 생성
            conversation_data.append({
                "session_id": f"large_scale_session_{i}",
                "user_id": f"large_scale_user_{i % 50}",
                "query": f"법률 질문 {i} - 손해배상, 계약, 형법 관련 문의",
                "response": f"법률 답변 {i} - 상세한 법률 설명과 판례 인용",
                "question_type": random.choice(["legal_advice", "law_inquiry", "procedure_guide"])
            })
        
        start_time = time.time()
        
        # 대량 대화 처리
        processed_count = 0
        for data in conversation_data:
            try:
                # 턴 추가
                context = self.session_manager.add_turn(
                    data["session_id"],
                    data["query"],
                    data["response"],
                    data["question_type"],
                    data["user_id"]
                )
                
                # 메모리 저장
                facts = {
                    "legal_knowledge": [data["response"]],
                    "user_context": [data["query"]]
                }
                self.memory_manager.store_important_facts(
                    data["session_id"], data["user_id"], facts
                )
                
                processed_count += 1
                
                # 진행률 표시
                if processed_count % 100 == 0:
                    print(f"  처리된 대화: {processed_count}/500")
                    
            except Exception as e:
                print(f"  오류 발생 (대화 {processed_count}): {e}")
        
        processing_time = time.time() - start_time
        
        print(f"\n대규모 대화 처리 결과:")
        print(f"  처리된 대화 수: {processed_count}")
        print(f"  총 처리 시간: {processing_time:.2f}초")
        print(f"  평균 처리 시간: {processing_time/processed_count:.3f}초/대화")
        print(f"  처리 속도: {processed_count/processing_time:.1f}대화/초")
        
        # 결과 검증
        self.assertGreaterEqual(processed_count, 450)  # 최소 90% 성공
        self.assertLess(processing_time, 60)  # 60초 이내 완료
        
        print("✅ 대규모 대화 처리 테스트 완료")
    
    def test_concurrent_sessions(self):
        """동시 세션 처리 테스트"""
        print("\n=== 동시 세션 처리 테스트 ===")
        
        def process_session(session_data):
            """세션 처리 함수"""
            try:
                # 턴 추가
                context = self.session_manager.add_turn(
                    session_data["session_id"],
                    session_data["query"],
                    session_data["response"],
                    session_data["question_type"],
                    session_data["user_id"]
                )
                
                # 메모리 저장
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
        
        # 동시 처리할 세션 데이터 생성
        concurrent_sessions = []
        for i in range(100):
            concurrent_sessions.append({
                "session_id": f"concurrent_session_{i}",
                "user_id": f"concurrent_user_{i % 20}",
                "query": f"동시 처리 테스트 질문 {i}",
                "response": f"동시 처리 테스트 응답 {i}",
                "question_type": "test"
            })
        
        start_time = time.time()
        
        # ThreadPoolExecutor를 사용한 동시 처리
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_session, session) for session in concurrent_sessions]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        processing_time = time.time() - start_time
        
        # 결과 분석
        successful_sessions = [r for r in results if r["success"]]
        failed_sessions = [r for r in results if not r["success"]]
        
        print(f"\n동시 세션 처리 결과:")
        print(f"  총 세션 수: {len(concurrent_sessions)}")
        print(f"  성공한 세션: {len(successful_sessions)}")
        print(f"  실패한 세션: {len(failed_sessions)}")
        print(f"  성공률: {len(successful_sessions)/len(concurrent_sessions)*100:.1f}%")
        print(f"  총 처리 시간: {processing_time:.2f}초")
        print(f"  평균 처리 시간: {processing_time/len(concurrent_sessions):.3f}초/세션")
        
        # 결과 검증
        self.assertGreaterEqual(len(successful_sessions), 90)  # 최소 90% 성공
        self.assertLess(processing_time, 30)  # 30초 이내 완료
        
        if failed_sessions:
            print(f"\n실패한 세션들:")
            for failed in failed_sessions[:5]:  # 처음 5개만 표시
                print(f"  - {failed['session_id']}: {failed['error']}")
        
        print("✅ 동시 세션 처리 테스트 완료")
    
    def test_memory_stress(self):
        """메모리 스트레스 테스트"""
        print("\n=== 메모리 스트레스 테스트 ===")
        
        # 초기 메모리 사용량
        initial_memory_obj = self.memory_optimizer.get_memory_usage()
        initial_memory = initial_memory_obj.process_memory / 1024 / 1024  # MB로 변환
        print(f"초기 메모리 사용량: {initial_memory:.1f} MB")
        
        # 대량 메모리 할당
        large_data_sets = []
        for i in range(100):
            # 각각 1MB 정도의 데이터 생성
            large_data = {
                "id": i,
                "content": "x" * 1000000,  # 1MB 문자열
                "metadata": {"created_at": datetime.now().isoformat()},
                "entities": [f"entity_{j}" for j in range(100)]
            }
            large_data_sets.append(large_data)
        
        # 메모리 사용량 모니터링
        peak_memory = initial_memory
        for i, data in enumerate(large_data_sets):
            # 데이터 처리 시뮬레이션
            facts = {"large_data": [data["content"]]}
            self.memory_manager.store_important_facts(
                f"memory_stress_session_{i}", f"memory_stress_user_{i}", facts
            )
            
            # 메모리 사용량 체크
            current_memory_obj = self.memory_optimizer.get_memory_usage()
            current_memory = current_memory_obj.process_memory / 1024 / 1024  # MB로 변환
            peak_memory = max(peak_memory, current_memory)
            
            if i % 20 == 0:
                print(f"  처리된 데이터: {i+1}/100, 현재 메모리: {current_memory:.1f} MB")
        
        # 메모리 최적화
        print("\n메모리 최적화 실행...")
        freed_memory_result = self.memory_optimizer.optimize_memory()
        freed_memory = freed_memory_result["memory_freed_mb"]
        
        final_memory_obj = self.memory_optimizer.get_memory_usage()
        final_memory = final_memory_obj.process_memory / 1024 / 1024  # MB로 변환
        
        print(f"\n메모리 스트레스 테스트 결과:")
        print(f"  초기 메모리: {initial_memory:.1f} MB")
        print(f"  최대 메모리: {peak_memory:.1f} MB")
        print(f"  최종 메모리: {final_memory:.1f} MB")
        print(f"  해제된 메모리: {freed_memory:.1f} MB")
        print(f"  메모리 증가량: {peak_memory - initial_memory:.1f} MB")
        
        # 결과 검증
        self.assertLess(peak_memory - initial_memory, 500)  # 500MB 이하 증가
        self.assertGreater(freed_memory, 0)  # 메모리 해제 확인
        
        print("✅ 메모리 스트레스 테스트 완료")
    
    def test_cache_stress(self):
        """캐시 스트레스 테스트"""
        print("\n=== 캐시 스트레스 테스트 ===")
        
        # 대량 캐시 데이터 생성 및 저장
        cache_data_count = 1000
        start_time = time.time()
        
        print(f"캐시 데이터 {cache_data_count}개 생성 중...")
        
        for i in range(cache_data_count):
            cache_key = f"stress_cache_key_{i}"
            cache_value = {
                "id": i,
                "data": f"스트레스 테스트 데이터 {i}",
                "timestamp": datetime.now().isoformat(),
                "large_content": "x" * 1000  # 1KB 데이터
            }
            
            self.cache_manager.set(cache_key, cache_value)
            
            if i % 200 == 0:
                print(f"  저장된 캐시: {i+1}/{cache_data_count}")
        
        storage_time = time.time() - start_time
        
        # 캐시 조회 테스트
        print(f"\n캐시 조회 테스트 시작...")
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
                print(f"  조회된 캐시: {i+1}/{cache_data_count}")
        
        retrieval_time = time.time() - start_time
        
        # 캐시 통계
        cache_stats = self.cache_manager.get_stats()
        
        print(f"\n캐시 스트레스 테스트 결과:")
        print(f"  저장된 캐시 수: {cache_data_count}")
        print(f"  저장 시간: {storage_time:.2f}초")
        print(f"  조회 시간: {retrieval_time:.2f}초")
        print(f"  히트 수: {hit_count}")
        print(f"  미스 수: {miss_count}")
        print(f"  히트율: {hit_count/(hit_count+miss_count)*100:.1f}%")
        print(f"  캐시 크기: {cache_stats['cache_size']}")
        print(f"  평균 저장 시간: {storage_time/cache_data_count*1000:.2f}ms/항목")
        print(f"  평균 조회 시간: {retrieval_time/cache_data_count*1000:.2f}ms/항목")
        
        # 결과 검증
        self.assertGreater(hit_count, cache_data_count * 0.8)  # 80% 이상 히트율
        self.assertLess(storage_time, 10)  # 10초 이내 저장
        self.assertLess(retrieval_time, 5)  # 5초 이내 조회
        
        print("✅ 캐시 스트레스 테스트 완료")
    
    def test_database_stress(self):
        """데이터베이스 스트레스 테스트"""
        print("\n=== 데이터베이스 스트레스 테스트 ===")
        
        # 대량 데이터베이스 작업
        operations_count = 1000
        start_time = time.time()
        
        print(f"데이터베이스 작업 {operations_count}개 실행 중...")
        
        successful_operations = 0
        
        for i in range(operations_count):
            try:
                # 세션 생성 및 저장
                session_data = {
                    "session_id": f"db_stress_session_{i}",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "topic_stack": [f"topic_{i}"],
                    "metadata": {"user_id": f"db_stress_user_{i % 100}"},
                    "turns": [
                        {
                            "user_query": f"데이터베이스 스트레스 테스트 질문 {i}",
                            "bot_response": f"데이터베이스 스트레스 테스트 응답 {i}",
                            "timestamp": datetime.now().isoformat(),
                            "question_type": "test",
                            "entities": {"test_entities": [f"entity_{i}"]}
                        }
                    ],
                    "entities": {"test_entities": [f"entity_{i}"]}
                }
                
                # 세션 저장
                success = self.session_manager.conversation_store.save_session(session_data)
                if success:
                    successful_operations += 1
                
                # 진행률 표시
                if i % 200 == 0:
                    print(f"  완료된 작업: {i+1}/{operations_count}")
                    
            except Exception as e:
                print(f"  오류 발생 (작업 {i}): {e}")
        
        processing_time = time.time() - start_time
        
        # 데이터베이스 통계 조회
        try:
            stats = self.session_manager.conversation_store.get_statistics()
            print(f"\n데이터베이스 통계:")
            print(f"  총 세션 수: {stats.get('total_sessions', 0)}")
            print(f"  총 턴 수: {stats.get('total_turns', 0)}")
            print(f"  총 엔티티 수: {stats.get('total_entities', 0)}")
        except Exception as e:
            print(f"  통계 조회 오류: {e}")
        
        print(f"\n데이터베이스 스트레스 테스트 결과:")
        print(f"  총 작업 수: {operations_count}")
        print(f"  성공한 작업: {successful_operations}")
        print(f"  실패한 작업: {operations_count - successful_operations}")
        print(f"  성공률: {successful_operations/operations_count*100:.1f}%")
        print(f"  총 처리 시간: {processing_time:.2f}초")
        print(f"  평균 처리 시간: {processing_time/operations_count*1000:.2f}ms/작업")
        print(f"  처리 속도: {operations_count/processing_time:.1f}작업/초")
        
        # 결과 검증
        self.assertGreaterEqual(successful_operations, operations_count * 0.9)  # 90% 이상 성공
        self.assertLess(processing_time, 60)  # 60초 이내 완료
        
        print("✅ 데이터베이스 스트레스 테스트 완료")
    
    def test_error_recovery(self):
        """오류 복구 테스트"""
        print("\n=== 오류 복구 테스트 ===")
        
        # 다양한 오류 상황 시뮬레이션
        error_scenarios = [
            {
                "name": "잘못된 세션 ID",
                "test": lambda: self.session_manager.add_turn("", "test", "test", "test", "test")
            },
            {
                "name": "잘못된 사용자 ID",
                "test": lambda: self.session_manager.add_turn("test", "test", "test", "test", "")
            },
            {
                "name": "잘못된 메모리 데이터",
                "test": lambda: self.memory_manager.store_important_facts("test", "test", None)
            },
            {
                "name": "잘못된 캐시 키",
                "test": lambda: self.cache_manager.get(None)
            }
        ]
        
        recovery_success_count = 0
        
        for scenario in error_scenarios:
            print(f"\n오류 시나리오: {scenario['name']}")
            
            try:
                # 오류 발생 시도
                result = scenario['test']()
                print(f"  예상치 못한 성공: {result}")
            except Exception as e:
                print(f"  예상된 오류: {type(e).__name__}")
                
                # 시스템이 여전히 작동하는지 확인
                try:
                    # 정상적인 작업 수행
                    context = self.session_manager.add_turn(
                        "recovery_test_session",
                        "복구 테스트 질문",
                        "복구 테스트 응답",
                        "test",
                        "recovery_test_user"
                    )
                    
                    if context and len(context.turns) > 0:
                        print(f"  ✅ 시스템 복구 성공")
                        recovery_success_count += 1
                    else:
                        print(f"  ❌ 시스템 복구 실패")
                        
                except Exception as recovery_error:
                    print(f"  ❌ 복구 중 오류: {recovery_error}")
        
        print(f"\n오류 복구 테스트 결과:")
        print(f"  테스트된 시나리오: {len(error_scenarios)}")
        print(f"  성공한 복구: {recovery_success_count}")
        print(f"  복구 성공률: {recovery_success_count/len(error_scenarios)*100:.1f}%")
        
        # 결과 검증
        self.assertGreaterEqual(recovery_success_count, len(error_scenarios) * 0.8)  # 80% 이상 복구 성공
        
        print("✅ 오류 복구 테스트 완료")


def run_stress_tests():
    """스트레스 테스트 실행"""
    print("=" * 60)
    print("LawFirmAI 스트레스 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStressSystem))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("스트레스 테스트 결과 요약")
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
        print("🎉 우수한 스트레스 저항성! 시스템이 안정적으로 작동합니다.")
    elif success_rate >= 75:
        print("✅ 양호한 스트레스 저항성! 일부 개선이 필요합니다.")
    else:
        print("⚠️ 스트레스 저항성 개선이 필요합니다. 시스템을 점검해주세요.")
    
    return result


if __name__ == "__main__":
    run_stress_tests()
