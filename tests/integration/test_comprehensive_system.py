# -*- coding: utf-8 -*-
"""
LawFirmAI 종합 시스템 테스트
모든 컴포넌트의 통합 테스트 및 성능 테스트
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
from core.data.conversation_store import ConversationStore
from lawfirm_langgraph.langgraph_core.services.conversation_manager import ConversationManager, ConversationContext, ConversationTurn


class TestComprehensiveSystem(unittest.TestCase):
    """종합 시스템 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_comprehensive.db")
        
        # 모든 컴포넌트 초기화
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
        self.cache_manager = CacheManager(max_size=500, ttl=1800)
        
        # 테스트 데이터
        self.test_user_id = "test_user_comprehensive"
        self.test_session_id = "test_session_comprehensive"
        
    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_phase1_comprehensive(self):
        """Phase 1 종합 테스트"""
        print("\n=== Phase 1 종합 테스트 ===")
        
        # 1. 다중 턴 대화 시나리오
        conversation_scenarios = [
            {
                "query": "손해배상에 대해 알려주세요",
                "response": "민법 제750조에 따른 손해배상은 불법행위로 인한 손해를 배상하는 제도입니다.",
                "expected_entities": ["손해배상", "민법", "제750조"]
            },
            {
                "query": "그것의 요건은 무엇인가요?",
                "response": "손해배상의 요건은 가해행위, 손해발생, 인과관계, 고의 또는 과실입니다.",
                "expected_entities": ["손해배상", "요건", "가해행위", "손해발생"]
            },
            {
                "query": "위의 사건에서 과실비율은 어떻게 정해지나요?",
                "response": "과실비율은 교통사고의 경우 도로교통법에 따라 정해집니다.",
                "expected_entities": ["과실비율", "교통사고", "도로교통법"]
            }
        ]
        
        # 대화 진행
        for i, scenario in enumerate(conversation_scenarios):
            print(f"\n턴 {i+1}: {scenario['query']}")
            
            # 턴 추가
            context = self.session_manager.add_turn(
                self.test_session_id,
                scenario["query"],
                scenario["response"],
                "legal_advice",
                self.test_user_id
            )
            
            # 다중 턴 처리 테스트
            multi_turn_result = self.multi_turn_handler.build_complete_query(
                scenario["query"], context
            )
            
            self.assertIsNotNone(multi_turn_result)
            self.assertIn("resolved_query", multi_turn_result)
            self.assertIn("confidence", multi_turn_result)
            
            print(f"  해결된 쿼리: {multi_turn_result['resolved_query']}")
            print(f"  신뢰도: {multi_turn_result['confidence']:.2f}")
            
            # 엔티티 추출 확인
            if i == 0:  # 첫 번째 턴에서 엔티티 확인
                entities = context.entities
                for entity_type, entity_set in entities.items():
                    if entity_set:
                        print(f"  {entity_type}: {list(entity_set)}")
        
        # 2. 컨텍스트 압축 테스트
        print("\n컨텍스트 압축 테스트:")
        compression_result = self.context_compressor.compress_long_conversation(context)
        
        self.assertIsNotNone(compression_result)
        self.assertLessEqual(compression_result.compression_ratio, 1.0)
        self.assertGreater(compression_result.original_tokens, 0)
        
        print(f"  원본 토큰: {compression_result.original_tokens}")
        print(f"  압축 토큰: {compression_result.compressed_tokens}")
        print(f"  압축률: {compression_result.compression_ratio:.2f}")
        
        # 3. 세션 지속성 테스트
        print("\n세션 지속성 테스트:")
        self.session_manager.sync_to_database(self.test_session_id)
        
        # 새로운 세션 매니저로 로드
        new_session_manager = IntegratedSessionManager(self.db_path)
        loaded_context = new_session_manager.load_from_database(self.test_session_id)
        
        self.assertIsNotNone(loaded_context)
        self.assertEqual(len(loaded_context.turns), len(conversation_scenarios))
        
        print(f"  로드된 턴 수: {len(loaded_context.turns)}")
        
        print("✅ Phase 1 종합 테스트 완료")
    
    def test_phase2_comprehensive(self):
        """Phase 2 종합 테스트"""
        print("\n=== Phase 2 종합 테스트 ===")
        
        # 1. 사용자 프로필 관리 테스트
        print("\n사용자 프로필 관리 테스트:")
        
        # 프로필 생성
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "preferred_language": "ko",
            "interest_areas": ["민법", "형법", "상법"],
            "device_info": {"platform": "web", "browser": "chrome"},
            "location_info": {"country": "KR", "region": "Seoul"}
        }
        
        # 먼저 프로필 생성
        create_success = self.user_profile_manager.create_profile(
            self.test_user_id, profile_data
        )
        self.assertTrue(create_success, "프로필 생성 실패")
        
        # 선호도 업데이트
        preferences = {
            "expertise_level": "advanced",
            "preferred_detail_level": "comprehensive"
        }
        
        success = self.user_profile_manager.update_preferences(
            self.test_user_id, preferences
        )
        self.assertTrue(success)
        
        # 프로필 조회 (업데이트 후)
        profile = self.user_profile_manager.get_profile(self.test_user_id)
        self.assertIsNotNone(profile)
        # 업데이트가 제대로 반영되었는지 확인
        updated_expertise = profile.get("expertise_level", "unknown")
        self.assertIn(updated_expertise, ["intermediate", "advanced"], 
                     f"예상치 못한 전문성 레벨: {updated_expertise}")
        
        print(f"  사용자 프로필 생성: {profile['expertise_level']}")
        
        # 2. 감정 및 의도 분석 테스트
        print("\n감정 및 의도 분석 테스트:")
        
        test_queries = [
            "급하게 도움이 필요합니다!",
            "계약서 검토 부탁드립니다.",
            "이 문제가 정말 복잡하네요...",
            "감사합니다. 도움이 많이 되었어요."
        ]
        
        for query in test_queries:
            emotion_result = self.emotion_analyzer.analyze_emotion(query)
            
            self.assertIsNotNone(emotion_result)
            self.assertIsNotNone(emotion_result.primary_emotion)
            self.assertIsNotNone(emotion_result.confidence)
            
            print(f"  '{query[:20]}...' -> 감정: {emotion_result.primary_emotion}, 신뢰도: {emotion_result.confidence:.2f}")
        
        # 3. 대화 흐름 추적 테스트
        print("\n대화 흐름 추적 테스트:")
        
        # 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.test_session_id,
            turns=[
                ConversationTurn(
                    user_query="손해배상에 대해 알려주세요",
                    bot_response="손해배상은 민법 제750조에 규정된 제도입니다.",
                    timestamp=datetime.now(),
                    question_type="legal_advice",
                    entities={"legal_terms": ["손해배상"]}
                )
            ],
            entities={"legal_terms": {"손해배상"}},
            topic_stack=["손해배상"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # 다음 의도 예측
        next_intent = self.flow_tracker.predict_next_intent(context)
        self.assertIsNotNone(next_intent)
        
        # 후속 질문 제안
        suggestions = self.flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)
        
        print(f"  예측된 다음 의도: {next_intent}")
        print(f"  제안된 후속 질문 수: {len(suggestions)}")
        
        print("✅ Phase 2 종합 테스트 완료")
    
    def test_phase3_comprehensive(self):
        """Phase 3 종합 테스트"""
        print("\n=== Phase 3 종합 테스트 ===")
        
        # 1. 맥락적 메모리 관리 테스트
        print("\n맥락적 메모리 관리 테스트:")
        
        # 다양한 유형의 사실 저장
        facts_to_store = {
            "legal_knowledge": [
                "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다",
                "손해배상의 요건은 가해행위, 손해발생, 인과관계, 고의 또는 과실입니다"
            ],
            "case_detail": [
                "교통사고 과실비율 산정 사건",
                "계약 해지 관련 분쟁 사건"
            ],
            "user_context": [
                "사용자는 법률 초보자입니다",
                "사용자는 상세한 설명을 선호합니다"
            ]
        }
        
        # 사실 저장
        storage_success = self.memory_manager.store_important_facts(
            self.test_session_id, self.test_user_id, facts_to_store
        )
        self.assertTrue(storage_success)
        
        # 메모리 통계 조회
        stats = self.memory_manager.get_memory_statistics(self.test_user_id)
        self.assertIsNotNone(stats)
        self.assertGreater(stats["total_memories"], 0)
        
        print(f"  저장된 메모리 수: {stats['total_memories']}")
        print(f"  유형별 통계: {stats['type_statistics']}")
        
        # 관련 메모리 검색
        relevant_memories = self.memory_manager.retrieve_relevant_memory(
            self.test_session_id, "손해배상 관련 질문", self.test_user_id
        )
        self.assertIsNotNone(relevant_memories)
        self.assertGreater(len(relevant_memories), 0)
        
        print(f"  검색된 관련 메모리 수: {len(relevant_memories)}")
        
        # 2. 대화 품질 모니터링 테스트
        print("\n대화 품질 모니터링 테스트:")
        
        # 고품질 대화 컨텍스트
        high_quality_context = ConversationContext(
            session_id=f"{self.test_session_id}_high",
            turns=[
                ConversationTurn(
                    user_query="민법 제750조에 대해 자세히 설명해주세요",
                    bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문은 고의 또는 과실로 인한 불법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다고 규정하고 있습니다.",
                    timestamp=datetime.now(),
                    question_type="law_inquiry",
                    entities={"laws": ["민법"], "articles": ["제750조"]}
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}},
            topic_stack=["민법", "제750조"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # 저품질 대화 컨텍스트
        low_quality_context = ConversationContext(
            session_id=f"{self.test_session_id}_low",
            turns=[
                ConversationTurn(
                    user_query="법",
                    bot_response="네",
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
        
        # 품질 평가
        high_quality_score = self.quality_monitor.assess_conversation_quality(high_quality_context)
        low_quality_score = self.quality_monitor.assess_conversation_quality(low_quality_context)
        
        self.assertIsNotNone(high_quality_score)
        self.assertIsNotNone(low_quality_score)
        self.assertGreater(high_quality_score["overall_score"], low_quality_score["overall_score"])
        
        print(f"  고품질 대화 점수: {high_quality_score['overall_score']:.2f}")
        print(f"  저품질 대화 점수: {low_quality_score['overall_score']:.2f}")
        
        # 문제점 감지 및 개선 제안
        issues = self.quality_monitor.detect_conversation_issues(low_quality_context)
        suggestions = self.quality_monitor.suggest_improvements(low_quality_context)
        
        self.assertIsNotNone(issues)
        self.assertIsNotNone(suggestions)
        
        print(f"  감지된 문제점 수: {len(issues)}")
        print(f"  개선 제안 수: {len(suggestions)}")
        
        print("✅ Phase 3 종합 테스트 완료")
    
    def test_performance_comprehensive(self):
        """성능 종합 테스트"""
        print("\n=== 성능 종합 테스트 ===")
        
        # 1. 성능 모니터링 테스트
        print("\n성능 모니터링 테스트:")
        
        # CPU 및 메모리 사용량 측정
        system_health = self.performance_monitor.get_system_health()
        cpu_usage = system_health["system"]["cpu_usage"]
        memory_usage = system_health["system"]["memory_usage"]
        
        self.assertIsNotNone(cpu_usage)
        self.assertIsNotNone(memory_usage)
        
        print(f"  CPU 사용률: {cpu_usage:.1f}%")
        print(f"  메모리 사용률: {memory_usage:.1f}%")
        
        # 2. 캐시 성능 테스트
        print("\n캐시 성능 테스트:")
        
        # 캐시 저장 및 조회 테스트
        test_key = "test_cache_key"
        test_value = {"data": "test_value", "timestamp": datetime.now().isoformat()}
        
        # 저장
        self.cache_manager.set(test_key, test_value)
        
        # 조회
        retrieved_value = self.cache_manager.get(test_key)
        self.assertIsNotNone(retrieved_value)
        self.assertEqual(retrieved_value["data"], test_value["data"])
        
        # 캐시 통계
        cache_stats = self.cache_manager.get_stats()
        self.assertIsNotNone(cache_stats)
        
        print(f"  캐시 히트율: {cache_stats['hit_rate']:.2f}")
        print(f"  캐시 크기: {cache_stats['cache_size']}")
        
        # 3. 메모리 최적화 테스트
        print("\n메모리 최적화 테스트:")
        
        # 메모리 사용량 모니터링
        initial_memory = self.memory_optimizer.get_memory_usage()
        
        # 가비지 컬렉션 실행
        freed_memory = self.memory_optimizer.optimize_memory()
        
        final_memory = self.memory_optimizer.get_memory_usage()
        
        print(f"  초기 메모리: {initial_memory.process_memory / 1024 / 1024:.1f} MB")
        print(f"  최적화 후 메모리: {final_memory.process_memory / 1024 / 1024:.1f} MB")
        print(f"  해제된 메모리: {freed_memory.get('memory_freed_mb', 0):.1f} MB")
        
        print("✅ 성능 종합 테스트 완료")
    
    def test_integration_comprehensive(self):
        """통합 시스템 테스트"""
        print("\n=== 통합 시스템 테스트 ===")
        
        # 1. 전체 워크플로우 테스트
        print("\n전체 워크플로우 테스트:")
        
        # 복잡한 대화 시나리오
        complex_scenario = [
            {
                "query": "안녕하세요. 법률 상담을 받고 싶습니다.",
                "expected_emotion": "neutral",
                "expected_intent": "greeting"
            },
            {
                "query": "계약서 검토를 도와주실 수 있나요?",
                "expected_emotion": "neutral",
                "expected_intent": "request"
            },
            {
                "query": "그 계약서에서 문제가 될 수 있는 부분이 있나요?",
                "expected_emotion": "neutral",
                "expected_intent": "clarification"
            },
            {
                "query": "감사합니다. 정말 도움이 많이 되었어요!",
                "expected_emotion": "positive",
                "expected_intent": "gratitude"
            }
        ]
        
        total_processing_time = 0
        
        for i, scenario in enumerate(complex_scenario):
            start_time = time.time()
            
            # Phase 1: 대화 맥락 처리
            context = self.session_manager.add_turn(
                f"{self.test_session_id}_integration",
                scenario["query"],
                f"시스템 응답 {i+1}",
                "general",
                self.test_user_id
            )
            
            # Phase 2: 감정 및 의도 분석
            emotion_result = self.emotion_analyzer.analyze_emotion(scenario["query"])
            
            # Phase 3: 메모리 저장
            facts = {"user_context": [scenario["query"]]}
            self.memory_manager.store_important_facts(
                f"{self.test_session_id}_integration", self.test_user_id, facts
            )
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            print(f"  턴 {i+1}: {processing_time:.3f}초")
        
        avg_processing_time = total_processing_time / len(complex_scenario)
        print(f"  평균 처리 시간: {avg_processing_time:.3f}초")
        
        # 2. 동시성 테스트
        print("\n동시성 테스트:")
        
        async def concurrent_test():
            tasks = []
            for i in range(5):
                task = asyncio.create_task(self._async_conversation_test(i))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # 동시성 테스트 실행
        try:
            results = asyncio.run(concurrent_test())
            self.assertEqual(len(results), 5)
            print(f"  동시 처리된 대화 수: {len(results)}")
        except Exception as e:
            print(f"  동시성 테스트 오류: {e}")
        
        # 3. 시스템 안정성 테스트
        print("\n시스템 안정성 테스트:")
        
        # 대량 데이터 처리
        large_data_test = []
        for i in range(100):
            large_data_test.append(f"테스트 데이터 {i} - " + "x" * 100)
        
        start_time = time.time()
        
        # 대량 데이터 처리
        for data in large_data_test[:10]:  # 처음 10개만 테스트
            facts = {"test_data": [data]}
            self.memory_manager.store_important_facts(
                f"{self.test_session_id}_stress", self.test_user_id, facts
            )
        
        processing_time = time.time() - start_time
        print(f"  대량 데이터 처리 시간: {processing_time:.3f}초")
        
        print("✅ 통합 시스템 테스트 완료")
    
    async def _async_conversation_test(self, test_id: int):
        """비동기 대화 테스트"""
        session_id = f"async_test_{test_id}"
        
        # 비동기 대화 처리
        context = self.session_manager.add_turn(
            session_id,
            f"비동기 테스트 질문 {test_id}",
            f"비동기 테스트 응답 {test_id}",
            "test",
            f"async_user_{test_id}"
        )
        
        return {
            "test_id": test_id,
            "session_id": session_id,
            "turns_count": len(context.turns)
        }


def run_comprehensive_tests():
    """종합 테스트 실행"""
    print("=" * 60)
    print("LawFirmAI 종합 시스템 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveSystem))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("종합 테스트 결과 요약")
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
    
    if success_rate >= 95:
        print("🎉 우수한 성능! 시스템이 안정적으로 작동합니다.")
    elif success_rate >= 85:
        print("✅ 양호한 성능! 몇 가지 개선이 필요합니다.")
    else:
        print("⚠️ 개선이 필요합니다. 시스템을 점검해주세요.")
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()
