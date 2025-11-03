# -*- coding: utf-8 -*-
"""
Phase 1 단위 및 통합 테스트
ConversationStore 확장, IntegratedSessionManager, MultiTurnQuestionHandler, ContextCompressor 테스트
"""

import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.data.conversation_store import ConversationStore
from source.services.integrated_session_manager import IntegratedSessionManager
from lawfirm_langgraph.langgraph_core.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor
from lawfirm_langgraph.langgraph_core.services.conversation_manager import ConversationContext, ConversationTurn


class TestConversationStoreExtensions(unittest.TestCase):
    """ConversationStore 확장 기능 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_conversations.db")
        self.store = ConversationStore(self.db_path)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_user_sessions(self):
        """사용자별 세션 조회 테스트"""
        # 테스트 세션 데이터 생성
        test_session = {
            "session_id": "test_user_session_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["손해배상", "계약"],
            "metadata": {"user_id": "test_user_001"},
            "turns": [
                {
                    "user_query": "손해배상 청구 방법을 알려주세요",
                    "bot_response": "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"laws": ["민법"], "articles": ["제750조"]}
                }
            ],
            "entities": {
                "laws": ["민법"],
                "articles": ["제750조"],
                "precedents": [],
                "legal_terms": ["손해배상"]
            }
        }
        
        # 세션 저장
        self.assertTrue(self.store.save_session(test_session))
        
        # 사용자 세션 조회
        user_sessions = self.store.get_user_sessions("test_user_001")
        self.assertEqual(len(user_sessions), 1)
        self.assertEqual(user_sessions[0]["session_id"], "test_user_session_001")
    
    def test_session_search(self):
        """세션 검색 테스트"""
        # 테스트 세션들 생성
        sessions = [
            {
                "session_id": "search_test_001",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "topic_stack": ["손해배상"],
                "metadata": {"user_id": "user_001"},
                "turns": [
                    {
                        "user_query": "손해배상 청구 방법",
                        "bot_response": "손해배상 관련 답변",
                        "timestamp": datetime.now().isoformat(),
                        "question_type": "legal_advice",
                        "entities": {"legal_terms": ["손해배상"]}
                    }
                ],
                "entities": {"legal_terms": ["손해배상"]}
            },
            {
                "session_id": "search_test_002",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "topic_stack": ["계약"],
                "metadata": {"user_id": "user_002"},
                "turns": [
                    {
                        "user_query": "계약 해지 절차",
                        "bot_response": "계약 관련 답변",
                        "timestamp": datetime.now().isoformat(),
                        "question_type": "procedure_guide",
                        "entities": {"legal_terms": ["계약"]}
                    }
                ],
                "entities": {"legal_terms": ["계약"]}
            }
        ]
        
        # 세션들 저장
        for session in sessions:
            self.assertTrue(self.store.save_session(session))
        
        # 키워드 검색
        results = self.store.search_sessions("손해배상", {"limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "search_test_001")
        
        # 사용자 필터 검색
        results = self.store.search_sessions("", {"user_id": "user_001", "limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "search_test_001")
    
    def test_session_backup_restore(self):
        """세션 백업 및 복원 테스트"""
        # 테스트 세션 생성
        test_session = {
            "session_id": "backup_test_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["손해배상"],
            "metadata": {},
            "turns": [
                {
                    "user_query": "손해배상 청구 방법",
                    "bot_response": "손해배상 관련 답변",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"legal_terms": ["손해배상"]}
                }
            ],
            "entities": {"legal_terms": ["손해배상"]}
        }
        
        # 세션 저장
        self.assertTrue(self.store.save_session(test_session))
        
        # 백업
        backup_dir = os.path.join(self.temp_dir, "backup")
        self.assertTrue(self.store.backup_session("backup_test_001", backup_dir))
        
        # 백업 파일 확인
        backup_file = os.path.join(backup_dir, "backup_test_001_backup.json")
        self.assertTrue(os.path.exists(backup_file))
        
        # 복원
        restored_session_id = self.store.restore_session(backup_file)
        self.assertIsNotNone(restored_session_id)
        self.assertTrue(restored_session_id.startswith("backup_test_001_restored_"))
    
    def test_statistics(self):
        """통계 조회 테스트"""
        # 테스트 데이터 생성
        test_session = {
            "session_id": "stats_test_001",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "topic_stack": ["손해배상"],
            "metadata": {},
            "turns": [
                {
                    "user_query": "손해배상 청구 방법",
                    "bot_response": "손해배상 관련 답변",
                    "timestamp": datetime.now().isoformat(),
                    "question_type": "legal_advice",
                    "entities": {"legal_terms": ["손해배상"]}
                }
            ],
            "entities": {"legal_terms": ["손해배상"]}
        }
        
        # 세션 저장
        self.assertTrue(self.store.save_session(test_session))
        
        # 통계 조회
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
    """IntegratedSessionManager 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integrated_conversations.db")
        self.manager = IntegratedSessionManager(self.db_path)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_add_turn(self):
        """턴 추가 테스트"""
        session_id = "test_session_001"
        user_id = "test_user_001"
        
        # 턴 추가
        context = self.manager.add_turn(
            session_id, 
            "손해배상 청구 방법을 알려주세요",
            "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            "legal_advice",
            user_id
        )
        
        self.assertIsNotNone(context)
        self.assertEqual(context.session_id, session_id)
        self.assertEqual(len(context.turns), 1)
        self.assertEqual(context.turns[0].user_query, "손해배상 청구 방법을 알려주세요")
    
    def test_session_persistence(self):
        """세션 지속성 테스트"""
        session_id = "test_persistence_001"
        user_id = "test_user_001"
        
        # 턴 추가
        context1 = self.manager.add_turn(
            session_id,
            "손해배상 청구 방법을 알려주세요",
            "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            "legal_advice",
            user_id
        )
        
        # 강제 동기화
        self.assertTrue(self.manager.sync_to_database(session_id))
        
        # 새로운 매니저 인스턴스로 세션 로드
        new_manager = IntegratedSessionManager(self.db_path)
        context2 = new_manager.get_or_create_session(session_id, user_id)
        
        self.assertIsNotNone(context2)
        self.assertEqual(len(context2.turns), 1)
        self.assertEqual(context2.turns[0].user_query, "손해배상 청구 방법을 알려주세요")
    
    def test_user_sessions(self):
        """사용자 세션 조회 테스트"""
        user_id = "test_user_002"
        
        # 여러 세션 생성
        for i in range(3):
            session_id = f"test_user_sessions_{i:03d}"
            self.manager.add_turn(
                session_id,
                f"질문 {i}",
                f"답변 {i}",
                "legal_advice",
                user_id
            )
            self.manager.sync_to_database(session_id)
        
        # 사용자 세션 조회
        user_sessions = self.manager.get_user_sessions(user_id)
        self.assertEqual(len(user_sessions), 3)
    
    def test_session_search(self):
        """세션 검색 테스트"""
        user_id = "test_user_003"
        
        # 검색용 세션 생성
        session_id = "test_search_001"
        self.manager.add_turn(
            session_id,
            "손해배상 청구 방법",
            "손해배상 관련 답변",
            "legal_advice",
            user_id
        )
        self.manager.sync_to_database(session_id)
        
        # 검색 실행
        results = self.manager.search_sessions("손해배상", {"limit": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], session_id)
    
    def test_cleanup(self):
        """세션 정리 테스트"""
        # 오래된 세션 생성
        old_session_id = "test_cleanup_001"
        self.manager.add_turn(
            old_session_id,
            "오래된 질문",
            "오래된 답변",
            "legal_advice"
        )
        
        # 정리 실행
        cleaned_count = self.manager.cleanup_old_sessions(days=0)  # 즉시 정리
        self.assertGreaterEqual(cleaned_count, 0)


class TestMultiTurnQuestionHandler(unittest.TestCase):
    """MultiTurnQuestionHandler 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.handler = MultiTurnQuestionHandler()
        
        # 테스트용 대화 맥락 생성
        self.test_turns = [
            ConversationTurn(
                user_query="계약 해지 절차는 어떻게 되나요?",
                bot_response="계약 해지 절차는 다음과 같습니다...",
                timestamp=datetime.now(),
                question_type="procedure_guide",
                entities={"legal_terms": ["계약", "해지"]}
            ),
            ConversationTurn(
                user_query="손해배상 청구 방법을 알려주세요",
                bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"laws": ["민법"], "articles": ["제750조"], "legal_terms": ["손해배상"]}
            )
        ]
        
        self.context = ConversationContext(
            session_id="test_session",
            turns=self.test_turns,
            entities={"laws": {"민법"}, "articles": {"제750조"}, 
                     "precedents": set(), "legal_terms": {"손해배상", "계약", "해지"}},
            topic_stack=["손해배상", "계약"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def test_detect_multi_turn_question(self):
        """다중 턴 질문 감지 테스트"""
        # 다중 턴 질문들
        multi_turn_queries = [
            "그것에 대해 더 자세히 알려주세요",
            "위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
            "그 판례는 어떤 사건이었나요?",
            "이것의 법적 근거는 무엇인가요?"
        ]
        
        for query in multi_turn_queries:
            with self.subTest(query=query):
                is_multi_turn = self.handler.detect_multi_turn_question(query, self.context)
                self.assertTrue(is_multi_turn, f"'{query}' should be detected as multi-turn")
        
        # 일반 질문
        normal_query = "일반적인 질문입니다"
        is_multi_turn = self.handler.detect_multi_turn_question(normal_query, self.context)
        self.assertFalse(is_multi_turn, f"'{normal_query}' should not be detected as multi-turn")
    
    def test_resolve_pronouns(self):
        """대명사 해결 테스트"""
        # 대명사가 포함된 질문
        pronoun_query = "그것에 대해 더 자세히 알려주세요"
        resolved_query = self.handler.resolve_pronouns(pronoun_query, self.context)
        
        self.assertNotEqual(pronoun_query, resolved_query)
        # 대명사가 해결되어야 함 (손해배상, 민법, 계약, 해지 중 하나)
        self.assertTrue(
            any(keyword in resolved_query for keyword in ["손해배상", "민법", "계약", "해지"]),
            f"Expected one of ['손해배상', '민법', '계약', '해지'] in '{resolved_query}'"
        )
    
    def test_build_complete_query(self):
        """완전한 질문 구성 테스트"""
        query = "그것에 대해 더 자세히 알려주세요"
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
        """참조 엔티티 추출 테스트"""
        # 엔티티 참조가 포함된 질문
        query = "그 판례는 어떤 사건이었나요?"
        entities = self.handler.extract_reference_entities(query)
        
        self.assertIsInstance(entities, list)
        # 판례 참조가 감지되어야 함
        self.assertTrue(any("판례" in entity for entity in entities))


class TestContextCompressor(unittest.TestCase):
    """ContextCompressor 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.compressor = ContextCompressor(max_tokens=1000)
        
        # 테스트용 대화 맥락 생성
        self.test_turns = [
            ConversationTurn(
                user_query="손해배상 청구 방법을 알려주세요",
                bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다. 손해배상은 불법행위로 인한 손해를 배상받는 제도입니다. 손해의 발생, 가해자의 고의 또는 과실, 인과관계, 손해의 발생이 필요합니다.",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"laws": ["민법"], "articles": ["제750조"], "legal_terms": ["손해배상", "불법행위"]}
            ),
            ConversationTurn(
                user_query="계약 해지 절차는 어떻게 되나요?",
                bot_response="계약 해지 절차는 다음과 같습니다. 1) 해지 사유 확인 2) 해지 통지 3) 손해배상 청구 4) 소송 제기 등이 있습니다. 민법 제543조에 따라 계약은 당사자 일방의 의사표시로 해지할 수 있습니다.",
                timestamp=datetime.now(),
                question_type="procedure_guide",
                entities={"laws": ["민법"], "articles": ["제543조"], "legal_terms": ["계약", "해지"]}
            ),
            ConversationTurn(
                user_query="위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
                bot_response="과실비율은 교통사고의 경우 보험회사에서 정한 기준표를 참고하여 결정됩니다. 대법원 2023다12345 판례에 따르면 과실비율은 사고 상황, 도로 상황, 차량 상태 등을 종합적으로 고려하여 결정됩니다.",
                timestamp=datetime.now(),
                question_type="legal_advice",
                entities={"precedents": ["2023다12345"], "legal_terms": ["과실비율", "교통사고"]}
            )
        ]
        
        self.context = ConversationContext(
            session_id="test_session",
            turns=self.test_turns,
            entities={"laws": {"민법"}, "articles": {"제750조", "제543조"}, 
                     "precedents": {"2023다12345"}, 
                     "legal_terms": {"손해배상", "불법행위", "계약", "해지", "과실비율", "교통사고"}},
            topic_stack=["손해배상", "계약", "교통사고"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def test_calculate_tokens(self):
        """토큰 수 계산 테스트"""
        tokens = self.compressor.calculate_tokens(self.context)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_calculate_tokens_from_text(self):
        """텍스트 토큰 수 계산 테스트"""
        text = "손해배상 청구 방법을 알려주세요"
        tokens = self.compressor.calculate_tokens_from_text(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_extract_key_information(self):
        """핵심 정보 추출 테스트"""
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
        """관련 컨텍스트 유지 테스트"""
        relevant_turns = self.compressor.maintain_relevant_context(self.context, "")
        
        self.assertIsInstance(relevant_turns, list)
        self.assertLessEqual(len(relevant_turns), len(self.context.turns))
        # 최근 턴은 항상 포함되어야 함
        self.assertIn(self.context.turns[-1], relevant_turns)
    
    def test_compress_long_conversation(self):
        """긴 대화 압축 테스트"""
        # 작은 토큰 제한으로 압축 강제
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
    """Phase 1 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        
        # 모든 컴포넌트 초기화
        self.session_manager = IntegratedSessionManager(self.db_path)
        self.multi_turn_handler = MultiTurnQuestionHandler()
        self.context_compressor = ContextCompressor(max_tokens=500)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_conversation(self):
        """전체 대화 흐름 테스트"""
        session_id = "integration_test_001"
        user_id = "test_user_001"
        
        # 1. 첫 번째 질문
        context1 = self.session_manager.add_turn(
            session_id,
            "손해배상 청구 방법을 알려주세요",
            "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            "legal_advice",
            user_id
        )
        
        self.assertEqual(len(context1.turns), 1)
        
        # 2. 다중 턴 질문 처리
        multi_turn_query = "그것에 대해 더 자세히 알려주세요"
        is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(multi_turn_query, context1)
        self.assertTrue(is_multi_turn)
        
        resolved_result = self.multi_turn_handler.build_complete_query(multi_turn_query, context1)
        resolved_query = resolved_result["resolved_query"]
        
        # 3. 해결된 질문으로 두 번째 턴 추가
        context2 = self.session_manager.add_turn(
            session_id,
            resolved_query,
            "손해배상의 구체적인 요건에 대해 자세히 설명드리겠습니다...",
            "legal_advice",
            user_id
        )
        
        self.assertEqual(len(context2.turns), 2)
        
        # 4. 컨텍스트 압축 테스트
        compression_result = self.context_compressor.compress_long_conversation(context2)
        self.assertLessEqual(compression_result.compressed_tokens, 500)
        
        # 5. 세션 지속성 확인
        self.session_manager.sync_to_database(session_id)
        
        # 새로운 매니저로 세션 로드
        new_manager = IntegratedSessionManager(self.db_path)
        loaded_context = new_manager.get_or_create_session(session_id, user_id)
        
        self.assertEqual(len(loaded_context.turns), 2)
        self.assertEqual(loaded_context.turns[0].user_query, "손해배상 청구 방법을 알려주세요")
    
    def test_performance_metrics(self):
        """성능 메트릭 테스트"""
        session_id = "performance_test_001"
        user_id = "test_user_001"
        
        # 여러 턴 추가
        for i in range(5):
            self.session_manager.add_turn(
                session_id,
                f"질문 {i}",
                f"답변 {i}",
                "legal_advice",
                user_id
            )
        
        # 통계 조회
        stats = self.session_manager.get_session_stats()
        self.assertIn("memory_stats", stats)
        self.assertIn("database_stats", stats)
        self.assertIn("sync_stats", stats)
        
        # 성능 확인
        memory_stats = stats["memory_stats"]
        self.assertEqual(memory_stats["total_sessions"], 1)
        self.assertEqual(memory_stats["total_turns"], 5)


def run_phase1_tests():
    """Phase 1 테스트 실행"""
    print("=== Phase 1 단위 및 통합 테스트 실행 ===")
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 단위 테스트 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestConversationStoreExtensions))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegratedSessionManager))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMultiTurnQuestionHandler))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestContextCompressor))
    
    # 통합 테스트 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPhase1Integration))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print(f"\n=== 테스트 결과 요약 ===")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.failures:
        print(f"\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase1_tests()
    sys.exit(0 if success else 1)

