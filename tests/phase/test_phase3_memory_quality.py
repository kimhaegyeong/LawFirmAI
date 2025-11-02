# -*- coding: utf-8 -*-
"""
Phase 3: 장기 기억 및 품질 모니터링 기능 테스트 스크립트
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

# 프로젝트 루트 경로 추가
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

        # 중요한 사실 저장
        facts = {
            "legal_knowledge": [
                "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다",
                "계약 해제 시 위약금은 손해배상액을 초과할 수 없습니다"
            ],
            "user_context": [
                "사용자는 부동산 관련 문제를 자주 문의합니다",
                "사용자는 법률 초보자입니다"
            ],
            "preference": [
                "사용자는 상세한 설명을 선호합니다",
                "사용자는 판례 중심의 답변을 원합니다"
            ]
        }

        success = self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)
        self.assertTrue(success)

        # 저장된 메모리 확인
        memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        self.assertGreater(len(memories), 0)

    def test_contextual_memory_manager_retrieve_memory(self):
        print("\n--- Test ContextualMemoryManager: Retrieve Memory ---")

        # 테스트 메모리 저장
        facts = {
            "legal_knowledge": ["민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다"]
        }
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # 관련 메모리 검색
        query = "손해배상 관련 질문입니다"
        relevant_memories = self.memory_manager.retrieve_relevant_memory(self.session_id, query, self.user_id)

        self.assertIsNotNone(relevant_memories)
        self.assertIsInstance(relevant_memories, list)
        if relevant_memories:
            self.assertGreater(relevant_memories[0].relevance_score, 0)

    def test_contextual_memory_manager_extract_facts(self):
        print("\n--- Test ContextualMemoryManager: Extract Facts ---")

        # 테스트 대화 턴
        turn = ConversationTurn(
            user_query="민법 제750조에 대해 자세히 설명해주세요",
            bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다...",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # 사실 추출
        extracted_facts = self.memory_manager.extract_facts_from_conversation(turn)

        self.assertIsNotNone(extracted_facts)
        self.assertIsInstance(extracted_facts, list)
        self.assertGreater(len(extracted_facts), 0)

    def test_contextual_memory_manager_consolidate_memories(self):
        print("\n--- Test ContextualMemoryManager: Consolidate Memories ---")

        # 중복 메모리 저장
        facts1 = {"legal_knowledge": ["민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다"]}
        facts2 = {"legal_knowledge": ["민법 제750조는 불법행위 손해배상 책임을 규정합니다"]}  # 유사한 내용

        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts1)
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts2)

        # 메모리 통합
        consolidated_count = self.memory_manager.consolidate_memories(self.user_id)
        self.assertGreaterEqual(consolidated_count, 0)

    def test_contextual_memory_manager_statistics(self):
        print("\n--- Test ContextualMemoryManager: Statistics ---")

        # 테스트 메모리 저장
        facts = {
            "legal_knowledge": ["민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다"],
            "user_context": ["사용자는 법률 초보자입니다"]
        }
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # 통계 조회
        stats = self.memory_manager.get_memory_statistics(self.user_id)

        self.assertIsNotNone(stats)
        self.assertIn("user_id", stats)
        self.assertIn("total_memories", stats)
        self.assertGreater(stats["total_memories"], 0)

    def test_contextual_memory_manager_importance_update(self):
        print("\n--- Test ContextualMemoryManager: Importance Update ---")

        # 테스트 메모리 저장
        facts = {"legal_knowledge": ["민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다"]}
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # 메모리 검색
        memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        self.assertGreater(len(memories), 0)

        # 중요도 업데이트
        memory_id = memories[0].memory_id
        success = self.memory_manager.update_memory_importance(memory_id, 0.9)
        self.assertTrue(success)

    def test_contextual_memory_manager_cleanup(self):
        print("\n--- Test ContextualMemoryManager: Cleanup ---")

        # 테스트 메모리 저장
        facts = {"legal_knowledge": ["민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다"]}
        self.memory_manager.store_important_facts(self.session_id, self.user_id, facts)

        # 오래된 메모리 정리 (테스트를 위해 1일로 설정)
        cleaned_count = self.memory_manager.cleanup_old_memories(days=1)
        self.assertGreaterEqual(cleaned_count, 0)

    def test_conversation_quality_monitor_assess_quality(self):
        print("\n--- Test ConversationQualityMonitor: Assess Quality ---")

        # 테스트 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="민법 제750조에 대해 자세히 설명해주세요",
                    bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문에 따르면...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="감사합니다. 정말 도움이 되었어요!",
                    bot_response="천만에요. 추가로 궁금한 것이 있으시면 언제든지 문의해주세요.",
                    timestamp=datetime.now(),
                    question_type="thanks"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["민법", "손해배상"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 품질 평가
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

        # 문제가 있는 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="더 자세히 설명해주세요. 이해가 안 돼요",
                    bot_response="네, 설명드리겠습니다.",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 문제점 감지
        issues = self.quality_monitor.detect_conversation_issues(context)

        self.assertIsNotNone(issues)
        self.assertIsInstance(issues, list)

    def test_conversation_quality_monitor_suggest_improvements(self):
        print("\n--- Test ConversationQualityMonitor: Suggest Improvements ---")

        # 테스트 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="민법 제750조에 대해 설명해주세요",
                    bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다.",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["민법"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 개선 제안 생성
        suggestions = self.quality_monitor.suggest_improvements(context)

        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)

    def test_conversation_quality_monitor_calculate_turn_quality(self):
        print("\n--- Test ConversationQualityMonitor: Calculate Turn Quality ---")

        # 테스트 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["민법"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 테스트 턴
        turn = ConversationTurn(
            user_query="민법 제750조에 대해 자세히 설명해주세요",
            bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문에 따르면...",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # 턴 품질 계산
        turn_quality = self.quality_monitor.calculate_turn_quality(turn, context)

        self.assertIsNotNone(turn_quality)
        self.assertIn("completeness", turn_quality)
        self.assertIn("satisfaction", turn_quality)
        self.assertIn("accuracy", turn_quality)
        self.assertGreaterEqual(turn_quality["completeness"], 0.0)
        self.assertLessEqual(turn_quality["completeness"], 1.0)

    def test_conversation_quality_monitor_dashboard_data(self):
        print("\n--- Test ConversationQualityMonitor: Dashboard Data ---")

        # 품질 대시보드 데이터 생성
        dashboard_data = self.quality_monitor.get_quality_dashboard_data(self.user_id)

        self.assertIsNotNone(dashboard_data)
        self.assertIsInstance(dashboard_data, dict)

    def test_conversation_quality_monitor_trend_analysis(self):
        print("\n--- Test ConversationQualityMonitor: Trend Analysis ---")

        # 품질 트렌드 분석
        trend_analysis = self.quality_monitor.analyze_quality_trends([self.session_id])

        self.assertIsNotNone(trend_analysis)
        self.assertIsInstance(trend_analysis, dict)

    def test_integrated_phase3_components(self):
        print("\n--- Test Integrated Phase 3 Components ---")

        # 테스트 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="민법 제750조에 대해 자세히 설명해주세요",
                    bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문에 따르면...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="감사합니다. 정말 도움이 되었어요!",
                    bot_response="천만에요. 추가로 궁금한 것이 있으시면 언제든지 문의해주세요.",
                    timestamp=datetime.now(),
                    question_type="thanks"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["민법", "손해배상"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 통합 테스트: 메모리 관리 + 품질 모니터링

        # 1. 대화에서 사실 추출 및 저장
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

        # 2. 관련 메모리 검색
        query = "손해배상 관련 질문입니다"
        relevant_memories = self.memory_manager.retrieve_relevant_memory(
            self.session_id, query, self.user_id
        )

        # 디버깅: 저장된 메모리 확인
        all_memories = self.memory_manager._search_memories(self.session_id, self.user_id)
        print(f"   디버깅: 저장된 메모리 수: {len(all_memories)}")
        for i, mem in enumerate(all_memories):
            print(f"   메모리 {i+1}: {mem.content[:50]}... (점수: {mem.importance_score})")
            # 관련성 점수 계산
            score = self.memory_manager._calculate_relevance_score(mem, query)
            print(f"   관련성 점수: {score}")

        self.assertIsNotNone(relevant_memories)

        # 3. 대화 품질 평가
        quality_assessment = self.quality_monitor.assess_conversation_quality(context)
        self.assertIsNotNone(quality_assessment)
        self.assertGreaterEqual(quality_assessment["overall_score"], 0.0)

        # 4. 문제점 감지 및 개선 제안
        issues = self.quality_monitor.detect_conversation_issues(context)
        suggestions = self.quality_monitor.suggest_improvements(context)

        self.assertIsNotNone(issues)
        self.assertIsNotNone(suggestions)

        # 5. 메모리 통계 및 품질 대시보드
        memory_stats = self.memory_manager.get_memory_statistics(self.user_id)
        quality_dashboard = self.quality_monitor.get_quality_dashboard_data(self.user_id)

        self.assertIsNotNone(memory_stats)
        self.assertIsNotNone(quality_dashboard)

        # 통합 결과 검증
        self.assertGreater(len(relevant_memories), 0)
        self.assertGreater(quality_assessment["overall_score"], 0.0)
        self.assertGreater(memory_stats["total_memories"], 0)

        print(f"   통합 테스트 완료:")
        print(f"   - 저장된 메모리 수: {memory_stats['total_memories']}")
        print(f"   - 검색된 관련 메모리 수: {len(relevant_memories)}")
        print(f"   - 대화 품질 점수: {quality_assessment['overall_score']:.2f}")
        print(f"   - 감지된 문제점 수: {len(issues)}")
        print(f"   - 개선 제안 수: {len(suggestions)}")

    def test_memory_quality_integration(self):
        print("\n--- Test Memory-Quality Integration ---")

        # 품질이 높은 대화와 낮은 대화 생성
        high_quality_context = ConversationContext(
            session_id=f"{self.session_id}_high",
            turns=[
                ConversationTurn(
                    user_query="민법 제750조에 대해 자세히 설명해주세요",
                    bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문에 따르면 고의 또는 과실로 인하여 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있습니다. 이는 민법의 핵심 원칙 중 하나로...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["민법"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        low_quality_context = ConversationContext(
            session_id=f"{self.session_id}_low",
            turns=[
                ConversationTurn(
                    user_query="더 자세히 설명해주세요. 이해가 안 돼요",
                    bot_response="네, 설명드리겠습니다.",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 각 대화의 품질 평가
        high_quality_assessment = self.quality_monitor.assess_conversation_quality(high_quality_context)
        low_quality_assessment = self.quality_monitor.assess_conversation_quality(low_quality_context)

        # 품질이 높은 대화에서 더 많은 사실 추출
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

        # 메모리 저장
        self.memory_manager.store_important_facts(
            high_quality_context.session_id, self.user_id, high_quality_facts
        )
        self.memory_manager.store_important_facts(
            low_quality_context.session_id, self.user_id, low_quality_facts
        )

        # 통계 확인
        memory_stats = self.memory_manager.get_memory_statistics(self.user_id)

        # 검증
        self.assertGreater(high_quality_assessment["overall_score"], low_quality_assessment["overall_score"])
        self.assertGreater(memory_stats["total_memories"], 0)

        print(f"   메모리-품질 통합 테스트 완료:")
        print(f"   - 고품질 대화 점수: {high_quality_assessment['overall_score']:.2f}")
        print(f"   - 저품질 대화 점수: {low_quality_assessment['overall_score']:.2f}")
        print(f"   - 총 메모리 수: {memory_stats['total_memories']}")


if __name__ == '__main__':
    unittest.main()
