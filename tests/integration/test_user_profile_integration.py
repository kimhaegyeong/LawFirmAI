# -*- coding: utf-8 -*-
"""
사용자 프로필 통합 테스트
LangGraph 워크플로우에서 UserProfileManager 통합 테스트
"""

import os
import sys
import unittest
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.user_profile_manager import ExpertiseLevel, UserProfileManager
from source.utils.langgraph_config import LangGraphConfig


class TestUserProfileIntegration(unittest.TestCase):
    """사용자 프로필 통합 테스트 클래스"""

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 테스트용 데이터베이스 경로
        cls.test_db_path = "test_conversations.db"

        # 기존 테스트 DB 삭제
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

    def setUp(self):
        """각 테스트 전 초기화"""
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)

    def tearDown(self):
        """각 테스트 후 정리"""
        pass

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 정리"""
        # 테스트 DB 정리
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

    def test_user_profile_manager_initialized(self):
        """UserProfileManager가 제대로 초기화되었는지 테스트"""
        print("\n=== UserProfileManager 초기화 테스트 ===")

        # UserProfileManager가 초기화되었는지 확인
        self.assertIsNotNone(self.workflow.user_profile_manager,
                            "UserProfileManager가 초기화되지 않았습니다")
        print("✅ UserProfileManager 초기화 확인됨")

    def test_create_user_profile(self):
        """사용자 프로필 생성 테스트"""
        print("\n=== 사용자 프로필 생성 테스트 ===")

        if not self.workflow.user_profile_manager:
            self.skipTest("UserProfileManager가 사용할 수 없습니다")

        import time
        user_id = f"test_user_{int(time.time())}"
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "interest_areas": ["민법", "형법"],
            "preferences": {"response_style": "professional"}
        }

        # 프로필 생성
        success = self.workflow.user_profile_manager.create_profile(
            user_id, profile_data
        )
        self.assertTrue(success, "프로필 생성 실패")
        print(f"✅ 프로필 생성 성공: {user_id}")

        # 프로필 조회
        profile = self.workflow.user_profile_manager.get_profile(user_id)
        self.assertIsNotNone(profile, "프로필 조회 실패")
        self.assertEqual(profile["expertise_level"], "intermediate")
        self.assertEqual(profile["preferred_detail_level"], "detailed")
        print(f"✅ 프로필 조회 성공 - 전문성: {profile['expertise_level']}")

    def test_personalized_context(self):
        """개인화된 컨텍스트 생성 테스트"""
        print("\n=== 개인화된 컨텍스트 테스트 ===")

        if not self.workflow.user_profile_manager:
            self.skipTest("UserProfileManager가 사용할 수 없습니다")

        import time
        user_id = f"test_user_ctx_{int(time.time())}"

        # 프로필 생성
        profile_data = {
            "expertise_level": "advanced",
            "preferred_detail_level": "detailed",
            "interest_areas": ["상법", "민법"]
        }
        self.workflow.user_profile_manager.create_profile(user_id, profile_data)

        # 개인화된 컨텍스트 생성
        query = "손해배상 청구 방법을 알려주세요"
        context = self.workflow.user_profile_manager.get_personalized_context(
            user_id, query
        )

        self.assertIsNotNone(context, "개인화된 컨텍스트 생성 실패")
        self.assertIn("expertise_level", context)
        self.assertIn("interest_areas", context)
        self.assertIn("response_style", context)
        print(f"✅ 개인화된 컨텍스트 생성 성공")
        print(f"   - 전문성 수준: {context['expertise_level']}")
        print(f"   - 관심 분야: {context['interest_areas']}")
        print(f"   - 답변 스타일: {context['response_style']}")

    def test_interest_areas_update(self):
        """관심 분야 업데이트 테스트"""
        print("\n=== 관심 분야 업데이트 테스트 ===")

        if not self.workflow.user_profile_manager:
            self.skipTest("UserProfileManager가 사용할 수 없습니다")

        import time
        user_id = f"test_user_update_{int(time.time())}"

        # 프로필 생성
        profile_data = {
            "expertise_level": "beginner",
            "interest_areas": ["민법"]
        }
        self.workflow.user_profile_manager.create_profile(user_id, profile_data)

        # 관심 분야 업데이트
        query = "근로기준법에 따른 퇴직금 계산 방법"
        success = self.workflow.user_profile_manager.update_interest_areas(
            user_id, query
        )
        self.assertTrue(success, "관심 분야 업데이트 실패")

        # 업데이트된 프로필 확인
        profile = self.workflow.user_profile_manager.get_profile(user_id)
        self.assertIn("민법", profile["interest_areas"])
        print(f"✅ 관심 분야 업데이트 성공: {profile['interest_areas']}")

    def test_workflow_with_user_profile(self):
        """UserProfileManager를 포함한 워크플로우 통합 테스트"""
        print("\n=== 워크플로우 통합 테스트 ===")

        if not self.workflow.user_profile_manager:
            self.skipTest("UserProfileManager가 사용할 수 없습니다")

        import time
        user_id = f"test_workflow_{int(time.time())}"

        # 프로필 생성
        self.workflow.user_profile_manager.create_profile(user_id, {
            "expertise_level": "intermediate",
            "interest_areas": ["형법"]
        })

        # 개인화 노드 실행을 위한 상태 생성
        from source.services.langgraph_workflow.state_definitions import (
            LegalWorkflowState,
        )

        initial_state: LegalWorkflowState = {
            "query": "범죄와 처벌에 대해 알려주세요",
            "user_query": "범죄와 처벌에 대해 알려주세요",
            "context": None,
            "session_id": "test_session_001",
            "user_id": user_id,
            "input_validation": {},
            "question_classification": {},
            "domain_analysis": {},
            "retrieved_docs": [],
            "legal_analysis": {},
            "generated_response": "",
            "answer": "",
            "response": "",
            "quality_metrics": {},
            "workflow_steps": [],
            "processing_time": 0.0,
            "confidence_score": 0.0,
            "error_messages": [],
            "conversation_history": [],
            "user_preferences": {},
            "intermediate_results": {},
            "validation_results": {},
            "enriched_context": {},
            "agent_coordination": {},
            "synthesis_result": {},
            "quality_assurance_result": {},
            "research_agent_result": {},
            "analysis_agent_result": {},
            "review_agent_result": {},
            "performance_metrics": {},
            "memory_usage": {},
            "user_expertise_level": "beginner",
            "preferred_response_style": "formal",
            "device_info": {},
            "expertise_context": {},
            "interest_areas": [],
            "personalization_score": 0.0,
            "legal_domain": "",
            "statute_references": [],
            "precedent_references": [],
            "legal_confidence": 0.0,
            "phase1_context": {},
            "phase2_personalization": {},
            "phase3_memory_quality": {},
            "legal_restriction_result": {},
            "is_restricted": False,
            "is_law_article_query": False,
            "is_contract_query": False,
            "completion_result": {},
            "disclaimer_added": False,
            "query_analysis": {},
            "hybrid_classification": {},
            "generation_success": False,
            "generation_method": "",
            "current_step": "",
            "next_step": None,
            "workflow_completed": False,
            "requires_human_review": False,
            "cache_hits": [],
            "cache_misses": [],
            "optimization_applied": [],
            "trace_id": None,
            "span_id": None,
            "log_entries": [],
            "custom_fields": {},
            "plugin_results": {},
            "external_api_responses": {}
        }

        # 개인화 노드 실행
        personalized_state = self.workflow.personalize_response(initial_state)

        # 개인화 결과 확인
        self.assertIn("phase2_personalization", personalized_state)
        phase2_info = personalized_state["phase2_personalization"]

        # user_profile 정보 확인
        if phase2_info.get("user_profile"):
            user_profile = phase2_info["user_profile"]
            self.assertIn("expertise_level", user_profile)
            self.assertIn("interest_areas", user_profile)
            print(f"✅ 워크플로우 개인화 성공")
            print(f"   - 전문성 수준: {personalized_state.get('user_expertise_level')}")
            print(f"   - 관심 분야: {len(personalized_state.get('interest_areas', []))}개")
            print(f"   - 개인화 점수: {personalized_state.get('personalization_score', 0):.2f}")
        else:
            print("⚠️ UserProfile 정보가 비어있음 (기본 모드로 동작)")

    def test_user_statistics(self):
        """사용자 통계 조회 테스트"""
        print("\n=== 사용자 통계 조회 테스트 ===")

        if not self.workflow.user_profile_manager:
            self.skipTest("UserProfileManager가 사용할 수 없습니다")

        import time
        user_id = f"test_stats_{int(time.time())}"

        # 프로필 생성
        profile_data = {
            "expertise_level": "expert",
            "preferred_detail_level": "detailed",
            "interest_areas": ["민법", "형법", "상법"]
        }
        self.workflow.user_profile_manager.create_profile(user_id, profile_data)

        # 통계 조회
        stats = self.workflow.user_profile_manager.get_user_statistics(user_id)

        self.assertIsNotNone(stats)
        self.assertEqual(stats["user_id"], user_id)
        print(f"✅ 사용자 통계 조회 성공")
        print(f"   - 사용자 ID: {stats.get('user_id')}")
        print(f"   - 전문성 수준: {stats.get('expertise_level')}")
        print(f"   - 관심 분야: {stats.get('interest_areas', [])}")


def run_tests():
    """테스트 실행"""
    unittest.main()


if __name__ == "__main__":
    run_tests()
