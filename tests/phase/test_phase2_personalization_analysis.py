# -*- coding: utf-8 -*-
"""
Phase 2: 개인화 및 지능형 분석 기능 테스트 스크립트
- UserProfileManager
- EmotionIntentAnalyzer
- ConversationFlowTracker
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

from core.data.conversation_store import ConversationStore
from source.services.conversation_flow_tracker import ConversationFlowTracker
from lawfirm_langgraph.langgraph_core.services.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationTurn,
)
from lawfirm_langgraph.langgraph_core.services.emotion_intent_analyzer import (
    EmotionIntentAnalyzer,
    EmotionType,
    IntentType,
    UrgencyLevel,
)
from source.services.user_profile_manager import (
    DetailLevel,
    ExpertiseLevel,
    UserProfileManager,
)


class TestPhase2PersonalizationAndAnalysis(unittest.TestCase):

    def setUp(self):
        self.test_db_path = "test_conversations_phase2.db"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.conversation_store = ConversationStore(self.test_db_path)
        self.user_profile_manager = UserProfileManager(self.conversation_store)
        self.emotion_intent_analyzer = EmotionIntentAnalyzer()
        self.conversation_flow_tracker = ConversationFlowTracker()

        self.user_id = "test_user_phase2"
        self.session_id = "test_session_phase2"

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_user_profile_manager_create_and_get(self):
        print("\n--- Test UserProfileManager: Create and Get ---")

        # 프로필 생성
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "interest_areas": ["민법", "형법"],
            "preferences": {"response_style": "professional"}
        }

        success = self.user_profile_manager.create_profile(self.user_id, profile_data)
        self.assertTrue(success)

        # 프로필 조회
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["expertise_level"], "intermediate")
        self.assertEqual(profile["preferred_detail_level"], "detailed")
        self.assertIn("민법", profile["interest_areas"])
        self.assertIn("형법", profile["interest_areas"])

    def test_user_profile_manager_expertise_tracking(self):
        print("\n--- Test UserProfileManager: Expertise Tracking ---")

        # 프로필 생성
        self.user_profile_manager.create_profile(self.user_id, {})

        # 질문 이력 시뮬레이션
        question_history = [
            {"user_query": "민법 제750조의 손해배상 요건에 대해 자세히 설명해주세요"},
            {"user_query": "대법원 판례에서 과실비율을 어떻게 정하는지 알고 싶습니다"},
            {"user_query": "법리적 해석과 실무적 적용의 차이점은 무엇인가요?"}
        ]

        estimated_level = self.user_profile_manager.track_expertise_level(self.user_id, question_history)
        self.assertIn(estimated_level, ["beginner", "intermediate", "advanced", "expert"])

        # 프로필 업데이트 확인
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIn("estimated_expertise_level", profile["preferences"])

    def test_user_profile_manager_personalized_context(self):
        print("\n--- Test UserProfileManager: Personalized Context ---")

        # 프로필 생성
        profile_data = {
            "expertise_level": "advanced",
            "preferred_detail_level": "detailed",
            "interest_areas": ["민법", "상법"]
        }
        self.user_profile_manager.create_profile(self.user_id, profile_data)

        # 개인화된 컨텍스트 조회
        query = "손해배상 청구 방법을 알려주세요"
        personalized_context = self.user_profile_manager.get_personalized_context(self.user_id, query)

        self.assertIsNotNone(personalized_context)
        self.assertEqual(personalized_context["user_id"], self.user_id)
        self.assertEqual(personalized_context["expertise_level"], "advanced")
        self.assertIn("민법", personalized_context["interest_areas"])
        self.assertGreater(personalized_context["personalization_score"], 0)

    def test_user_profile_manager_interest_areas_update(self):
        print("\n--- Test UserProfileManager: Interest Areas Update ---")

        # 프로필 생성
        self.user_profile_manager.create_profile(self.user_id, {})

        # 관심 분야 업데이트
        query = "근로기준법에 따른 퇴직금 계산 방법"
        success = self.user_profile_manager.update_interest_areas(self.user_id, query)
        self.assertTrue(success)

        # 업데이트 확인
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIn("근로기준법", profile["interest_areas"])

    def test_user_profile_manager_statistics(self):
        print("\n--- Test UserProfileManager: User Statistics ---")

        # 프로필 생성
        self.user_profile_manager.create_profile(self.user_id, {
            "expertise_level": "intermediate",
            "interest_areas": ["민법"]
        })

        # 통계 조회
        stats = self.user_profile_manager.get_user_statistics(self.user_id)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["user_id"], self.user_id)
        self.assertEqual(stats["expertise_level"], "intermediate")
        self.assertIn("민법", stats["interest_areas"])

    def test_emotion_intent_analyzer_emotion_analysis(self):
        print("\n--- Test EmotionIntentAnalyzer: Emotion Analysis ---")

        # 긍정적 감정 테스트
        positive_text = "감사합니다. 정말 도움이 되었어요!"
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(positive_text)
        self.assertIsNotNone(emotion_result)
        self.assertGreater(emotion_result.confidence, 0)
        self.assertGreater(emotion_result.intensity, 0)
        self.assertIsNotNone(emotion_result.reasoning)

        # 긴급한 감정 테스트
        urgent_text = "급해요! 오늘까지 답변해주세요!"
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(urgent_text)
        self.assertIsNotNone(emotion_result)
        self.assertGreater(emotion_result.confidence, 0)

    def test_emotion_intent_analyzer_intent_analysis(self):
        print("\n--- Test EmotionIntentAnalyzer: Intent Analysis ---")

        # 질문 의도 테스트
        question_text = "손해배상 청구 방법을 알려주세요"
        intent_result = self.emotion_intent_analyzer.analyze_intent(question_text)
        self.assertIsNotNone(intent_result)
        self.assertGreater(intent_result.confidence, 0)
        self.assertIsNotNone(intent_result.reasoning)

        # 감사 의도 테스트
        thanks_text = "감사합니다. 정말 도움이 되었어요"
        intent_result = self.emotion_intent_analyzer.analyze_intent(thanks_text)
        self.assertIsNotNone(intent_result)
        self.assertGreater(intent_result.confidence, 0)

    def test_emotion_intent_analyzer_urgency_assessment(self):
        print("\n--- Test EmotionIntentAnalyzer: Urgency Assessment ---")

        # 긴급한 텍스트
        urgent_text = "지금당장 답변해주세요! 긴급합니다!"
        urgency = self.emotion_intent_analyzer.assess_urgency(urgent_text, {})
        self.assertIn(urgency, [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH])

        # 일반적인 텍스트
        normal_text = "천천히 답변해주세요"
        urgency = self.emotion_intent_analyzer.assess_urgency(normal_text, {})
        self.assertIn(urgency, [UrgencyLevel.LOW, UrgencyLevel.MEDIUM])

    def test_emotion_intent_analyzer_response_tone(self):
        print("\n--- Test EmotionIntentAnalyzer: Response Tone ---")

        # 감정 및 의도 분석
        emotion_result = self.emotion_intent_analyzer.analyze_emotion("걱정돼요. 어떻게 해야 할까요?")
        intent_result = self.emotion_intent_analyzer.analyze_intent("걱정돼요. 어떻게 해야 할까요?")

        # 응답 톤 결정
        response_tone = self.emotion_intent_analyzer.get_contextual_response_tone(
            emotion_result, intent_result
        )

        self.assertIsNotNone(response_tone)
        self.assertIsNotNone(response_tone.tone_type)
        self.assertGreaterEqual(response_tone.empathy_level, 0)
        self.assertLessEqual(response_tone.empathy_level, 1)
        self.assertGreaterEqual(response_tone.formality_level, 0)
        self.assertLessEqual(response_tone.formality_level, 1)

    def test_conversation_flow_tracker_flow_tracking(self):
        print("\n--- Test ConversationFlowTracker: Flow Tracking ---")

        # 테스트 턴 생성
        turn = ConversationTurn(
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따라...",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # 흐름 추적
        self.conversation_flow_tracker.track_conversation_flow(self.session_id, turn)

        # 패턴 업데이트 확인
        self.assertGreater(len(self.conversation_flow_tracker.flow_patterns), 0)

    def test_conversation_flow_tracker_intent_prediction(self):
        print("\n--- Test ConversationFlowTracker: Intent Prediction ---")

        # 테스트 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="손해배상 청구 방법을 알려주세요",
                    bot_response="민법 제750조에 따라...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="더 자세히 설명해주세요",
                    bot_response="구체적으로 말씀드리면...",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["손해배상"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 의도 예측
        predicted_intents = self.conversation_flow_tracker.predict_next_intent(context)
        self.assertIsNotNone(predicted_intents)
        self.assertIsInstance(predicted_intents, list)
        self.assertGreater(len(predicted_intents), 0)

    def test_conversation_flow_tracker_follow_up_suggestions(self):
        print("\n--- Test ConversationFlowTracker: Follow-up Suggestions ---")

        # 테스트 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="계약서 검토를 요청합니다",
                    bot_response="계약서를 검토해드리겠습니다...",
                    timestamp=datetime.now(),
                    question_type="contract_review"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": {"계약서"}},
            topic_stack=["계약서"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 후속 질문 제안
        suggestions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    def test_conversation_flow_tracker_branch_detection(self):
        print("\n--- Test ConversationFlowTracker: Branch Detection ---")

        # 분기점 감지 테스트
        branch = self.conversation_flow_tracker.detect_conversation_branch("더 자세히 설명해주세요")
        self.assertIsNotNone(branch)
        self.assertEqual(branch.branch_type, "detailed_explanation")
        self.assertGreater(len(branch.follow_up_suggestions), 0)

        # 분기점이 없는 경우
        branch = self.conversation_flow_tracker.detect_conversation_branch("안녕하세요")
        self.assertIsNone(branch)

    def test_conversation_flow_tracker_conversation_state(self):
        print("\n--- Test ConversationFlowTracker: Conversation State ---")

        # 초기 상태 테스트
        context = ConversationContext(
            session_id=self.session_id,
            turns=[],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
            topic_stack=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        state = self.conversation_flow_tracker.get_conversation_state(context)
        self.assertEqual(state, "initial")

        # 턴 추가 후 상태 확인
        context.turns.append(ConversationTurn(
            user_query="질문이 있습니다",
            bot_response="답변입니다",
            timestamp=datetime.now(),
            question_type="general_question"
        ))

        state = self.conversation_flow_tracker.get_conversation_state(context)
        self.assertIsNotNone(state)

    def test_conversation_flow_tracker_pattern_analysis(self):
        print("\n--- Test ConversationFlowTracker: Pattern Analysis ---")

        # 테스트 세션들 생성
        sessions = []
        for i in range(3):
            context = ConversationContext(
                session_id=f"test_session_{i}",
                turns=[
                    ConversationTurn(
                        user_query=f"질문 {i}",
                        bot_response=f"답변 {i}",
                        timestamp=datetime.now(),
                        question_type="law_inquiry"
                    ),
                    ConversationTurn(
                        user_query=f"추가 질문 {i}",
                        bot_response=f"추가 답변 {i}",
                        timestamp=datetime.now(),
                        question_type="follow_up"
                    )
                ],
                entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": set()},
                topic_stack=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            sessions.append(context)

        # 패턴 분석
        pattern_analysis = self.conversation_flow_tracker.analyze_flow_patterns(sessions)
        self.assertIsNotNone(pattern_analysis)
        self.assertIn("total_sessions", pattern_analysis)
        self.assertEqual(pattern_analysis["total_sessions"], 3)

    def test_integrated_phase2_components(self):
        print("\n--- Test Integrated Phase 2 Components ---")

        # 사용자 프로필 생성
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "interest_areas": ["민법"]
        }
        self.user_profile_manager.create_profile(self.user_id, profile_data)

        # 대화 컨텍스트 생성
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="손해배상 청구 방법을 알려주세요",
                    bot_response="민법 제750조에 따라...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
            topic_stack=["손해배상"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 통합 테스트: 개인화된 컨텍스트 + 감정/의도 분석 + 흐름 추적
        query = "더 자세히 설명해주세요"

        # 1. 개인화된 컨텍스트
        personalized_context = self.user_profile_manager.get_personalized_context(self.user_id, query)
        self.assertIsNotNone(personalized_context)

        # 2. 감정/의도 분석
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(query)
        intent_result = self.emotion_intent_analyzer.analyze_intent(query, context)
        self.assertIsNotNone(emotion_result)
        self.assertIsNotNone(intent_result)

        # 3. 응답 톤 결정
        response_tone = self.emotion_intent_analyzer.get_contextual_response_tone(
            emotion_result, intent_result, personalized_context
        )
        self.assertIsNotNone(response_tone)

        # 4. 흐름 추적
        turn = ConversationTurn(
            user_query=query,
            bot_response="자세한 설명입니다",
            timestamp=datetime.now(),
            question_type="clarification"
        )
        self.conversation_flow_tracker.track_conversation_flow(self.session_id, turn)

        # 5. 후속 질문 제안
        suggestions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)

        # 통합 결과 검증
        self.assertEqual(personalized_context["user_id"], self.user_id)
        self.assertGreater(emotion_result.confidence, 0)
        self.assertGreater(intent_result.confidence, 0)
        self.assertIsNotNone(response_tone.tone_type)
        self.assertGreater(len(suggestions), 0)

        print(f"   통합 테스트 완료:")
        print(f"   - 개인화 점수: {personalized_context['personalization_score']:.2f}")
        print(f"   - 감정: {emotion_result.primary_emotion.value}")
        print(f"   - 의도: {intent_result.primary_intent.value}")
        print(f"   - 응답 톤: {response_tone.tone_type}")
        print(f"   - 제안된 질문 수: {len(suggestions)}")


if __name__ == '__main__':
    unittest.main()
