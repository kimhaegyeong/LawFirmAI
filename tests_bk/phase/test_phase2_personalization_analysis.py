# -*- coding: utf-8 -*-
"""
Phase 2: ê°œì¸??ë°?ì§€?¥í˜• ë¶„ì„ ê¸°ëŠ¥ ?ŒìŠ¤???¤í¬ë¦½íŠ¸
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.conversation_store import ConversationStore
from source.services.conversation_flow_tracker import ConversationFlowTracker
from source.services.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationTurn,
)
from source.services.emotion_intent_analyzer import (
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

        # ?„ë¡œ???ì„±
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "interest_areas": ["ë¯¼ë²•", "?•ë²•"],
            "preferences": {"response_style": "professional"}
        }

        success = self.user_profile_manager.create_profile(self.user_id, profile_data)
        self.assertTrue(success)

        # ?„ë¡œ??ì¡°íšŒ
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["expertise_level"], "intermediate")
        self.assertEqual(profile["preferred_detail_level"], "detailed")
        self.assertIn("ë¯¼ë²•", profile["interest_areas"])
        self.assertIn("?•ë²•", profile["interest_areas"])

    def test_user_profile_manager_expertise_tracking(self):
        print("\n--- Test UserProfileManager: Expertise Tracking ---")

        # ?„ë¡œ???ì„±
        self.user_profile_manager.create_profile(self.user_id, {})

        # ì§ˆë¬¸ ?´ë ¥ ?œë??ˆì´??
        question_history = [
            {"user_query": "ë¯¼ë²• ??50ì¡°ì˜ ?í•´ë°°ìƒ ?”ê±´???€???ì„¸???¤ëª…?´ì£¼?¸ìš”"},
            {"user_query": "?€ë²•ì› ?ë??ì„œ ê³¼ì‹¤ë¹„ìœ¨???´ë–»ê²??•í•˜?”ì? ?Œê³  ?¶ìŠµ?ˆë‹¤"},
            {"user_query": "ë²•ë¦¬???´ì„ê³??¤ë¬´???ìš©??ì°¨ì´?ì? ë¬´ì—‡?¸ê???"}
        ]

        estimated_level = self.user_profile_manager.track_expertise_level(self.user_id, question_history)
        self.assertIn(estimated_level, ["beginner", "intermediate", "advanced", "expert"])

        # ?„ë¡œ???…ë°?´íŠ¸ ?•ì¸
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIn("estimated_expertise_level", profile["preferences"])

    def test_user_profile_manager_personalized_context(self):
        print("\n--- Test UserProfileManager: Personalized Context ---")

        # ?„ë¡œ???ì„±
        profile_data = {
            "expertise_level": "advanced",
            "preferred_detail_level": "detailed",
            "interest_areas": ["ë¯¼ë²•", "?ë²•"]
        }
        self.user_profile_manager.create_profile(self.user_id, profile_data)

        # ê°œì¸?”ëœ ì»¨í…?¤íŠ¸ ì¡°íšŒ
        query = "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??
        personalized_context = self.user_profile_manager.get_personalized_context(self.user_id, query)

        self.assertIsNotNone(personalized_context)
        self.assertEqual(personalized_context["user_id"], self.user_id)
        self.assertEqual(personalized_context["expertise_level"], "advanced")
        self.assertIn("ë¯¼ë²•", personalized_context["interest_areas"])
        self.assertGreater(personalized_context["personalization_score"], 0)

    def test_user_profile_manager_interest_areas_update(self):
        print("\n--- Test UserProfileManager: Interest Areas Update ---")

        # ?„ë¡œ???ì„±
        self.user_profile_manager.create_profile(self.user_id, {})

        # ê´€??ë¶„ì•¼ ?…ë°?´íŠ¸
        query = "ê·¼ë¡œê¸°ì?ë²•ì— ?°ë¥¸ ?´ì§ê¸?ê³„ì‚° ë°©ë²•"
        success = self.user_profile_manager.update_interest_areas(self.user_id, query)
        self.assertTrue(success)

        # ?…ë°?´íŠ¸ ?•ì¸
        profile = self.user_profile_manager.get_profile(self.user_id)
        self.assertIn("ê·¼ë¡œê¸°ì?ë²?, profile["interest_areas"])

    def test_user_profile_manager_statistics(self):
        print("\n--- Test UserProfileManager: User Statistics ---")

        # ?„ë¡œ???ì„±
        self.user_profile_manager.create_profile(self.user_id, {
            "expertise_level": "intermediate",
            "interest_areas": ["ë¯¼ë²•"]
        })

        # ?µê³„ ì¡°íšŒ
        stats = self.user_profile_manager.get_user_statistics(self.user_id)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["user_id"], self.user_id)
        self.assertEqual(stats["expertise_level"], "intermediate")
        self.assertIn("ë¯¼ë²•", stats["interest_areas"])

    def test_emotion_intent_analyzer_emotion_analysis(self):
        print("\n--- Test EmotionIntentAnalyzer: Emotion Analysis ---")

        # ê¸ì •??ê°ì • ?ŒìŠ¤??
        positive_text = "ê°ì‚¬?©ë‹ˆ?? ?•ë§ ?„ì????˜ì—ˆ?´ìš”!"
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(positive_text)
        self.assertIsNotNone(emotion_result)
        self.assertGreater(emotion_result.confidence, 0)
        self.assertGreater(emotion_result.intensity, 0)
        self.assertIsNotNone(emotion_result.reasoning)

        # ê¸´ê¸‰??ê°ì • ?ŒìŠ¤??
        urgent_text = "ê¸‰í•´?? ?¤ëŠ˜ê¹Œì? ?µë??´ì£¼?¸ìš”!"
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(urgent_text)
        self.assertIsNotNone(emotion_result)
        self.assertGreater(emotion_result.confidence, 0)

    def test_emotion_intent_analyzer_intent_analysis(self):
        print("\n--- Test EmotionIntentAnalyzer: Intent Analysis ---")

        # ì§ˆë¬¸ ?˜ë„ ?ŒìŠ¤??
        question_text = "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??
        intent_result = self.emotion_intent_analyzer.analyze_intent(question_text)
        self.assertIsNotNone(intent_result)
        self.assertGreater(intent_result.confidence, 0)
        self.assertIsNotNone(intent_result.reasoning)

        # ê°ì‚¬ ?˜ë„ ?ŒìŠ¤??
        thanks_text = "ê°ì‚¬?©ë‹ˆ?? ?•ë§ ?„ì????˜ì—ˆ?´ìš”"
        intent_result = self.emotion_intent_analyzer.analyze_intent(thanks_text)
        self.assertIsNotNone(intent_result)
        self.assertGreater(intent_result.confidence, 0)

    def test_emotion_intent_analyzer_urgency_assessment(self):
        print("\n--- Test EmotionIntentAnalyzer: Urgency Assessment ---")

        # ê¸´ê¸‰???ìŠ¤??
        urgent_text = "ì§€ê¸ˆë‹¹???µë??´ì£¼?¸ìš”! ê¸´ê¸‰?©ë‹ˆ??"
        urgency = self.emotion_intent_analyzer.assess_urgency(urgent_text, {})
        self.assertIn(urgency, [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH])

        # ?¼ë°˜?ì¸ ?ìŠ¤??
        normal_text = "ì²œì²œ???µë??´ì£¼?¸ìš”"
        urgency = self.emotion_intent_analyzer.assess_urgency(normal_text, {})
        self.assertIn(urgency, [UrgencyLevel.LOW, UrgencyLevel.MEDIUM])

    def test_emotion_intent_analyzer_response_tone(self):
        print("\n--- Test EmotionIntentAnalyzer: Response Tone ---")

        # ê°ì • ë°??˜ë„ ë¶„ì„
        emotion_result = self.emotion_intent_analyzer.analyze_emotion("ê±±ì •?¼ìš”. ?´ë–»ê²??´ì•¼ ? ê¹Œ??")
        intent_result = self.emotion_intent_analyzer.analyze_intent("ê±±ì •?¼ìš”. ?´ë–»ê²??´ì•¼ ? ê¹Œ??")

        # ?‘ë‹µ ??ê²°ì •
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

        # ?ŒìŠ¤?????ì„±
        turn = ConversationTurn(
            user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¼...",
            timestamp=datetime.now(),
            question_type="law_inquiry"
        )

        # ?ë¦„ ì¶”ì 
        self.conversation_flow_tracker.track_conversation_flow(self.session_id, turn)

        # ?¨í„´ ?…ë°?´íŠ¸ ?•ì¸
        self.assertGreater(len(self.conversation_flow_tracker.flow_patterns), 0)

    def test_conversation_flow_tracker_intent_prediction(self):
        print("\n--- Test ConversationFlowTracker: Intent Prediction ---")

        # ?ŒìŠ¤??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                    bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¼...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                ),
                ConversationTurn(
                    user_query="???ì„¸???¤ëª…?´ì£¼?¸ìš”",
                    bot_response="êµ¬ì²´?ìœ¼ë¡?ë§ì??œë¦¬ë©?..",
                    timestamp=datetime.now(),
                    question_type="clarification"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["?í•´ë°°ìƒ"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?˜ë„ ?ˆì¸¡
        predicted_intents = self.conversation_flow_tracker.predict_next_intent(context)
        self.assertIsNotNone(predicted_intents)
        self.assertIsInstance(predicted_intents, list)
        self.assertGreater(len(predicted_intents), 0)

    def test_conversation_flow_tracker_follow_up_suggestions(self):
        print("\n--- Test ConversationFlowTracker: Follow-up Suggestions ---")

        # ?ŒìŠ¤??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="ê³„ì•½??ê²€? ë? ?”ì²­?©ë‹ˆ??,
                    bot_response="ê³„ì•½?œë? ê²€? í•´?œë¦¬ê² ìŠµ?ˆë‹¤...",
                    timestamp=datetime.now(),
                    question_type="contract_review"
                )
            ],
            entities={"laws": set(), "articles": set(), "precedents": set(), "legal_terms": {"ê³„ì•½??}},
            topic_stack=["ê³„ì•½??],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?„ì† ì§ˆë¬¸ ?œì•ˆ
        suggestions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    def test_conversation_flow_tracker_branch_detection(self):
        print("\n--- Test ConversationFlowTracker: Branch Detection ---")

        # ë¶„ê¸°??ê°ì? ?ŒìŠ¤??
        branch = self.conversation_flow_tracker.detect_conversation_branch("???ì„¸???¤ëª…?´ì£¼?¸ìš”")
        self.assertIsNotNone(branch)
        self.assertEqual(branch.branch_type, "detailed_explanation")
        self.assertGreater(len(branch.follow_up_suggestions), 0)

        # ë¶„ê¸°?ì´ ?†ëŠ” ê²½ìš°
        branch = self.conversation_flow_tracker.detect_conversation_branch("?ˆë…•?˜ì„¸??)
        self.assertIsNone(branch)

    def test_conversation_flow_tracker_conversation_state(self):
        print("\n--- Test ConversationFlowTracker: Conversation State ---")

        # ì´ˆê¸° ?íƒœ ?ŒìŠ¤??
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

        # ??ì¶”ê? ???íƒœ ?•ì¸
        context.turns.append(ConversationTurn(
            user_query="ì§ˆë¬¸???ˆìŠµ?ˆë‹¤",
            bot_response="?µë??…ë‹ˆ??,
            timestamp=datetime.now(),
            question_type="general_question"
        ))

        state = self.conversation_flow_tracker.get_conversation_state(context)
        self.assertIsNotNone(state)

    def test_conversation_flow_tracker_pattern_analysis(self):
        print("\n--- Test ConversationFlowTracker: Pattern Analysis ---")

        # ?ŒìŠ¤???¸ì…˜???ì„±
        sessions = []
        for i in range(3):
            context = ConversationContext(
                session_id=f"test_session_{i}",
                turns=[
                    ConversationTurn(
                        user_query=f"ì§ˆë¬¸ {i}",
                        bot_response=f"?µë? {i}",
                        timestamp=datetime.now(),
                        question_type="law_inquiry"
                    ),
                    ConversationTurn(
                        user_query=f"ì¶”ê? ì§ˆë¬¸ {i}",
                        bot_response=f"ì¶”ê? ?µë? {i}",
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

        # ?¨í„´ ë¶„ì„
        pattern_analysis = self.conversation_flow_tracker.analyze_flow_patterns(sessions)
        self.assertIsNotNone(pattern_analysis)
        self.assertIn("total_sessions", pattern_analysis)
        self.assertEqual(pattern_analysis["total_sessions"], 3)

    def test_integrated_phase2_components(self):
        print("\n--- Test Integrated Phase 2 Components ---")

        # ?¬ìš©???„ë¡œ???ì„±
        profile_data = {
            "expertise_level": "intermediate",
            "preferred_detail_level": "detailed",
            "interest_areas": ["ë¯¼ë²•"]
        }
        self.user_profile_manager.create_profile(self.user_id, profile_data)

        # ?€??ì»¨í…?¤íŠ¸ ?ì„±
        context = ConversationContext(
            session_id=self.session_id,
            turns=[
                ConversationTurn(
                    user_query="?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                    bot_response="ë¯¼ë²• ??50ì¡°ì— ?°ë¼...",
                    timestamp=datetime.now(),
                    question_type="law_inquiry"
                )
            ],
            entities={"laws": {"ë¯¼ë²•"}, "articles": {"??50ì¡?}, "precedents": set(), "legal_terms": {"?í•´ë°°ìƒ"}},
            topic_stack=["?í•´ë°°ìƒ"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # ?µí•© ?ŒìŠ¤?? ê°œì¸?”ëœ ì»¨í…?¤íŠ¸ + ê°ì •/?˜ë„ ë¶„ì„ + ?ë¦„ ì¶”ì 
        query = "???ì„¸???¤ëª…?´ì£¼?¸ìš”"

        # 1. ê°œì¸?”ëœ ì»¨í…?¤íŠ¸
        personalized_context = self.user_profile_manager.get_personalized_context(self.user_id, query)
        self.assertIsNotNone(personalized_context)

        # 2. ê°ì •/?˜ë„ ë¶„ì„
        emotion_result = self.emotion_intent_analyzer.analyze_emotion(query)
        intent_result = self.emotion_intent_analyzer.analyze_intent(query, context)
        self.assertIsNotNone(emotion_result)
        self.assertIsNotNone(intent_result)

        # 3. ?‘ë‹µ ??ê²°ì •
        response_tone = self.emotion_intent_analyzer.get_contextual_response_tone(
            emotion_result, intent_result, personalized_context
        )
        self.assertIsNotNone(response_tone)

        # 4. ?ë¦„ ì¶”ì 
        turn = ConversationTurn(
            user_query=query,
            bot_response="?ì„¸???¤ëª…?…ë‹ˆ??,
            timestamp=datetime.now(),
            question_type="clarification"
        )
        self.conversation_flow_tracker.track_conversation_flow(self.session_id, turn)

        # 5. ?„ì† ì§ˆë¬¸ ?œì•ˆ
        suggestions = self.conversation_flow_tracker.suggest_follow_up_questions(context)
        self.assertIsNotNone(suggestions)

        # ?µí•© ê²°ê³¼ ê²€ì¦?
        self.assertEqual(personalized_context["user_id"], self.user_id)
        self.assertGreater(emotion_result.confidence, 0)
        self.assertGreater(intent_result.confidence, 0)
        self.assertIsNotNone(response_tone.tone_type)
        self.assertGreater(len(suggestions), 0)

        print(f"   ?µí•© ?ŒìŠ¤???„ë£Œ:")
        print(f"   - ê°œì¸???ìˆ˜: {personalized_context['personalization_score']:.2f}")
        print(f"   - ê°ì •: {emotion_result.primary_emotion.value}")
        print(f"   - ?˜ë„: {intent_result.primary_intent.value}")
        print(f"   - ?‘ë‹µ ?? {response_tone.tone_type}")
        print(f"   - ?œì•ˆ??ì§ˆë¬¸ ?? {len(suggestions)}")


if __name__ == '__main__':
    unittest.main()
