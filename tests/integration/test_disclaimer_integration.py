#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
면책 조항 통합 테스트
UserPreferenceManager와 LangGraph 워크플로우의 통합 테스트
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from source.services.langgraph_workflow.state_definitions import LegalWorkflowState
from source.services.user_preference_manager import (
    DisclaimerPosition,
    DisclaimerStyle,
    UserPreferenceManager,
)


class TestDisclaimerIntegration(unittest.TestCase):
    """면책 조항 통합 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.preference_manager = UserPreferenceManager()

    def test_natural_disclaimer_style(self):
        """자연스러운 면책 조항 스타일 테스트"""
        # 설정 변경
        self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        self.preference_manager.preferences.disclaimer_position = DisclaimerPosition.END

        # 테스트 응답
        response = "계약서 검토 결과, 중요한 조항들에는 주의가 필요합니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증
        self.assertNotEqual(enhanced_response, response)
        self.assertIn("일반적인", enhanced_response.lower() or "참고", enhanced_response)
        print(f"✅ 자연스러운 스타일: {enhanced_response}")

    def test_formal_disclaimer_style(self):
        """공식적인 면책 조항 스타일 테스트"""
        # 설정 변경
        self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.FORMAL
        self.preference_manager.preferences.disclaimer_position = DisclaimerPosition.END

        # 테스트 응답
        response = "법령 해석 결과, 해당 조항은 다음과 같이 이해됩니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증
        self.assertNotEqual(enhanced_response, response)
        # 공식적 스타일의 면책 조항에 전문가 관련 키워드가 포함되어야 함
        self.assertTrue(
            "법률" in enhanced_response or "변호사" in enhanced_response or "참고용" in enhanced_response or "전문가" in enhanced_response,
            f"면책 조항에 법률 관련 키워드가 없습니다: {enhanced_response}"
        )
        print(f"✅ 공식적 스타일: {enhanced_response}")

    def test_no_disclaimer(self):
        """면책 조항 없음 테스트"""
        # 설정 변경
        self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.NONE

        # 테스트 응답
        response = "판례 분석 결과, 유사한 사례가 있습니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증 (면책 조항이 추가되지 않아야 함)
        self.assertEqual(enhanced_response, response)
        print(f"✅ 면책 조항 없음: {enhanced_response}")

    def test_disclaimer_disabled(self):
        """면책 조항 비활성화 테스트"""
        # 설정 변경
        self.preference_manager.preferences.show_disclaimer = False

        # 테스트 응답
        response = "법적 자문 결과를 정리해드리겠습니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증 (면책 조항이 추가되지 않아야 함)
        self.assertEqual(enhanced_response, response)
        print(f"✅ 면책 조항 비활성화: {enhanced_response}")

    def test_integrated_position(self):
        """통합 위치 테스트"""
        # 설정 변경
        self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        self.preference_manager.preferences.disclaimer_position = DisclaimerPosition.INTEGRATED

        # 테스트 응답
        response = "답변입니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증
        self.assertNotEqual(enhanced_response, response)
        print(f"✅ 통합 위치: {enhanced_response}")

    def test_none_position(self):
        """면책 조항 위치 없음 테스트"""
        # 설정 변경
        self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        self.preference_manager.preferences.disclaimer_position = DisclaimerPosition.NONE

        # 테스트 응답
        response = "법률 검색 결과입니다."

        # 면책 조항 추가
        enhanced_response = self.preference_manager.add_disclaimer_to_response(response)

        # 검증
        self.assertEqual(enhanced_response, response)
        print(f"✅ 위치 없음: {enhanced_response}")

    def test_random_disclaimer_text(self):
        """랜덤 면책 조항 텍스트 테스트"""
        # 여러 번 실행하여 다른 텍스트가 나오는지 확인
        disclaimers = []

        for _ in range(10):
            self.preference_manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
            response = "테스트 답변입니다."
            enhanced_response = self.preference_manager.add_disclaimer_to_response(response)
            disclaimers.append(enhanced_response)

        # 검증 (최소 2개 이상의 서로 다른 텍스트가 있어야 함)
        unique_disclaimers = set(disclaimers)
        print(f"✅ 랜덤 텍스트 생성: {len(unique_disclaimers)}개의 고유한 텍스트")
        self.assertGreaterEqual(len(unique_disclaimers), 2)


class TestLangGraphDisclaimerIntegration(unittest.TestCase):
    """LangGraph 면책 조항 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.test_state: LegalWorkflowState = {
            "user_query": "계약서 검토 요청",
            "response": "계약서 검토 결과입니다.",
            "answer": "",
            "generated_response": "",
            "disclaimer_added": False,
            "processing_steps": [],
            "processing_time": 0.0,
            "user_preferences": {},
            "workflow_steps": [],
            "confidence_score": 0.0,
            "error_messages": [],
            "query": "",
            "user_id": "test_user",
            "session_id": "test_session",
            "context": None
        }

    def test_default_user_preferences(self):
        """기본 사용자 설정 테스트"""
        state = self.test_state.copy()

        # 기본 설정이 없으면 모든 필드가 있어야 함
        self.assertIn("user_preferences", state)

        user_prefs = state["user_preferences"]
        self.assertIsInstance(user_prefs, dict)

    def test_custom_user_preferences_natural_style(self):
        """사용자 설정 - 자연스러운 스타일 테스트"""
        state = self.test_state.copy()
        state["user_preferences"] = {
            "show_disclaimer": True,
            "disclaimer_style": "natural",
            "disclaimer_position": "end",
            "preferred_tone": "friendly",
            "example_preference": True
        }

        # UserPreferenceManager 사용
        manager = UserPreferenceManager()

        # 설정 적용
        manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        manager.preferences.disclaimer_position = DisclaimerPosition.END
        manager.preferences.show_disclaimer = True

        # 면책 조항 추가
        enhanced_response = manager.add_disclaimer_to_response(
            state["response"],
            state["user_query"]
        )

        # 검증
        self.assertNotEqual(enhanced_response, state["response"])
        # 면책 조항에 전문가나 안내 관련 키워드가 포함되어야 함
        self.assertTrue(
            "전문가" in enhanced_response or "변호사" in enhanced_response or "안내" in enhanced_response,
            f"면책 조항에 전문가 관련 키워드가 없습니다: {enhanced_response}"
        )
        print(f"✅ 사용자 설정 적용: {enhanced_response}")

    def test_custom_user_preferences_formal_style(self):
        """사용자 설정 - 공식적인 스타일 테스트"""
        state = self.test_state.copy()
        state["user_preferences"] = {
            "show_disclaimer": True,
            "disclaimer_style": "formal",
            "disclaimer_position": "end"
        }

        # UserPreferenceManager 사용
        manager = UserPreferenceManager()

        # 설정 적용
        manager.preferences.disclaimer_style = DisclaimerStyle.FORMAL
        manager.preferences.disclaimer_position = DisclaimerPosition.END
        manager.preferences.show_disclaimer = True

        # 면책 조항 추가
        enhanced_response = manager.add_disclaimer_to_response(
            state["response"],
            state["user_query"]
        )

        # 검증
        self.assertNotEqual(enhanced_response, state["response"])
        print(f"✅ 공식적 스타일 적용: {enhanced_response}")

    def test_custom_user_preferences_no_disclaimer(self):
        """사용자 설정 - 면책 조항 없음 테스트"""
        state = self.test_state.copy()
        state["user_preferences"] = {
            "show_disclaimer": False
        }

        # UserPreferenceManager 사용
        manager = UserPreferenceManager()

        # 설정 적용
        manager.preferences.show_disclaimer = False

        # 면책 조항 추가
        enhanced_response = manager.add_disclaimer_to_response(
            state["response"],
            state["user_query"]
        )

        # 검증
        self.assertEqual(enhanced_response, state["response"])
        print(f"✅ 면책 조항 제외: {enhanced_response}")

    def test_disclaimer_with_existing_punctuation(self):
        """기존 문장 부호가 있는 경우 테스트"""
        state = self.test_state.copy()
        state["response"] = "답변입니다!"

        manager = UserPreferenceManager()
        manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        manager.preferences.disclaimer_position = DisclaimerPosition.END

        enhanced_response = manager.add_disclaimer_to_response(
            state["response"],
            state["user_query"]
        )

        # 검증
        self.assertNotEqual(enhanced_response, state["response"])
        print(f"✅ 기존 문장 부호 포함: {enhanced_response}")

    def test_disclaimer_without_existing_punctuation(self):
        """문장 부호가 없는 경우 테스트"""
        state = self.test_state.copy()
        state["response"] = "답변입니다"

        manager = UserPreferenceManager()
        manager.preferences.disclaimer_style = DisclaimerStyle.NATURAL
        manager.preferences.disclaimer_position = DisclaimerPosition.END

        enhanced_response = manager.add_disclaimer_to_response(
            state["response"],
            state["user_query"]
        )

        # 검증
        self.assertNotEqual(enhanced_response, state["response"])
        print(f"✅ 문장 부호 없음: {enhanced_response}")


if __name__ == "__main__":
    unittest.main()
