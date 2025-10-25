# -*- coding: utf-8 -*-
"""
자연스러운 답변 개선 시스템 테스트
구현된 모든 컴포넌트들의 기능을 테스트하는 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.conversation_connector import ConversationConnector
from source.services.emotional_tone_adjuster import EmotionalToneAdjuster
from source.services.personalized_style_learner import PersonalizedStyleLearner
from source.services.realtime_feedback_system import RealtimeFeedbackSystem
from source.services.naturalness_evaluator import NaturalnessEvaluator


def test_conversation_connector():
    """대화 연결어 시스템 테스트"""
    print("=== 대화 연결어 시스템 테스트 ===")
    
    connector = ConversationConnector()
    
    # 테스트 케이스들
    test_cases = [
        {
            "answer": "계약서 작성 시 주의사항을 설명드리겠습니다.",
            "context": {
                "previous_topic": "계약 관련 질문",
                "conversation_flow": "follow_up",
                "user_emotion": "neutral",
                "question_type": "contract"
            }
        },
        {
            "answer": "손해배상 청구 요건에 대해 답변드리겠습니다.",
            "context": {
                "previous_topic": "",
                "conversation_flow": "new",
                "user_emotion": "urgent",
                "question_type": "civil_law"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"원본 답변: {test_case['answer']}")
        
        improved_answer = connector.add_natural_connectors(
            test_case['answer'], test_case['context']
        )
        
        print(f"개선된 답변: {improved_answer}")
        print(f"맥락: {test_case['context']}")


def test_emotional_tone_adjuster():
    """감정 톤 조절기 테스트"""
    print("\n=== 감정 톤 조절기 테스트 ===")
    
    adjuster = EmotionalToneAdjuster()
    
    # 테스트 케이스들
    test_cases = [
        {
            "answer": "계약 해지 조건에 대해 설명드리겠습니다.",
            "emotion": "urgent",
            "intensity": 0.8
        },
        {
            "answer": "이혼 절차에 대해 안내드리겠습니다.",
            "emotion": "anxious",
            "intensity": 0.7
        },
        {
            "answer": "법률 상담에 대해 답변드리겠습니다.",
            "emotion": "neutral",
            "intensity": 0.5
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"원본 답변: {test_case['answer']}")
        print(f"감정: {test_case['emotion']}, 강도: {test_case['intensity']}")
        
        adjusted_answer = adjuster.adjust_tone(
            test_case['answer'], 
            test_case['emotion'], 
            test_case['intensity']
        )
        
        print(f"조절된 답변: {adjusted_answer}")


def test_personalized_style_learner():
    """개인화된 스타일 학습 시스템 테스트"""
    print("\n=== 개인화된 스타일 학습 시스템 테스트 ===")
    
    learner = PersonalizedStyleLearner()
    
    # 테스트 케이스들
    test_cases = [
        {
            "answer": "계약서 검토 시 주의사항을 설명드리겠습니다.",
            "user_id": "test_user_1",
            "interaction_data": {
                "question": "계약서 검토 방법을 알려주세요",
                "answer": "계약서 검토 시 주의사항을 설명드리겠습니다.",
                "satisfaction_score": 0.8,
                "interaction_time": 2.5,
                "question_length": 15
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"원본 답변: {test_case['answer']}")
        
        # 스타일 학습
        preferences = learner.learn_user_preferences(
            test_case['user_id'], 
            test_case['interaction_data']
        )
        
        print(f"학습된 선호도: {preferences}")
        
        # 개인화된 스타일 적용
        personalized_answer = learner.apply_personalized_style(
            test_case['answer'], 
            test_case['user_id']
        )
        
        print(f"개인화된 답변: {personalized_answer}")


def test_realtime_feedback_system():
    """실시간 피드백 시스템 테스트"""
    print("\n=== 실시간 피드백 시스템 테스트 ===")
    
    feedback_system = RealtimeFeedbackSystem()
    
    # 테스트 케이스들
    test_cases = [
        {
            "session_id": "test_session_1",
            "feedback_data": {
                "response_id": "response_1",
                "type": "explicit",
                "satisfaction": 0.3,
                "accuracy": 0.4,
                "speed": 0.6,
                "usability": 0.5,
                "comments": "답변이 너무 길고 복잡해요",
                "issue_types": ["답변이 부정확함", "인터페이스가 복잡함"]
            }
        },
        {
            "session_id": "test_session_2",
            "feedback_data": {
                "response_id": "response_2",
                "type": "explicit",
                "satisfaction": 0.9,
                "accuracy": 0.8,
                "speed": 0.7,
                "usability": 0.9,
                "comments": "정말 도움이 되었어요!",
                "issue_types": []
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"피드백 데이터: {test_case['feedback_data']}")
        
        result = feedback_system.collect_feedback(
            test_case['session_id'], 
            test_case['feedback_data']
        )
        
        print(f"처리 결과: {result}")
        
        # 다음 응답 개선사항 확인
        improvements = feedback_system.get_next_response_improvements()
        print(f"다음 응답 개선사항: {improvements}")


def test_naturalness_evaluator():
    """자연스러움 평가 시스템 테스트"""
    print("\n=== 자연스러움 평가 시스템 테스트 ===")
    
    evaluator = NaturalnessEvaluator()
    
    # 테스트 케이스들
    test_cases = [
        {
            "answer": "계약서 작성 시 주의사항을 설명드리겠습니다. 먼저 계약 당사자를 명확히 하고, 계약 목적과 내용을 구체적으로 명시해야 합니다.",
            "context": {
                "user_preference": "medium",
                "user_emotion": "neutral",
                "user_type": "general",
                "expertise_level": "beginner",
                "question_type": "contract",
                "previous_topic": "계약 관련"
            }
        },
        {
            "answer": "귀하의 질문에 대해 답변드리겠습니다. 민법 제543조에 따르면 계약 해지 조건은 다음과 같습니다.",
            "context": {
                "user_preference": "formal",
                "user_emotion": "urgent",
                "user_type": "lawyer",
                "expertise_level": "expert",
                "question_type": "law_article",
                "previous_topic": ""
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"답변: {test_case['answer']}")
        print(f"맥락: {test_case['context']}")
        
        metrics = evaluator.evaluate_naturalness(
            test_case['answer'], 
            test_case['context']
        )
        
        print(f"자연스러움 점수: {metrics.overall_naturalness:.2f}")
        print(f"자연스러움 레벨: {metrics.detailed_analysis['naturalness_level']}")
        print(f"강점: {metrics.detailed_analysis['strengths']}")
        print(f"약점: {metrics.detailed_analysis['weaknesses']}")
        print(f"개선 제안: {metrics.detailed_analysis['suggestions']}")


def test_integrated_system():
    """통합 시스템 테스트"""
    print("\n=== 통합 시스템 테스트 ===")
    
    # 모든 컴포넌트 초기화
    connector = ConversationConnector()
    adjuster = EmotionalToneAdjuster()
    learner = PersonalizedStyleLearner()
    feedback_system = RealtimeFeedbackSystem()
    evaluator = NaturalnessEvaluator()
    
    # 테스트 시나리오
    original_answer = "계약서 작성 시 주의사항을 설명드리겠습니다."
    user_id = "test_user"
    
    print(f"원본 답변: {original_answer}")
    
    # 1단계: 대화 연결어 추가
    context = {
        "previous_topic": "계약 관련 질문",
        "conversation_flow": "follow_up",
        "user_emotion": "anxious",
        "question_type": "contract"
    }
    
    step1_answer = connector.add_natural_connectors(original_answer, context)
    print(f"1단계 (연결어 추가): {step1_answer}")
    
    # 2단계: 감정 톤 조절
    step2_answer = adjuster.adjust_tone(step1_answer, "anxious", 0.7)
    print(f"2단계 (감정 톤 조절): {step2_answer}")
    
    # 3단계: 개인화된 스타일 적용
    step3_answer = learner.apply_personalized_style(step2_answer, user_id)
    print(f"3단계 (개인화): {step3_answer}")
    
    # 4단계: 자연스러움 평가
    evaluation_context = {
        "user_preference": "medium",
        "user_emotion": "anxious",
        "user_type": "general",
        "expertise_level": "beginner",
        "question_type": "contract",
        "previous_topic": "계약 관련 질문"
    }
    
    metrics = evaluator.evaluate_naturalness(step3_answer, evaluation_context)
    print(f"4단계 (자연스러움 평가): {metrics.overall_naturalness:.2f}")
    print(f"최종 개선된 답변: {step3_answer}")
    print(f"자연스러움 레벨: {metrics.detailed_analysis['naturalness_level']}")


def main():
    """메인 테스트 함수"""
    print("🚀 자연스러운 답변 개선 시스템 테스트 시작")
    print("=" * 60)
    
    try:
        # 각 컴포넌트별 테스트
        test_conversation_connector()
        test_emotional_tone_adjuster()
        test_personalized_style_learner()
        test_realtime_feedback_system()
        test_naturalness_evaluator()
        
        # 통합 시스템 테스트
        test_integrated_system()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
