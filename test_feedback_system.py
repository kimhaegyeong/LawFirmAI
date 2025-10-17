#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
피드백 시스템 테스트 스크립트
"""

def test_feedback_system():
    """피드백 시스템 테스트"""
    try:
        from source.services.feedback_system import FeedbackCollector, FeedbackAnalyzer, FeedbackType, FeedbackRating, Feedback
        print("✅ 피드백 시스템 임포트 성공")
        
        # 피드백 수집기 생성
        collector = FeedbackCollector()
        print("✅ 피드백 수집기 생성 성공")
        
        # 피드백 분석기 생성
        analyzer = FeedbackAnalyzer(collector)
        print("✅ 피드백 분석기 생성 성공")
        
        # 피드백 제출 테스트
        result = collector.submit_feedback(
            feedback_type=FeedbackType.RATING,
            rating=FeedbackRating.GOOD,
            text_content="테스트 피드백입니다.",
            question="손해배상 관련 질문",
            answer="테스트 응답",
            session_id="test_session",
            user_id="test_user",
            context={"test": "context"},
            metadata={"test": "metadata"}
        )
        print(f"✅ 피드백 제출 성공: {result}")
        
        # 피드백 조회 테스트
        feedbacks = collector.get_feedback_list(limit=5)
        print(f"✅ 피드백 조회 성공: {len(feedbacks)}개")
        
        # 피드백 분석 테스트
        analysis = analyzer.analyze_feedback_trends(days=7)
        print(f"✅ 피드백 분석 성공: {len(analysis)}개 항목")
        
        return True
        
    except Exception as e:
        print(f"❌ 피드백 시스템 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("피드백 시스템 테스트 시작")
    print("=" * 50)
    
    success = test_feedback_system()
    
    print("=" * 50)
    if success:
        print("🎉 피드백 시스템 테스트 성공!")
    else:
        print("💥 피드백 시스템 테스트 실패!")
