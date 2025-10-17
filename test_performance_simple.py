#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 모니터링 및 피드백 시스템 테스트
"""

import sys
import os
import time
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_performance_monitoring():
    """성능 모니터링 테스트"""
    print("성능 모니터링 테스트")
    print("=" * 40)
    
    try:
        from source.services.performance_monitoring import PerformanceMonitorService
        
        # 성능 모니터 생성
        monitor = PerformanceMonitorService()
        
        # 모니터링 시작
        monitor.start_monitoring()
        
        # 테스트 이벤트 로깅
        monitor.log_event("test_event", {"test": True, "timestamp": time.time()})
        
        # 메트릭 조회
        metrics = monitor.get_metrics()
        print(f"현재 메트릭 수: {len(metrics)}")
        
        # 모니터링 중지
        monitor.stop_monitoring()
        
        print("성능 모니터링 테스트 성공")
        return True
        
    except Exception as e:
        print(f"성능 모니터링 테스트 실패: {e}")
        return False

def test_feedback_system():
    """피드백 시스템 테스트"""
    print("\n피드백 시스템 테스트")
    print("=" * 40)
    
    try:
        from source.services.feedback_system import FeedbackSystem
        
        # 피드백 시스템 생성
        feedback_system = FeedbackSystem()
        
        # 테스트 피드백 제출
        test_feedback = {
            "feedback_type": "rating",
            "rating": 5,
            "text_content": "테스트용 피드백입니다",
            "question": "테스트 질문",
            "answer": "테스트 답변",
            "session_id": "test_session",
            "context": {"test": True}
        }
        
        feedback_id = feedback_system.submit_feedback(test_feedback)
        print(f"피드백 제출 성공: {feedback_id}")
        
        # 피드백 조회
        feedback = feedback_system.get_feedback(feedback_id)
        if feedback:
            print(f"피드백 조회 성공: {feedback['feedback_type']}")
        
        # 피드백 분석
        analysis = feedback_system.analyze_feedback()
        print(f"피드백 분석 결과: {len(analysis)}개 항목")
        
        print("피드백 시스템 테스트 성공")
        return True
        
    except Exception as e:
        print(f"피드백 시스템 테스트 실패: {e}")
        return False

def test_vector_search():
    """벡터 검색 테스트"""
    print("\n벡터 검색 테스트")
    print("=" * 40)
    
    try:
        from source.data.vector_store import LegalVectorStore
        
        # 벡터 스토어 생성
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        
        # 검색 테스트
        query = "손해배상"
        results = vector_store.search(query, top_k=5)
        
        print(f"검색 쿼리: {query}")
        print(f"검색 결과 수: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. 유사도: {result.get('similarity', 0):.3f}")
            print(f"     내용: {result.get('content', '')[:100]}...")
        
        if len(results) > 0:
            print("벡터 검색 테스트 성공")
            return True
        else:
            print("검색 결과가 없습니다")
            return False
            
    except Exception as e:
        print(f"벡터 검색 테스트 실패: {e}")
        return False

def test_question_classifier():
    """질문 분류기 테스트"""
    print("\n질문 분류기 테스트")
    print("=" * 40)
    
    try:
        from source.services.question_classifier import QuestionClassifier
        
        # 질문 분류기 생성
        classifier = QuestionClassifier()
        
        # 테스트 질문들
        test_questions = [
            "손해배상 관련 판례를 찾아주세요",
            "민법 제750조의 내용이 무엇인가요?",
            "계약 해제 조건이 무엇인가요?",
            "이혼 절차는 어떻게 진행하나요?",
            "불법행위의 법적 근거를 알려주세요"
        ]
        
        for question in test_questions:
            classification = classifier.classify_question(question)
            print(f"질문: {question}")
            print(f"  유형: {classification.question_type.value}")
            print(f"  신뢰도: {classification.confidence:.2f}")
            print(f"  법률 가중치: {classification.law_weight}")
            print(f"  판례 가중치: {classification.precedent_weight}")
            print()
        
        print("질문 분류기 테스트 성공")
        return True
        
    except Exception as e:
        print(f"질문 분류기 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LawFirmAI 성능 모니터링 및 피드백 시스템 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("성능 모니터링", test_performance_monitoring),
        ("피드백 시스템", test_feedback_system),
        ("벡터 검색", test_vector_search),
        ("질문 분류기", test_question_classifier)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 테스트 중 오류 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "통과" if result else "실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 결과: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("모든 성능 모니터링 및 피드백 시스템 테스트가 성공적으로 완료되었습니다!")
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인하세요.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
