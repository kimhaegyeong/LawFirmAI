#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContentFilterEngine 분석 스크립트
"""

import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def analyze_content_filter_engine():
    """ContentFilterEngine 분석"""
    print("=" * 60)
    print("ContentFilterEngine 분석")
    print("=" * 60)
    
    try:
        from source.services.content_filter_engine import ContentFilterEngine
        
        # 테스트 질문들
        test_questions = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
            "부동산 매매 계약 시 주의사항이 있나요?",
            "제가 겪고 있는 문제에 대해 조언해주세요.",  # 개인적 조언 요청
        ]
        
        filter_engine = ContentFilterEngine()
        
        print("\n" + "-" * 40)
        print("질문별 필터링 상태 분석")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 30)
            
            # 필터링 검사
            result = filter_engine.filter_content(question)
            
            print(f"차단 여부: {'차단됨' if result.is_blocked else '허용됨'}")
            print(f"의도 유형: {result.intent_analysis.intent_type}")
            print(f"맥락 유형: {result.intent_analysis.context_type}")
            print(f"위험 수준: {result.intent_analysis.risk_level}")
            print(f"신뢰도: {result.intent_analysis.confidence:.3f}")
            print(f"차단 사유: {result.block_reason if result.block_reason else '없음'}")
            
            if result.safe_alternatives:
                print(f"안전한 대안: {len(result.safe_alternatives)}개")
                for alt in result.safe_alternatives[:2]:  # 처음 2개만 표시
                    print(f"  - {alt}")
        
        print("\n" + "-" * 40)
        print("의도 분석 패턴 확인")
        print("-" * 40)
        
        # 의도 분석 패턴 확인
        intent_patterns = filter_engine.intent_patterns
        print(f"의도 패턴 수: {len(intent_patterns)}")
        
        for intent_type, patterns_info in intent_patterns.items():
            print(f"\n{intent_type}:")
            print(f"  패턴 수: {len(patterns_info['patterns'])}")
            print(f"  키워드 수: {len(patterns_info['keywords'])}")
            print(f"  위험 수준: {patterns_info['risk_level']}")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ContentFilterEngine 분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    analyze_content_filter_engine()

