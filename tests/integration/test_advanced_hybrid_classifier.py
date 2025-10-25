#!/usr/bin/env python3
"""
하이브리드 분류기 고급 테스트 스크립트
더 복잡한 질문들로 하이브리드 분류 테스트
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.enhanced_chat_service import EnhancedChatService
from source.utils.config import Config

async def test_advanced_hybrid_classifier():
    """하이브리드 분류기 고급 테스트"""
    print("=" * 60)
    print("하이브리드 분류기 고급 테스트")
    print("=" * 60)
    
    try:
        # 설정 로드
        config = Config()
        
        # EnhancedChatService 초기화
        print("EnhancedChatService 초기화 중...")
        chat_service = EnhancedChatService(config)
        print("✓ EnhancedChatService 초기화 완료")
        
        # 하이브리드 분류기 상태 확인
        if hasattr(chat_service, 'hybrid_classifier') and chat_service.hybrid_classifier:
            print("✓ 하이브리드 분류기 초기화 완료")
            
            # 임계값을 낮춰서 ML 모델이 더 자주 사용되도록 설정
            chat_service.adjust_classifier_threshold(0.9)
            print("✓ 임계값을 0.9로 조정 (ML 모델 사용 증가)")
        else:
            print("✗ 하이브리드 분류기 초기화 실패")
            return
        
        # 복잡한 테스트 질문들
        complex_questions = [
            "회사에서 직원을 해고할 때 어떤 절차를 따라야 하나요?",
            "부동산 매매 계약서에 문제가 있어서 손해를 입었습니다. 어떻게 해야 하나요?",
            "이혼할 때 자녀의 양육권을 어떻게 결정하나요?",
            "상속받은 부동산을 다른 형제들과 나누려고 합니다.",
            "근로시간이 법정 기준을 초과하는데 임금을 받지 못했습니다.",
            "계약서에 명시되지 않은 추가 비용을 요구받고 있습니다.",
            "사업자 등록을 하지 않고 사업을 했는데 문제가 될까요?",
            "임대차 계약 기간이 끝났는데 계속 거주할 수 있나요?",
            "회사에서 부당한 해고를 당했을 때 구제받을 방법이 있나요?",
            "상속 포기를 하고 싶은데 어떤 절차가 필요한가요?"
        ]
        
        print("\n복잡한 질문 분석 테스트:")
        print("=" * 60)
        
        for i, question in enumerate(complex_questions, 1):
            try:
                print(f"\n[{i}] 질문: {question}")
                
                # 질문 분석
                analysis = await chat_service._analyze_query(
                    message=question,
                    context=None,
                    user_id="test_user",
                    session_id="test_session"
                )
                
                print(f"분류 결과:")
                print(f"- 질문 유형: {analysis.get('query_type', 'unknown')}")
                print(f"- 도메인: {analysis.get('domain', 'unknown')}")
                print(f"- 신뢰도: {analysis.get('confidence', 0):.3f}")
                print(f"- 분류 방법: {analysis.get('classification_method', 'unknown')}")
                print(f"- 분류 이유: {analysis.get('classification_reasoning', 'unknown')}")
                
                if 'law_weight' in analysis:
                    print(f"- 법률 가중치: {analysis['law_weight']:.2f}")
                if 'precedent_weight' in analysis:
                    print(f"- 판례 가중치: {analysis['precedent_weight']:.2f}")
                
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
            
            print("-" * 40)
        
        # 성능 통계
        print("\n하이브리드 분류기 통계:")
        stats = chat_service.get_hybrid_classifier_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 임계값별 테스트
        print("\n임계값별 분류 방법 비교:")
        print("=" * 60)
        
        test_question = "회사에서 직원을 해고할 때 어떤 절차를 따라야 하나요?"
        
        for threshold in [0.5, 0.7, 0.9]:
            chat_service.adjust_classifier_threshold(threshold)
            analysis = await chat_service._analyze_query(
                message=test_question,
                context=None,
                user_id="test_user",
                session_id="test_session"
            )
            print(f"임계값 {threshold}: {analysis.get('classification_method', 'unknown')} "
                  f"(신뢰도: {analysis.get('confidence', 0):.3f})")
        
        print("\n✓ 고급 테스트 완료!")
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_advanced_hybrid_classifier())
