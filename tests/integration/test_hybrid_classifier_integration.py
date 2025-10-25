#!/usr/bin/env python3
"""
하이브리드 분류기 통합 테스트 스크립트
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

async def test_hybrid_classifier_integration():
    """하이브리드 분류기 통합 테스트"""
    print("=" * 60)
    print("하이브리드 분류기 통합 테스트")
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
        else:
            print("✗ 하이브리드 분류기 초기화 실패")
            return
        
        # 테스트 질문들
        test_questions = [
            "민법 제750조 손해배상에 대해 알려주세요",
            "계약서를 검토해주세요",
            "이혼 절차는 어떻게 되나요?",
            "어떻게 대응해야 하나요?",
            "관련 판례를 찾아주세요",
            "무엇이 불법행위인가요?"
        ]
        
        print("\n질문 분석 테스트:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
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
        
        print("\n✓ 통합 테스트 완료!")
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hybrid_classifier_integration())
