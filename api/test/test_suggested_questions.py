"""
추천 질문 기능 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
from api.services.chat_service import ChatService


async def test_suggested_questions():
    """추천 질문 생성 테스트"""
    print("=" * 60)
    print("추천 질문 기능 테스트 시작")
    print("=" * 60)
    
    # ChatService 인스턴스 생성
    chat_service = ChatService()
    
    # 테스트 질문
    test_queries = [
        "계약서 작성 시 주의사항은 무엇인가요?",
        "손해배상 청구는 어떻게 하나요?",
        "이혼 절차를 알려주세요",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}] 질문: {query}")
        print("-" * 60)
        
        try:
            # 메시지 처리
            result = await chat_service.process_message(
                message=query,
                session_id=f"test-session-{i}",
                enable_checkpoint=False
            )
            
            # 결과 확인
            print(f"✓ 응답 생성 완료")
            print(f"  - 답변 길이: {len(result.get('answer', ''))}자")
            print(f"  - 신뢰도: {result.get('confidence', 0.0):.2f}")
            print(f"  - 질문 유형: {result.get('query_type', 'unknown')}")
            
            # metadata 확인
            metadata = result.get('metadata', {})
            print(f"\n  [Metadata]")
            print(f"  - Keys: {list(metadata.keys())}")
            
            # related_questions 확인
            related_questions = metadata.get('related_questions')
            if related_questions:
                print(f"\n  ✓ 추천 질문 생성 성공!")
                print(f"  - 추천 질문 개수: {len(related_questions)}")
                for idx, question in enumerate(related_questions, 1):
                    print(f"    {idx}. {question}")
            else:
                print(f"\n  ✗ 추천 질문이 생성되지 않았습니다")
                print(f"  - metadata 내용: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
            
            # phase_info 확인 (디버깅용)
            if 'phase_info' in result:
                phase_info = result.get('phase_info', {})
                phase2 = phase_info.get('phase2', {})
                flow_tracking = phase2.get('flow_tracking_info', {})
                suggested = flow_tracking.get('suggested_questions', [])
                if suggested:
                    print(f"\n  [Phase Info]")
                    print(f"  - Phase2 flow_tracking_info에 suggested_questions 있음: {len(suggested)}개")
            
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_suggested_questions())

