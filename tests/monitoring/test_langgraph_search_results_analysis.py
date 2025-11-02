#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우에서 검색 결과 포함 분석
실제 워크플로우 실행 및 로그 분석
"""

import logging
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_langgraph_search_results_flow():
    """LangGraph 워크플로우에서 검색 결과 전달 확인"""
    print("\n" + "="*80)
    print("LangGraph 워크플로우 검색 결과 포함 분석")
    print("="*80 + "\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        workflow_service = LangGraphWorkflowService(config)

        print("✅ 워크플로우 서비스 초기화 완료\n")

        # 테스트 케이스
        test_query = "손해배상 청구 방법을 알려주세요"
        print(f"📋 테스트 질문: {test_query}\n")
        print("🔄 워크플로우 실행 중...\n")

        # 워크플로우 실행
        result = await workflow_service.process_query(test_query)

        # 결과 분석
        print("\n" + "="*80)
        print("결과 분석")
        print("="*80 + "\n")

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        processing_steps = result.get("processing_steps", [])

        print(f"📝 답변 길이: {len(answer)}자")
        print(f"📚 출처 수: {len(sources)}개")
        print(f"🎯 신뢰도: {confidence:.2f}")
        print(f"⏱️ 처리 단계: {len(processing_steps)}개")

        # 검색 관련 단계 확인
        search_steps = [step for step in processing_steps if "검색" in step or "search" in step.lower()]
        print(f"\n🔍 검색 관련 단계: {len(search_steps)}개")
        for step in search_steps[:10]:
            print(f"   - {step}")

        # sources 확인
        if sources:
            print(f"\n📚 출처 상세:")
            for i, source in enumerate(sources[:5], 1):
                print(f"   {i}. {source}")
        else:
            print("\n⚠️ 출처가 없습니다!")

        # 답변에 검색 결과 인용 확인
        answer_lower = answer.lower()
        citation_keywords = ["제", "조", "법", "판례", "대법원"]
        has_citations = any(kw in answer_lower for kw in citation_keywords)

        print(f"\n📖 답변 인용 확인:")
        print(f"   - 법률 조문/판례 키워드 포함: {'✅' if has_citations else '❌'}")

        # 답변 미리보기
        print(f"\n📋 답변 미리보기 (첫 500자):")
        print("-" * 80)
        print(answer[:500])
        print("-" * 80)

        return {
            "answer_length": len(answer),
            "sources_count": len(sources),
            "confidence": confidence,
            "has_citations": has_citations,
            "processing_steps_count": len(processing_steps)
        }

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(test_langgraph_search_results_flow())

    if result:
        print("\n" + "="*80)
        print("분석 결과 요약")
        print("="*80)
        print(f"답변 길이: {result['answer_length']}자")
        print(f"출처 수: {result['sources_count']}개")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"법률 인용: {'✅' if result['has_citations'] else '❌'}")
        print(f"처리 단계: {result['processing_steps_count']}개")
        print("="*80 + "\n")
