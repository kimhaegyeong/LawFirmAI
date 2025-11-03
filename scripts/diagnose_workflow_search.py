# -*- coding: utf-8 -*-
"""
워크플로우 검색 진단 스크립트
LangGraph 워크플로우 실행 후 검색 관련 문제 진단
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

# source/services/workflow_service.py를 사용하도록 변경
from langgraph_core.services.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/workflow_diagnosis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


def analyze_search_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """검색 결과 분석"""
    analysis = {
        "has_answer": bool(result.get("answer", "")),
        "answer_length": len(result.get("answer", "")),
        "has_sources": len(result.get("sources", [])) > 0,
        "sources_count": len(result.get("sources", [])),
        "sources_list": result.get("sources", [])[:10],
        "has_retrieved_docs": len(result.get("retrieved_docs", [])) > 0,
        "retrieved_docs_count": len(result.get("retrieved_docs", [])),
        "confidence": result.get("confidence", 0.0),
        "has_errors": len(result.get("errors", [])) > 0,
        "errors": result.get("errors", []),
        "processing_time": result.get("processing_time", 0.0),
    }

    # retrieved_docs 분석
    if analysis["has_retrieved_docs"]:
        docs = result.get("retrieved_docs", [])
        doc_types = {}
        doc_sources = {}
        doc_scores = []

        for doc in docs[:10]:  # 상위 10개만 분석
            doc_type = doc.get("type", doc.get("doc_type", "unknown"))
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            doc_source = doc.get("source", "Unknown")
            doc_sources[doc_source] = doc_sources.get(doc_source, 0) + 1

            score = doc.get("relevance_score", 0.0)
            if score > 0:
                doc_scores.append(score)

        analysis["doc_types"] = doc_types
        analysis["doc_sources"] = doc_sources
        if doc_scores:
            analysis["avg_score"] = sum(doc_scores) / len(doc_scores)
            analysis["min_score"] = min(doc_scores)
            analysis["max_score"] = max(doc_scores)

    return analysis


async def diagnose_workflow_search(query: str):
    """워크플로우 검색 진단"""
    print("=" * 80)
    print("워크플로우 검색 진단")
    print("=" * 80)
    print(f"\n진단 쿼리: {query}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()

        # 워크플로우 서비스 초기화
        logger.info("워크플로우 서비스 초기화 중...")
        workflow_service = LangGraphWorkflowService(config)

        # 쿼리 처리
        logger.info(f"쿼리 처리 시작: {query}")
        session_id = f"diagnosis_{int(datetime.now().timestamp())}"

        result = await workflow_service.process_query(
            query=query,
            session_id=session_id,
            enable_checkpoint=False
        )

        # 결과 분석
        analysis = analyze_search_results(result)

        # 진단 결과 출력
        print("\n" + "=" * 80)
        print("진단 결과")
        print("=" * 80)

        print(f"\n[답변]")
        print(f"  - 생성 여부: {'✅ 있음' if analysis['has_answer'] else '❌ 없음'}")
        print(f"  - 길이: {analysis['answer_length']}자")
        print(f"  - 신뢰도: {analysis['confidence']:.2%}")

        print(f"\n[검색 결과]")
        print(f"  - retrieved_docs: {'✅ 있음' if analysis['has_retrieved_docs'] else '❌ 없음'} ({analysis['retrieved_docs_count']}개)")
        print(f"  - sources: {'✅ 있음' if analysis['has_sources'] else '❌ 없음'} ({analysis['sources_count']}개)")

        if analysis['has_retrieved_docs']:
            print(f"\n  [문서 타입 분포]")
            for doc_type, count in analysis.get('doc_types', {}).items():
                print(f"    - {doc_type}: {count}개")

            print(f"\n  [문서 소스 분포]")
            for source, count in list(analysis.get('doc_sources', {}).items())[:5]:
                print(f"    - {source}: {count}개")

            if 'avg_score' in analysis:
                print(f"\n  [점수 통계]")
                print(f"    - 평균: {analysis['avg_score']:.3f}")
                print(f"    - 최소: {analysis['min_score']:.3f}")
                print(f"    - 최대: {analysis['max_score']:.3f}")

        if analysis['has_sources']:
            print(f"\n  [Sources 목록]")
            for i, source in enumerate(analysis['sources_list'][:10], 1):
                print(f"    {i}. {source}")

        print(f"\n[처리 정보]")
        print(f"  - 처리 시간: {analysis['processing_time']:.2f}초")
        print(f"  - 에러 여부: {'⚠️ 있음' if analysis['has_errors'] else '✅ 없음'}")

        if analysis['has_errors']:
            print(f"\n  [에러 목록]")
            for error in analysis['errors']:
                print(f"    - {error}")

        # 문제 진단
        print(f"\n" + "=" * 80)
        print("문제 진단")
        print("=" * 80)

        issues = []
        recommendations = []

        if not analysis['has_retrieved_docs']:
            issues.append("❌ 검색 결과가 없습니다 (retrieved_docs가 비어있음)")
            recommendations.append("  → 검색 쿼리 확인 필요")
            recommendations.append("  → 검색 컴포넌트 직접 테스트 필요")
            recommendations.append("  → 임계값 필터링 확인 필요")

        if not analysis['has_sources'] and analysis['has_retrieved_docs']:
            issues.append("⚠️ Sources가 추출되지 않았습니다")
            recommendations.append("  → retrieved_docs의 source 필드 확인 필요")
            recommendations.append("  → prepare_final_response의 sources 추출 로직 확인 필요")

        if analysis['has_retrieved_docs'] and 'avg_score' in analysis:
            if analysis['avg_score'] < 0.3:
                issues.append("⚠️ 검색 결과의 평균 점수가 낮습니다 (0.3 미만)")
                recommendations.append("  → 검색 쿼리 최적화 필요")
                recommendations.append("  → 임계값 조정 검토 필요")

        if not analysis['has_answer']:
            issues.append("❌ 답변이 생성되지 않았습니다")
            recommendations.append("  → 검색 결과 부족 가능성")
            recommendations.append("  → LLM 호출 실패 가능성")

        if issues:
            print("\n발견된 문제:")
            for issue in issues:
                print(f"  {issue}")

            if recommendations:
                print("\n권장 조치:")
                for rec in set(recommendations):  # 중복 제거
                    print(f"  {rec}")
        else:
            print("\n✅ 특별한 문제가 발견되지 않았습니다.")

        print("\n" + "=" * 80)
        print("진단 완료")
        print(f"로그 파일: logs/workflow_diagnosis_*.log")
        print("=" * 80)

        return result, analysis

    except Exception as e:
        logger.error(f"진단 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ 진단 실패: {e}")
        return None, None


def main():
    """메인 함수"""
    # 테스트 쿼리
    test_query = "민사법에서 계약 해지 요건은 무엇인가요?"

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 비동기 실행
    result, analysis = asyncio.run(diagnose_workflow_search(test_query))

    if result and analysis:
        print(f"\n✅ 진단이 완료되었습니다.")
        print(f"상세 로그는 logs/ 디렉토리를 확인하세요.")


if __name__ == "__main__":
    main()
