# -*- coding: utf-8 -*-
"""
?�크?�로??검??진단 ?�크립트
LangGraph ?�크?�로???�행 ??검??관??문제 진단
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# core/agents/workflow_service.py�??�용?�도�?변�?
from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 로깅 ?�정
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
    """검??결과 분석"""
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

        for doc in docs[:10]:  # ?�위 10개만 분석
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
    """?�크?�로??검??진단"""
    print("=" * 80)
    print("?�크?�로??검??진단")
    print("=" * 80)
    print(f"\n진단 쿼리: {query}")
    print(f"?�작 ?�간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # ?�정 로드
        config = LangGraphConfig.from_env()

        # ?�크?�로???�비??초기??
        logger.info("?�크?�로???�비??초기??�?..")
        workflow_service = LangGraphWorkflowService(config)

        # 쿼리 처리
        logger.info(f"쿼리 처리 ?�작: {query}")
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

        print(f"\n[?��?]")
        print(f"  - ?�성 ?��?: {'???�음' if analysis['has_answer'] else '???�음'}")
        print(f"  - 길이: {analysis['answer_length']}??)
        print(f"  - ?�뢰?? {analysis['confidence']:.2%}")

        print(f"\n[검??결과]")
        print(f"  - retrieved_docs: {'???�음' if analysis['has_retrieved_docs'] else '???�음'} ({analysis['retrieved_docs_count']}�?")
        print(f"  - sources: {'???�음' if analysis['has_sources'] else '???�음'} ({analysis['sources_count']}�?")

        if analysis['has_retrieved_docs']:
            print(f"\n  [문서 ?�??분포]")
            for doc_type, count in analysis.get('doc_types', {}).items():
                print(f"    - {doc_type}: {count}�?)

            print(f"\n  [문서 ?�스 분포]")
            for source, count in list(analysis.get('doc_sources', {}).items())[:5]:
                print(f"    - {source}: {count}�?)

            if 'avg_score' in analysis:
                print(f"\n  [?�수 ?�계]")
                print(f"    - ?�균: {analysis['avg_score']:.3f}")
                print(f"    - 최소: {analysis['min_score']:.3f}")
                print(f"    - 최�?: {analysis['max_score']:.3f}")

        if analysis['has_sources']:
            print(f"\n  [Sources 목록]")
            for i, source in enumerate(analysis['sources_list'][:10], 1):
                print(f"    {i}. {source}")

        print(f"\n[처리 ?�보]")
        print(f"  - 처리 ?�간: {analysis['processing_time']:.2f}�?)
        print(f"  - ?�러 ?��?: {'?�️ ?�음' if analysis['has_errors'] else '???�음'}")

        if analysis['has_errors']:
            print(f"\n  [?�러 목록]")
            for error in analysis['errors']:
                print(f"    - {error}")

        # 문제 진단
        print(f"\n" + "=" * 80)
        print("문제 진단")
        print("=" * 80)

        issues = []
        recommendations = []

        if not analysis['has_retrieved_docs']:
            issues.append("??검??결과가 ?�습?�다 (retrieved_docs가 비어?�음)")
            recommendations.append("  ??검??쿼리 ?�인 ?�요")
            recommendations.append("  ??검??컴포?�트 직접 ?�스???�요")
            recommendations.append("  ???�계�??�터�??�인 ?�요")

        if not analysis['has_sources'] and analysis['has_retrieved_docs']:
            issues.append("?�️ Sources가 추출?��? ?�았?�니??)
            recommendations.append("  ??retrieved_docs??source ?�드 ?�인 ?�요")
            recommendations.append("  ??prepare_final_response??sources 추출 로직 ?�인 ?�요")

        if analysis['has_retrieved_docs'] and 'avg_score' in analysis:
            if analysis['avg_score'] < 0.3:
                issues.append("?�️ 검??결과???�균 ?�수가 ??��?�다 (0.3 미만)")
                recommendations.append("  ??검??쿼리 최적???�요")
                recommendations.append("  ???�계�?조정 검???�요")

        if not analysis['has_answer']:
            issues.append("???��????�성?��? ?�았?�니??)
            recommendations.append("  ??검??결과 부�?가?�성")
            recommendations.append("  ??LLM ?�출 ?�패 가?�성")

        if issues:
            print("\n발견??문제:")
            for issue in issues:
                print(f"  {issue}")

            if recommendations:
                print("\n권장 조치:")
                for rec in set(recommendations):  # 중복 ?�거
                    print(f"  {rec}")
        else:
            print("\n???�별??문제가 발견?��? ?�았?�니??")

        print("\n" + "=" * 80)
        print("진단 ?�료")
        print(f"로그 ?�일: logs/workflow_diagnosis_*.log")
        print("=" * 80)

        return result, analysis

    except Exception as e:
        logger.error(f"진단 �??�류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n??진단 ?�패: {e}")
        return None, None


def main():
    """메인 ?�수"""
    # ?�스??쿼리
    test_query = "민사법에??계약 ?��? ?�건?� 무엇?��???"

    # 로그 ?�렉?�리 ?�성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 비동�??�행
    result, analysis = asyncio.run(diagnose_workflow_search(test_query))

    if result and analysis:
        print(f"\n??진단???�료?�었?�니??")
        print(f"?�세 로그??logs/ ?�렉?�리�??�인?�세??")


if __name__ == "__main__":
    main()
