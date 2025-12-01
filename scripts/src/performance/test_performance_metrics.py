# -*- coding: utf-8 -*-
"""
성능 지표 수집 및 분석 스크립트
Sources 변환률, Legal References 생성률 등 측정
"""

import sys
import os
from pathlib import Path
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 설정
os.environ['USE_STREAMING_MODE'] = 'false'

def _calculate_sources_conversion_rate(retrieved_docs: List[Dict[str, Any]], sources: List[str]) -> float:
    """Sources 변환률 계산"""
    if not retrieved_docs:
        return 0.0
    return (len(sources) / len(retrieved_docs)) * 100


def _calculate_legal_references_generation_rate(
    retrieved_docs: List[Dict[str, Any]], 
    legal_refs: List[str]
) -> float:
    """Legal References 생성률 계산 (개선된 로직)"""
    if not legal_refs:
        return 0.0
    
    statute_articles = [
        doc for doc in retrieved_docs 
        if (doc.get("type") == "statute_article" or 
            doc.get("source_type") == "statute_article" or
            (isinstance(doc.get("metadata"), dict) and 
             doc.get("metadata", {}).get("source_type") == "statute_article"))
    ]
    
    if statute_articles:
        return (len(legal_refs) / len(statute_articles)) * 100
    else:
        if retrieved_docs:
            return (len(legal_refs) / len(retrieved_docs)) * 100
        return 0.0


def _calculate_sources_detail_generation_rate(
    sources: List[str], 
    sources_detail: List[Dict[str, Any]]
) -> float:
    """Sources Detail 생성률 계산"""
    if not sources:
        return 0.0
    return (len(sources_detail) / len(sources)) * 100


def _calculate_type_distribution(docs: List[Dict[str, Any]], type_key: str = "type") -> Dict[str, int]:
    """문서 타입별 분포 계산"""
    doc_types = {}
    for doc in docs:
        doc_type = (
            doc.get(type_key) or 
            doc.get("source_type") or 
            (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None) or
            "unknown"
        )
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    return doc_types


def _collect_basic_metrics(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """기본 성능 지표 수집"""
    return {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "answer_length": len(result.get("answer", "")),
        "retrieved_docs_count": len(result.get("retrieved_docs", [])),
        "sources_count": len(result.get("sources", [])),
        "sources_detail_count": len(result.get("sources_detail", [])),
        "legal_references_count": len(result.get("legal_references", [])),
        "related_questions_count": len(result.get("related_questions", [])),
        "confidence": result.get("confidence", 0.0),
    }


async def test_query_and_collect_metrics(query: str) -> Dict[str, Any]:
    """질의 테스트 및 성능 지표 수집"""
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
    except ImportError:
        from config.langgraph_config import LangGraphConfig
        from core.workflow.workflow_service import LangGraphWorkflowService
    
    print(f"\n{'='*80}")
    print(f"테스트 질의: {query}")
    print(f"{'='*80}\n")
    
    config = LangGraphConfig.from_env()
    config.enable_checkpoint = False
    
    service = LangGraphWorkflowService(config)
    
    result = await service.process_query(
        query=query,
        session_id="performance_test",
        enable_checkpoint=False
    )
    
    metrics = _collect_basic_metrics(result, query)
    
    retrieved_docs = result.get("retrieved_docs", [])
    sources = result.get("sources", [])
    legal_refs = result.get("legal_references", [])
    sources_detail = result.get("sources_detail", [])
    
    metrics["sources_conversion_rate"] = _calculate_sources_conversion_rate(retrieved_docs, sources)
    metrics["legal_references_generation_rate"] = _calculate_legal_references_generation_rate(retrieved_docs, legal_refs)
    metrics["sources_detail_generation_rate"] = _calculate_sources_detail_generation_rate(sources, sources_detail)
    
    metrics["retrieved_docs_type_distribution"] = _calculate_type_distribution(retrieved_docs)
    metrics["sources_type_distribution"] = _calculate_type_distribution(sources_detail, "type")
    
    return metrics

def _print_metrics(metrics: Dict[str, Any], query: str) -> None:
    """성능 지표 출력"""
    print(f"\n{'='*80}")
    print(f"성능 지표 - {query}")
    print(f"{'='*80}")
    print(f"답변 길이: {metrics['answer_length']}자")
    print(f"검색된 문서 수: {metrics['retrieved_docs_count']}개")
    print(f"Sources 수: {metrics['sources_count']}개")
    print(f"Sources Detail 수: {metrics['sources_detail_count']}개")
    print(f"Legal References 수: {metrics['legal_references_count']}개")
    print(f"Related Questions 수: {metrics['related_questions_count']}개")
    print(f"Sources 변환률: {metrics['sources_conversion_rate']:.1f}%")
    print(f"Legal References 생성률: {metrics['legal_references_generation_rate']:.1f}%")
    print(f"Sources Detail 생성률: {metrics['sources_detail_generation_rate']:.1f}%")
    print(f"신뢰도: {metrics['confidence']:.2f}")
    print(f"\n문서 타입 분포: {metrics['retrieved_docs_type_distribution']}")
    print(f"Sources 타입 분포: {metrics['sources_type_distribution']}")
    print(f"{'='*80}\n")


def _calculate_average_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """평균 성능 지표 계산"""
    if not all_metrics:
        return {}
    
    return {
        "avg_conversion_rate": sum(m["sources_conversion_rate"] for m in all_metrics) / len(all_metrics),
        "avg_legal_refs_rate": sum(m["legal_references_generation_rate"] for m in all_metrics) / len(all_metrics),
        "avg_detail_rate": sum(m["sources_detail_generation_rate"] for m in all_metrics) / len(all_metrics),
        "avg_answer_length": sum(m["answer_length"] for m in all_metrics) / len(all_metrics),
    }


def _print_average_metrics(avg_metrics: Dict[str, float]) -> None:
    """평균 성능 지표 출력"""
    print(f"\n{'='*80}")
    print("평균 성능 지표")
    print(f"{'='*80}")
    print(f"평균 Sources 변환률: {avg_metrics['avg_conversion_rate']:.1f}%")
    print(f"평균 Legal References 생성률: {avg_metrics['avg_legal_refs_rate']:.1f}%")
    print(f"평균 Sources Detail 생성률: {avg_metrics['avg_detail_rate']:.1f}%")
    print(f"평균 답변 길이: {avg_metrics['avg_answer_length']:.0f}자")
    print(f"{'='*80}\n")


def _save_metrics(all_metrics: List[Dict[str, Any]], output_file: Path) -> None:
    """성능 지표 저장"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 성능 지표가 {output_file}에 저장되었습니다.")


async def main():
    """메인 함수"""
    test_queries = [
        "전세금 반환 보증에 대해 설명해주세요",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    
    all_metrics = []
    
    for query in test_queries:
        try:
            metrics = await test_query_and_collect_metrics(query)
            all_metrics.append(metrics)
            _print_metrics(metrics, query)
        except Exception as e:
            print(f"❌ 오류 발생 ({query}): {e}")
            import traceback
            traceback.print_exc()
    
    output_file = project_root / "data" / "ml_metrics" / "test_performance_metrics.json"
    _save_metrics(all_metrics, output_file)
    
    if all_metrics:
        avg_metrics = _calculate_average_metrics(all_metrics)
        _print_average_metrics(avg_metrics)

if __name__ == "__main__":
    asyncio.run(main())

