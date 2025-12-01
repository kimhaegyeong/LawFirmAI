#!/usr/bin/env python3
"""
검색된 문서의 타입 확인 스크립트
"""

import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

from core.workflow.workflow_service import LangGraphWorkflowService
from config.langgraph_config import LangGraphConfig

async def check_source_types():
    """검색된 문서의 타입 확인"""
    print("="*80)
    print("검색된 문서 타입 확인")
    print("="*80)
    
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)
    
    query = "계약서 작성 시 주의사항은 무엇인가요?"
    print(f"\n[질의]")
    print(f"  {query}")
    
    result = await service.process_query(query)
    
    retrieved_docs = result.get("retrieved_docs", [])
    sources = result.get("sources", [])
    
    print(f"\n[검색 결과]")
    print(f"  retrieved_docs 개수: {len(retrieved_docs)}")
    print(f"  sources 개수: {len(sources)}")
    
    # retrieved_docs의 타입 분포 확인
    print(f"\n[retrieved_docs 타입 분포]")
    type_counts = {}
    for i, doc in enumerate(retrieved_docs[:10], 1):
        if isinstance(doc, dict):
            doc_type = (
                doc.get("type") or
                doc.get("source_type") or
                (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else "") or
                "unknown"
            )
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            source_name = (
                doc.get("source") or
                doc.get("title") or
                doc.get("document_id") or
                "N/A"
            )
            
            print(f"  [{i}] type={doc_type}, source={source_name[:50]}...")
            if "metadata" in doc:
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    print(f"      metadata.source_type={metadata.get('source_type', 'N/A')}")
    
    print(f"\n  타입별 개수:")
    for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {doc_type}: {count}개")
    
    # sources의 타입 분포 확인
    print(f"\n[sources 타입 분포]")
    source_type_counts = {}
    for i, source in enumerate(sources, 1):
        source_type = source.get("type", "unknown")
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        source_name = source.get("source", "N/A")
        print(f"  [{i}] type={source_type}, source={source_name[:50]}...")
    
    print(f"\n  타입별 개수:")
    for source_type, count in sorted(source_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {source_type}: {count}개")
    
    # 판례, 해석례, 결정례 포함 여부 확인
    print(f"\n[법률 문서 타입 포함 여부]")
    has_law = any(s.get("type") == "law" for s in sources)
    has_precedent = any(s.get("type") == "precedent" for s in sources)
    has_decision = any(s.get("type") == "decision" for s in sources)
    has_interpretation = any(s.get("type") == "interpretation" for s in sources)
    
    print(f"  법령: {'✅' if has_law else '❌'}")
    print(f"  판례: {'✅' if has_precedent else '❌'}")
    print(f"  결정례: {'✅' if has_decision else '❌'}")
    print(f"  해석례: {'✅' if has_interpretation else '❌'}")
    
    # retrieved_docs에서도 확인
    print(f"\n[retrieved_docs에서 법률 문서 타입 포함 여부]")
    has_law_doc = any(
        doc.get("type") in ["statute_article", "law"] or
        doc.get("source_type") in ["statute_article", "law"] or
        (isinstance(doc.get("metadata"), dict) and doc.get("metadata", {}).get("source_type") in ["statute_article", "law"])
        for doc in retrieved_docs
    )
    has_precedent_doc = any(
        doc.get("type") in ["case_paragraph", "precedent"] or
        doc.get("source_type") in ["case_paragraph", "precedent"] or
        (isinstance(doc.get("metadata"), dict) and doc.get("metadata", {}).get("source_type") in ["case_paragraph", "precedent"])
        for doc in retrieved_docs
    )
    has_decision_doc = any(
        doc.get("type") in ["decision_paragraph", "decision"] or
        doc.get("source_type") in ["decision_paragraph", "decision"] or
        (isinstance(doc.get("metadata"), dict) and doc.get("metadata", {}).get("source_type") in ["decision_paragraph", "decision"])
        for doc in retrieved_docs
    )
    has_interpretation_doc = any(
        doc.get("type") in ["interpretation_paragraph", "interpretation"] or
        doc.get("source_type") in ["interpretation_paragraph", "interpretation"] or
        (isinstance(doc.get("metadata"), dict) and doc.get("metadata", {}).get("source_type") in ["interpretation_paragraph", "interpretation"])
        for doc in retrieved_docs
    )
    
    print(f"  법령: {'✅' if has_law_doc else '❌'}")
    print(f"  판례: {'✅' if has_precedent_doc else '❌'}")
    print(f"  결정례: {'✅' if has_decision_doc else '❌'}")
    print(f"  해석례: {'✅' if has_interpretation_doc else '❌'}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(check_source_types())

