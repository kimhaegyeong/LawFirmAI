# -*- coding: utf-8 -*-
"""평가 메트릭 0 문제 디버깅"""

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.scripts.test_search_quality_evaluation import SearchQualityEvaluator

async def debug_evaluation():
    """평가 문제 디버깅"""
    print("=" * 60)
    print("평가 메트릭 0 문제 디버깅")
    print("=" * 60)
    
    evaluator = SearchQualityEvaluator(enable_improvements=True)
    
    query = "민법 제1조의 내용은 무엇인가요?"
    print(f"\n테스트 쿼리: {query}")
    
    # 1. process_query 실행
    print(f"\n1. process_query 실행...")
    result = await evaluator.workflow_service.process_query(
        query=query,
        session_id="debug_session",
        enable_checkpoint=False
    )
    
    print(f"\n✅ process_query 완료")
    print(f"반환값 키: {list(result.keys())}")
    
    # 2. retrieved_docs 확인
    retrieved_docs = result.get("retrieved_docs", [])
    print(f"\n2. retrieved_docs 확인:")
    print(f"  - 타입: {type(retrieved_docs)}")
    print(f"  - 개수: {len(retrieved_docs)}")
    
    if not retrieved_docs:
        print(f"  ⚠️  retrieved_docs가 비어있습니다!")
        print(f"\n  대안 경로 확인:")
        
        # search 그룹 확인
        if "search" in result:
            search_group = result["search"]
            print(f"  - search 그룹 존재: {isinstance(search_group, dict)}")
            if isinstance(search_group, dict):
                print(f"    - search.retrieved_docs: {len(search_group.get('retrieved_docs', []))}")
                print(f"    - search.merged_documents: {len(search_group.get('merged_documents', []))}")
                print(f"    - search.semantic_results: {len(search_group.get('semantic_results', []))}")
        
        # common 그룹 확인
        if "common" in result:
            common = result["common"]
            if isinstance(common, dict) and "search" in common:
                search = common["search"]
                if isinstance(search, dict):
                    print(f"  - common.search.retrieved_docs: {len(search.get('retrieved_docs', []))}")
    
    # 3. extracted_keywords 확인
    extracted_keywords = result.get("extracted_keywords", [])
    if not extracted_keywords:
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            extracted_keywords = metadata.get("extracted_keywords", [])
    if not extracted_keywords:
        if "search" in result and isinstance(result["search"], dict):
            extracted_keywords = result["search"].get("extracted_keywords", [])
    
    print(f"\n3. extracted_keywords 확인:")
    print(f"  - 타입: {type(extracted_keywords)}")
    print(f"  - 개수: {len(extracted_keywords) if isinstance(extracted_keywords, list) else 'N/A'}")
    
    # 4. evaluate_query_async 실행
    print(f"\n4. evaluate_query_async 실행...")
    metrics = await evaluator.evaluate_query_async(
        query=query,
        query_type="statute_article"
    )
    
    print(f"\n✅ evaluate_query_async 완료")
    print(f"결과:")
    print(f"  - result_count: {metrics.get('result_count')}")
    print(f"  - keyword_coverage: {metrics.get('keyword_coverage')}")
    print(f"  - diversity_score: {metrics.get('diversity_score')}")
    print(f"  - avg_relevance: {metrics.get('avg_relevance')}")

if __name__ == "__main__":
    asyncio.run(debug_evaluation())

