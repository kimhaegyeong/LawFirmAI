# -*- coding: utf-8 -*-
"""검색 결과 추출 디버깅 스크립트"""

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.scripts.test_search_quality_evaluation import SearchQualityEvaluator

async def debug_retrieved_docs():
    """retrieved_docs 추출 디버깅"""
    print("=" * 60)
    print("검색 결과 추출 디버깅")
    print("=" * 60)
    
    evaluator = SearchQualityEvaluator(enable_improvements=True)
    
    query = "민법 제1조의 내용은 무엇인가요?"
    print(f"\n테스트 쿼리: {query}")
    
    # process_query 실행
    result = await evaluator.workflow_service.process_query(
        query=query,
        session_id="debug_session",
        enable_checkpoint=False
    )
    
    print(f"\n✅ process_query 완료")
    print(f"결과 키: {list(result.keys())}")
    
    # retrieved_docs 확인
    retrieved_docs = result.get("retrieved_docs", [])
    print(f"\nretrieved_docs:")
    print(f"  - 타입: {type(retrieved_docs)}")
    print(f"  - 개수: {len(retrieved_docs)}")
    
    if retrieved_docs:
        print(f"  - 첫 번째 문서 키: {list(retrieved_docs[0].keys()) if retrieved_docs else 'None'}")
    else:
        print("  - ⚠️ retrieved_docs가 비어있습니다!")
    
    # search 그룹 확인
    if "search" in result:
        search_group = result["search"]
        print(f"\nsearch 그룹:")
        print(f"  - 타입: {type(search_group)}")
        if isinstance(search_group, dict):
            print(f"  - 키: {list(search_group.keys())}")
            if "retrieved_docs" in search_group:
                search_docs = search_group["retrieved_docs"]
                print(f"  - search.retrieved_docs 개수: {len(search_docs) if isinstance(search_docs, list) else 'N/A'}")
            if "merged_documents" in search_group:
                merged_docs = search_group["merged_documents"]
                print(f"  - search.merged_documents 개수: {len(merged_docs) if isinstance(merged_docs, list) else 'N/A'}")
    
    # extracted_keywords 확인
    extracted_keywords = result.get("extracted_keywords", [])
    print(f"\nextracted_keywords:")
    print(f"  - 타입: {type(extracted_keywords)}")
    print(f"  - 개수: {len(extracted_keywords) if isinstance(extracted_keywords, list) else 'N/A'}")
    
    # evaluate_query_async로 평가
    print(f"\n" + "=" * 60)
    print("evaluate_query_async 실행")
    print("=" * 60)
    
    metrics = await evaluator.evaluate_query_async(
        query=query,
        query_type="statute_article"
    )
    
    print(f"\n평가 결과:")
    print(f"  - result_count: {metrics.get('result_count')}")
    print(f"  - response_time: {metrics.get('response_time'):.2f}s")
    print(f"  - precision_at_5: {metrics.get('precision_at_5')}")
    print(f"  - keyword_coverage: {metrics.get('keyword_coverage')}")
    print(f"  - diversity_score: {metrics.get('diversity_score')}")

if __name__ == "__main__":
    asyncio.run(debug_retrieved_docs())

