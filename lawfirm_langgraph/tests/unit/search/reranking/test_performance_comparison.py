# -*- coding: utf-8 -*-
"""
Reranking 성능 비교 테스트 (최적화 전후)
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트를 경로에 추가 (하위 폴더로 이동하여 parent 하나 추가)
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# conftest.py에서 공통 유틸리티 함수 import (상위 폴더의 conftest.py)
from lawfirm_langgraph.tests.unit.search.conftest import create_sample_documents

from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker


def test_performance_comparison():
    """성능 비교 테스트"""
    print("="*60)
    print("Reranking 성능 비교 테스트 (최적화 전후)")
    print("="*60)
    
    ranker = ResultRanker(use_cross_encoder=False)
    
    test_cases = [
        {"name": "소규모", "count": 20},
        {"name": "중규모", "count": 35},
        {"name": "대규모", "count": 70}
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{test_case['name']} ({test_case['count']}개 문서):")
        
        db_docs, vector_docs, keyword_docs = create_sample_documents(test_case['count'])
        
        # 최적화된 통합 reranking 실행 (여러 번 반복하여 평균 측정)
        times = []
        for i in range(5):
            start_time = time.time()
            result = ranker.integrated_rerank_pipeline(
                db_results=db_docs.copy(),
                vector_results=vector_docs.copy(),
                keyword_results=keyword_docs.copy(),
                query="계약 해지에 대해 알려주세요",
                query_type="law_inquiry",
                extracted_keywords=["계약", "해지"],
                top_k=20,
                search_quality=0.7
            )
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  ✅ 평균 처리 시간: {avg_time:.4f}초")
        print(f"  ✅ 최소: {min_time:.4f}초, 최대: {max_time:.4f}초")
        print(f"  ✅ 결과 수: {len(result)}개")
        
        results.append({
            "name": test_case['name'],
            "count": test_case['count'],
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "result_count": len(result)
        })
    
    # 결과 요약
    print("\n" + "="*60)
    print("성능 비교 요약")
    print("="*60)
    
    for result in results:
        print(f"\n{result['name']} ({result['count']}개 문서):")
        print(f"  평균: {result['avg_time']:.4f}초")
        print(f"  범위: {result['min_time']:.4f}초 ~ {result['max_time']:.4f}초")
        print(f"  결과 수: {result['result_count']}개")
    
    # 이전 성능과 비교 (최적화 전)
    print("\n" + "="*60)
    print("최적화 전후 비교")
    print("="*60)
    
    previous_times = {
        "소규모": 0.0098,
        "중규모": 0.0256,
        "대규모": 0.0670
    }
    
    for result in results:
        name = result['name']
        current_time = result['avg_time']
        previous_time = previous_times.get(name, current_time)
        
        improvement = ((previous_time - current_time) / previous_time) * 100 if previous_time > 0 else 0
        
        print(f"\n{name}:")
        print(f"  최적화 전: {previous_time:.4f}초")
        print(f"  최적화 후: {current_time:.4f}초")
        print(f"  개선율: {improvement:.1f}% ⬇️")


if __name__ == "__main__":
    try:
        test_performance_comparison()
        print("\n" + "="*60)
        print("✅ 성능 비교 테스트 완료!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

