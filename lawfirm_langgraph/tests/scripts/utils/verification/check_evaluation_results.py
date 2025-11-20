# -*- coding: utf-8 -*-
"""평가 결과 확인 스크립트"""

import json
from pathlib import Path

# 평가 결과 파일 확인
result_file = Path("logs/search_quality_evaluation_with_improvements_fixed.json")

if result_file.exists():
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    print("=" * 60)
    print("평가 결과 요약")
    print("=" * 60)
    print(f"총 쿼리: {result.get('total_queries', 0)}")
    print(f"성공: {result.get('successful_queries', 0)}")
    print(f"실패: {result.get('failed_queries', 0)}")
    
    print(f"\n평균 메트릭:")
    avg = result.get('average_metrics', {})
    print(f"  - avg_result_count: {avg.get('avg_result_count', 0):.4f}")
    print(f"  - avg_keyword_coverage: {avg.get('avg_keyword_coverage', 0):.4f}")
    print(f"  - avg_diversity_score: {avg.get('avg_diversity_score', 0):.4f}")
    print(f"  - avg_avg_relevance: {avg.get('avg_avg_relevance', 0):.4f}")
    
    print(f"\n첫 번째 쿼리 상세:")
    if result.get('detailed_metrics'):
        first = result['detailed_metrics'][0]
        print(f"  - query: {first.get('query', 'N/A')[:50]}...")
        print(f"  - result_count: {first.get('result_count', 0)}")
        print(f"  - keyword_coverage: {first.get('keyword_coverage', 0):.4f}")
        print(f"  - diversity_score: {first.get('diversity_score', 0):.4f}")
        print(f"  - avg_relevance: {first.get('avg_relevance', 0):.4f}")
        print(f"  - precision_at_5: {first.get('precision_at_5', 0):.4f}")
        print(f"  - recall_at_5: {first.get('recall_at_5', 0):.4f}")
        
        # 메트릭이 0인지 확인
        if first.get('result_count', 0) == 0:
            print(f"\n⚠️  경고: result_count가 0입니다!")
            print(f"   이는 retrieved_docs가 제대로 추출되지 않았음을 의미합니다.")
        else:
            print(f"\n✅ result_count가 정상입니다: {first.get('result_count', 0)}")
else:
    print("평가 결과 파일이 없습니다.")
    print(f"경로: {result_file.absolute()}")

