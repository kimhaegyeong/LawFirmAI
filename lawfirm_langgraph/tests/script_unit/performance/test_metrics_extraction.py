# -*- coding: utf-8 -*-
"""
메트릭 추출 테스트 스크립트

단일 쿼리로 메트릭 추출이 정상 작동하는지 확인합니다.
"""

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트 경로
script_dir = Path(__file__).parent
performance_dir = script_dir.parent
unit_dir = performance_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent
sys.path.insert(0, str(lawfirm_langgraph_dir))
sys.path.insert(0, str(project_root))

# 직접 import (경로 문제 해결)
import importlib.util
runners_dir = tests_dir / "runners"
validate_script = runners_dir / "validate_weight_configurations.py"

spec = importlib.util.spec_from_file_location(
    "validate_weight_configurations",
    validate_script
)
validate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_module)

QueryTestRunner = validate_module.QueryTestRunner
MetricsExtractor = validate_module.MetricsExtractor

async def test_single_query():
    """단일 쿼리로 메트릭 추출 테스트"""
    query = "계약 위약금에 대해 설명해주세요"
    
    print("="*80)
    print("메트릭 추출 테스트")
    print("="*80)
    print(f"\n테스트 쿼리: {query}\n")
    
    # 테스트 실행
    runner = QueryTestRunner()
    print("1. 쿼리 테스트 실행 중...")
    stdout, stderr = await runner.run_test(query)
    
    print(f"\n2. 출력 확인:")
    print(f"   stdout 길이: {len(stdout)}자")
    print(f"   stderr 길이: {len(stderr)}자")
    
    # 로깅은 stderr로 출력되므로 stderr도 포함
    combined_output = stdout + "\n" + stderr if stderr else stdout
    print(f"   combined 길이: {len(combined_output)}자")
    
    if combined_output:
        print(f"\n3. combined 출력 샘플 (마지막 1000자):")
        print("-"*80)
        print(combined_output[-1000:])
        print("-"*80)
    
    # 메트릭 추출
    print("\n4. 메트릭 추출 중...")
    extractor = MetricsExtractor()
    metrics = extractor.extract_metrics(combined_output, query)
    
    print("\n5. 추출된 메트릭:")
    print(f"   - 답변 품질 점수: {metrics.answer_quality_score}/100")
    print(f"   - 답변 길이: {metrics.answer_length}자")
    print(f"   - 검색된 문서 수: {metrics.retrieved_docs_count}개")
    print(f"   - 실제 사용 문서 수: {metrics.used_docs_count}개")
    print(f"   - 문서 활용률: {metrics.document_utilization_rate:.2%}")
    print(f"   - 평균 관련성 점수: {metrics.avg_relevance_score:.3f}")
    print(f"   - 소스 개수: {metrics.source_count}개")
    print(f"   - 소스 존재 여부: {metrics.has_sources}")
    print(f"   - 총 소요 시간: {metrics.total_time:.2f}초")
    print(f"   - 종합 점수: {metrics.overall_score:.3f}")
    
    # 패턴 매칭 확인
    print("\n6. 패턴 매칭 확인:")
    import re
    
    patterns = {
        "품질 점수": r'품질 점수:\s*(\d+)/100',
        "답변 길이": r'답변\s*\((\d+)자\)',
        "검색된 참고자료": r'검색된 참고자료.*?\((\d+)개\)',
        "소스": r'소스 \(sources\)\s*\((\d+)개\)',
        "처리 시간": r'처리 시간:\s*([\d.]+)초',
        "유사도 점수": r'평균=([\d.]+)',
        "score=": r'score=([\d.]+)'
    }
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, combined_output)
        if matches:
            print(f"   ✅ {name}: {len(matches)}개 매칭 - {matches[:3]}")
        else:
            print(f"   ❌ {name}: 매칭 없음")
    
    return metrics

if __name__ == "__main__":
    metrics = asyncio.run(test_single_query())
    print("\n" + "="*80)
    print("테스트 완료")
    print("="*80)

