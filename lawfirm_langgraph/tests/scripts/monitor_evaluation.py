# -*- coding: utf-8 -*-
"""
평가 진행 상황 모니터링 스크립트
"""

import sys
import time
import json
from pathlib import Path

log_file = Path("logs/evaluation_progress.log")
result_file = Path("logs/search_quality_evaluation_with_improvements.json")

print("=" * 60)
print("평가 진행 상황 모니터링")
print("=" * 60)
print(f"로그 파일: {log_file}")
print(f"결과 파일: {result_file}")
print()

if log_file.exists():
    print("최근 로그:")
    print("-" * 60)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
    print("-" * 60)
else:
    print("로그 파일이 아직 생성되지 않았습니다.")

if result_file.exists():
    print("\n결과 파일이 생성되었습니다!")
    print(f"결과 파일: {result_file}")
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"\n평가 완료!")
            print(f"- 총 쿼리 수: {data.get('total_queries', 0)}")
            print(f"- 성공한 쿼리: {data.get('successful_queries', 0)}")
            print(f"- 실패한 쿼리: {data.get('failed_queries', 0)}")
            if 'average_metrics' in data:
                print("\n평균 메트릭:")
                for key, value in data['average_metrics'].items():
                    if isinstance(value, float):
                        print(f"  - {key}: {value:.4f}")
    except Exception as e:
        print(f"결과 파일 읽기 오류: {e}")
else:
    print("\n평가가 아직 진행 중입니다...")

