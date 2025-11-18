# -*- coding: utf-8 -*-
"""
TODO 작업 자동 실행 스크립트
평가 완료 후 Before/After 비교 자동 실행
"""

import sys
import time
import subprocess
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("TODO 작업 자동 실행")
print("=" * 60)

# 1. 평가 완료 확인
result_file = project_root / "logs" / "search_quality_evaluation_with_improvements_fixed.json"

print(f"\n1️⃣ 평가 완료 확인 중...")
print(f"   결과 파일: {result_file}")

max_wait = 1800  # 30분
elapsed = 0
interval = 30  # 30초마다 확인

while elapsed < max_wait:
    if result_file.exists():
        print(f"\n✅ 평가 완료!")
        
        # 결과 요약
        try:
            import json
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"\n📊 평가 결과:")
            print(f"   - 총 쿼리: {result.get('total_queries', 0)}")
            print(f"   - 성공: {result.get('successful_queries', 0)}")
            print(f"   - 실패: {result.get('failed_queries', 0)}")
            
            metrics = result.get('average_metrics', {})
            if metrics:
                print(f"\n   주요 메트릭:")
                for key in ['avg_result_count', 'avg_keyword_coverage', 'avg_diversity_score', 'avg_avg_relevance']:
                    if key in metrics:
                        print(f"     - {key}: {metrics[key]:.4f}")
        except Exception as e:
            print(f"   ⚠️  결과 파일 읽기 실패: {e}")
        
        break
    
    minutes = elapsed // 60
    seconds = elapsed % 60
    print(f"   [{minutes:02d}:{seconds:02d}] 평가 진행 중...", end='\r')
    
    time.sleep(interval)
    elapsed += interval

if not result_file.exists():
    print(f"\n⚠️  평가가 아직 완료되지 않았습니다.")
    print(f"   최대 대기 시간 ({max_wait // 60}분)을 초과했습니다.")
    print(f"   수동으로 확인해주세요: {result_file}")
    sys.exit(1)

# 2. Before/After 비교 실행
print(f"\n" + "=" * 60)
print("2️⃣ Before/After 비교 실행")
print("=" * 60)

compare_script = project_root / "tests" / "scripts" / "compare_search_quality.py"

if compare_script.exists():
    print(f"\n비교 스크립트 실행 중...")
    print(f"   스크립트: {compare_script}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(compare_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print(f"\n✅ Before/After 비교 완료!")
            print(f"\n출력:")
            print(result.stdout)
        else:
            print(f"\n⚠️  비교 실행 중 오류 발생:")
            print(result.stderr)
    except Exception as e:
        print(f"\n❌ 비교 실행 실패: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n⚠️  비교 스크립트를 찾을 수 없습니다: {compare_script}")

print(f"\n" + "=" * 60)
print("TODO 작업 완료")
print("=" * 60)

