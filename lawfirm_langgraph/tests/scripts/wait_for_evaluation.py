# -*- coding: utf-8 -*-
"""평가 완료 대기 스크립트"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

result_file = project_root / "logs" / "search_quality_evaluation_with_improvements_fixed.json"
log_file = project_root / "logs" / "evaluation_progress_fixed.log"

print("=" * 60)
print("평가 완료 대기 중...")
print("=" * 60)
print(f"결과 파일: {result_file}")
print(f"로그 파일: {log_file}")
print()

max_wait = 1800  # 30분
elapsed = 0
interval = 30  # 30초마다 확인

while elapsed < max_wait:
    if result_file.exists():
        print(f"\n✅ 평가 완료! 결과 파일이 생성되었습니다.")
        print(f"   파일: {result_file}")
        
        # 결과 요약
        try:
            import json
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"\n📊 평가 결과 요약:")
            print(f"   - 총 쿼리: {result.get('total_queries', 0)}")
            print(f"   - 성공: {result.get('successful_queries', 0)}")
            print(f"   - 실패: {result.get('failed_queries', 0)}")
            
            if result.get('average_metrics'):
                print(f"\n   평균 메트릭:")
                metrics = result['average_metrics']
                for key in ['avg_result_count', 'avg_keyword_coverage', 'avg_diversity_score', 'avg_avg_relevance']:
                    if key in metrics:
                        print(f"     - {key}: {metrics[key]:.4f}")
        except Exception as e:
            print(f"   ⚠️  결과 파일 읽기 실패: {e}")
        
        break
    
    minutes = elapsed // 60
    seconds = elapsed % 60
    print(f"[{minutes:02d}:{seconds:02d}] 평가 진행 중...", end='\r')
    
    time.sleep(interval)
    elapsed += interval

if not result_file.exists():
    print(f"\n⚠️  평가가 아직 완료되지 않았습니다.")
    print(f"   최대 대기 시간 ({max_wait // 60}분)을 초과했습니다.")
    print(f"   수동으로 확인해주세요:")
    print(f"   - 결과 파일: {result_file}")
    print(f"   - 로그 파일: {log_file}")

