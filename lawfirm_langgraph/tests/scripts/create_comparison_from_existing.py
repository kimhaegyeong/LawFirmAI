# -*- coding: utf-8 -*-
"""
기존 평가 결과를 활용한 비교 리포트 생성 스크립트
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_dir))

from tests.scripts.compare_search_quality import compare_results, generate_report

def main():
    """기존 결과 파일을 활용한 비교 리포트 생성"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comparison report from existing results")
    parser.add_argument(
        "--before-file",
        type=str,
        required=True,
        help="Before results JSON file path"
    )
    parser.add_argument(
        "--after-file",
        type=str,
        required=True,
        help="After results JSON file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/search_quality_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 파일 로드
    print(f"Loading before results from: {args.before_file}")
    with open(args.before_file, 'r', encoding='utf-8') as f:
        before_results = json.load(f)
    
    print(f"Loading after results from: {args.after_file}")
    with open(args.after_file, 'r', encoding='utf-8') as f:
        after_results = json.load(f)
    
    # 비교 수행
    print("Comparing results...")
    comparison = compare_results(before_results, after_results)
    
    # 결과 저장
    comparison_output = output_dir / "comparison.json"
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Comparison saved: {comparison_output}")
    
    # 리포트 생성
    report_output = output_dir / "comparison_report.md"
    generate_report(comparison, report_output)
    print(f"Report generated: {report_output}")
    
    # 주요 개선 사항 출력
    print("\n주요 개선 사항:")
    for metric_key, metric_data in comparison["improvements"].items():
        metric_name = metric_key.replace("avg_", "").replace("_", " ").title()
        improvement_pct = metric_data['improvement_pct']
        print(f"  - {metric_name}: {improvement_pct:+.2f}%")

if __name__ == "__main__":
    main()

