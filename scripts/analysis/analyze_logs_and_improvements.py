# -*- coding: utf-8 -*-
"""
로그 분석 및 개선 사항 식별 스크립트
각 단계별 로그를 분석하여 개선 사항을 식별
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

from scripts.utils.log_analyzer import (
    analyze_sources_conversion_logs,
    analyze_legal_references_logs,
    analyze_answer_length_logs,
    analyze_context_usage_logs,
    identify_improvements
)


def main():
    """메인 함수"""
    # 프로젝트 루트 경로 설정
    from scripts.utils.path_utils import setup_project_path, get_project_root
    from scripts.utils.file_utils import load_json_file, save_json_file
    
    project_root = setup_project_path()
    
    # 로그 파일 경로 (프로젝트 루트 기준)
    log_file = project_root / "logs" / "lawfirm_ai.log"
    
    if not log_file.exists():
        print(f"⚠️  로그 파일을 찾을 수 없습니다: {log_file}")
        print("실제 로그 파일 경로를 확인하거나 테스트를 실행하여 로그를 생성하세요.")
        return
    
    # 로그 파일 읽기
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
    except Exception as e:
        print(f"⚠️  로그 파일 읽기 실패: {e}")
        return
    
    # 각 영역별 분석
    analysis_results = {
        "sources": analyze_sources_conversion_logs(log_content),
        "legal_references": analyze_legal_references_logs(log_content),
        "answer_length": analyze_answer_length_logs(log_content),
        "context_usage": analyze_context_usage_logs(log_content),
    }
    
    # 개선 사항 식별
    improvements = identify_improvements(analysis_results)
    
    # 결과 출력
    print("\n" + "="*80)
    print("로그 분석 결과")
    print("="*80)
    
    print("\n📊 Sources 변환 분석:")
    sources = analysis_results["sources"]
    if sources["total_docs"] > 0:
        avg_rate = (sources["total_conversions"] / sources["total_docs"]) * 100
        print(f"  - 평균 변환률: {avg_rate:.1f}%")
        print(f"  - 총 변환: {sources['total_conversions']}/{sources['total_docs']}")
        print(f"  - 실패: {sources['total_failed']}")
        print(f"  - Fallback 사용: {len(sources['fallback_usage'])}건")
        print(f"  - Critical Fallback: {len(sources['critical_fallbacks'])}건")
    
    print("\n⚖️  Legal References 분석:")
    legal = analysis_results["legal_references"]
    print(f"  - 총 추출: {legal['total_extracted']}개")
    print(f"  - Sources에서: {legal['extracted_from_sources']}개")
    print(f"  - Content에서: {legal['extracted_from_content']}개")
    print(f"  - Docs에서: {legal['extracted_from_docs']}개")
    
    print("\n📏 답변 길이 분석:")
    length = analysis_results["answer_length"]
    print(f"  - 너무 짧음: {length['too_short_count']}건")
    print(f"  - 너무 김: {length['too_long_count']}건")
    print(f"  - 조정됨: {length['adjusted_count']}건")
    
    print("\n📚 Context Usage 분석:")
    context = analysis_results["context_usage"]
    print(f"  - 평균 Coverage: {context['average_coverage']:.2f}")
    print(f"  - 평균 Relevance: {context['average_relevance']:.2f}")
    
    print("\n" + "="*80)
    print("개선 사항")
    print("="*80)
    
    if improvements:
        for i, improvement in enumerate(improvements, 1):
            print(f"\n{i}. [{improvement['priority']}] {improvement['category']}")
            print(f"   현재: {improvement['current']}")
            print(f"   목표: {improvement['target']}")
            print(f"   설명: {improvement['description']}")
            print(f"   권장사항: {improvement['recommendation']}")
    else:
        print("\n✅ 추가 개선 사항이 없습니다!")
    
    # 결과 저장
    from scripts.utils.file_utils import save_json_file
    
    output = {
        "analysis_results": analysis_results,
        "improvements": improvements,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = project_root / "data" / "ml_metrics" / "log_analysis_results.json"
    save_json_file(output, output_file)
    
    print(f"\n✅ 분석 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()

