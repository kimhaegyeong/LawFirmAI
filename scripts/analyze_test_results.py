# -*- coding: utf-8 -*-
"""
테스트 결과 분석 스크립트
run_query_test.py 실행 후 로그 파일을 분석하여 성능 개선 효과를 확인
"""

import os
import re
from pathlib import Path
from datetime import datetime

def find_latest_log_file():
    """최신 로그 파일 찾기"""
    log_dir = Path("logs/test")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("run_query_test_*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=lambda f: f.stat().st_mtime)

def analyze_log_file(log_file_path):
    """로그 파일 분석"""
    if not log_file_path or not log_file_path.exists():
        print("❌ 로그 파일을 찾을 수 없습니다.")
        return None
    
    print(f"📝 로그 파일 분석: {log_file_path}")
    print("=" * 80)
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {
        'performance': {},
        'keyword_coverage': [],
        'metadata_typos': [],
        'semantic_skipping': []
    }
    
    # 성능 메트릭 추출
    perf_pattern = r'process_search_results_combined가\s+([\d.]+)초|expand_keywords가\s+([\d.]+)초'
    perf_matches = re.findall(perf_pattern, content)
    for match in perf_matches:
        if match[0]:
            results['performance']['process_search_results_combined'] = float(match[0])
        if match[1]:
            results['performance']['expand_keywords'] = float(match[1])
    
    # Keyword Coverage 추출
    coverage_pattern = r'Keyword Coverage[:\s]+([\d.]+)'
    coverage_matches = re.findall(coverage_pattern, content)
    results['keyword_coverage'] = [float(c) for c in coverage_matches]
    
    # 메타데이터 오타 정규화 확인
    typo_pattern = r'(Normalized typo|Fixed typo).*interpretation_id'
    typo_matches = re.findall(typo_pattern, content)
    results['metadata_typos'] = typo_matches
    
    # 의미 기반 매칭 생략 확인
    skip_pattern = r'Skipping semantic matching.*coverage already high.*([\d.]+)'
    skip_matches = re.findall(skip_pattern, content)
    results['semantic_skipping'] = skip_matches
    
    # Missing required fields 확인
    missing_pattern = r'Missing required fields.*interpretation_id'
    missing_matches = re.findall(missing_pattern, content)
    results['metadata_typos'].extend([f"Missing: {m}" for m in missing_matches])
    
    return results

def print_results(results):
    """결과 출력"""
    if not results:
        return
    
    print("\n📊 성능 메트릭:")
    print("-" * 80)
    
    # process_search_results_combined
    if 'process_search_results_combined' in results['performance']:
        time = results['performance']['process_search_results_combined']
        target = 5.0
        improvement = ((15.82 - time) / 15.82) * 100
        status = "✅" if time <= target else "⚠️"
        print(f"{status} process_search_results_combined: {time:.2f}초 (목표: {target}초 이하, 개선: {improvement:.1f}%)")
    else:
        print("⚠️  process_search_results_combined 실행 시간을 찾을 수 없습니다.")
    
    # expand_keywords
    if 'expand_keywords' in results['performance']:
        time = results['performance']['expand_keywords']
        target = 5.0
        improvement = ((8.18 - time) / 8.18) * 100
        status = "✅" if time <= target else "⚠️"
        print(f"{status} expand_keywords: {time:.2f}초 (목표: {target}초 이하, 개선: {improvement:.1f}%)")
    else:
        print("⚠️  expand_keywords 실행 시간을 찾을 수 없습니다.")
    
    # Keyword Coverage
    print("\n📈 Keyword Coverage:")
    print("-" * 80)
    if results['keyword_coverage']:
        avg_coverage = sum(results['keyword_coverage']) / len(results['keyword_coverage'])
        max_coverage = max(results['keyword_coverage'])
        min_coverage = min(results['keyword_coverage'])
        status = "✅" if avg_coverage >= 0.70 else "⚠️"
        print(f"{status} 평균: {avg_coverage:.3f}, 최대: {max_coverage:.3f}, 최소: {min_coverage:.3f} (목표: 0.70 이상)")
        print(f"   측정 횟수: {len(results['keyword_coverage'])}회")
    else:
        print("⚠️  Keyword Coverage 데이터를 찾을 수 없습니다.")
    
    # 메타데이터 오타 정규화
    print("\n🔧 메타데이터 오타 정규화:")
    print("-" * 80)
    if results['metadata_typos']:
        normalized_count = len([t for t in results['metadata_typos'] if 'Normalized' in t or 'Fixed' in t])
        missing_count = len([t for t in results['metadata_typos'] if 'Missing' in t])
        print(f"✅ 정규화된 오타: {normalized_count}건")
        if missing_count > 0:
            print(f"⚠️  여전히 누락된 필드: {missing_count}건")
    else:
        print("ℹ️  메타데이터 오타 관련 로그를 찾을 수 없습니다.")
    
    # 의미 기반 매칭 생략
    print("\n⚡ 의미 기반 매칭 최적화:")
    print("-" * 80)
    if results['semantic_skipping']:
        print(f"✅ 의미 기반 매칭 생략 발생: {len(results['semantic_skipping'])}회")
        print(f"   Coverage: {', '.join(results['semantic_skipping'])}")
    else:
        print("ℹ️  의미 기반 매칭 생략 로그를 찾을 수 없습니다.")
    
    print("\n" + "=" * 80)

def main():
    """메인 함수"""
    print("🔍 테스트 결과 분석 시작")
    print("=" * 80)
    
    # 최신 로그 파일 찾기
    log_file = find_latest_log_file()
    
    if not log_file:
        print("❌ 로그 파일을 찾을 수 없습니다.")
        print("   먼저 다음 명령으로 테스트를 실행하세요:")
        print("   python lawfirm_langgraph/tests/scripts/run_query_test.py \"계약 해지 사유에 대해 알려주세요\"")
        return
    
    # 로그 파일 분석
    results = analyze_log_file(log_file)
    
    # 결과 출력
    print_results(results)
    
    # 검증 체크리스트
    print("\n✅ 검증 체크리스트:")
    print("-" * 80)
    
    if results:
        checks = []
        
        # 성능 검증
        if 'process_search_results_combined' in results['performance']:
            checks.append(("process_search_results_combined ≤ 5초", 
                          results['performance']['process_search_results_combined'] <= 5.0))
        
        if 'expand_keywords' in results['performance']:
            checks.append(("expand_keywords ≤ 5초", 
                          results['performance']['expand_keywords'] <= 5.0))
        
        if results['keyword_coverage']:
            avg = sum(results['keyword_coverage']) / len(results['keyword_coverage'])
            checks.append(("Keyword Coverage ≥ 0.70", avg >= 0.70))
        
        if results['semantic_skipping']:
            checks.append(("의미 기반 매칭 생략 작동", True))
        
        if results['metadata_typos']:
            normalized = len([t for t in results['metadata_typos'] if 'Normalized' in t or 'Fixed' in t])
            checks.append(("메타데이터 오타 정규화 작동", normalized > 0))
        
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")

if __name__ == "__main__":
    main()

