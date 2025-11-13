# -*- coding: utf-8 -*-
"""
로그 분석 및 개선 사항 식별 스크립트
각 단계별 로그를 분석하여 개선 사항을 식별
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

def analyze_sources_conversion_logs(log_content: str) -> Dict[str, Any]:
    """Sources 변환 관련 로그 분석"""
    analysis = {
        "conversion_statistics": [],
        "fallback_usage": [],
        "critical_fallbacks": [],
        "lost_documents": [],
        "total_conversions": 0,
        "total_docs": 0,
        "total_failed": 0,
    }
    
    # Conversion statistics 패턴
    pattern = r'\[SOURCES\] 📊 Conversion statistics: (\d+)/(\d+) docs converted \(([\d.]+)%\), failed: (\d+)'
    matches = re.findall(pattern, log_content)
    for match in matches:
        created, total, rate, failed = match
        analysis["conversion_statistics"].append({
            "created": int(created),
            "total": int(total),
            "rate": float(rate),
            "failed": int(failed)
        })
        analysis["total_conversions"] += int(created)
        analysis["total_docs"] += int(total)
        analysis["total_failed"] += int(failed)
    
    # Fallback source 생성 패턴
    fallback_pattern = r'\[SOURCES\] ✅ Generated fallback source for doc (\d+)/(\d+): (.+)'
    fallback_matches = re.findall(fallback_pattern, log_content)
    for match in fallback_matches:
        analysis["fallback_usage"].append({
            "doc_index": int(match[0]),
            "total_docs": int(match[1]),
            "source": match[2]
        })
    
    # Critical fallback 패턴
    critical_pattern = r'\[SOURCES\] ⚠️ CRITICAL: Using final fallback for doc (\d+)/(\d+): (.+)'
    critical_matches = re.findall(critical_pattern, log_content)
    for match in critical_matches:
        analysis["critical_fallbacks"].append({
            "doc_index": int(match[0]),
            "total_docs": int(match[1]),
            "source": match[2]
        })
    
    # Lost documents 패턴
    lost_pattern = r'\[SOURCES\] ⚠️ Lost document.*?doc_index=(\d+).*?type=([^,]+)'
    lost_matches = re.findall(lost_pattern, log_content)
    for match in lost_matches:
        analysis["lost_documents"].append({
            "doc_index": int(match[0]),
            "type": match[1]
        })
    
    return analysis

def analyze_legal_references_logs(log_content: str) -> Dict[str, Any]:
    """Legal References 관련 로그 분석"""
    analysis = {
        "extracted_from_sources": 0,
        "extracted_from_content": 0,
        "extracted_from_docs": 0,
        "total_extracted": 0,
        "legal_references": []
    }
    
    # Legal references 추출 패턴
    pattern = r'\[LEGAL_REFS\] Extracted (\d+) legal references'
    matches = re.findall(pattern, log_content)
    for match in matches:
        analysis["total_extracted"] += int(match)
    
    # Sources에서 추출
    sources_pattern = r'\[LEGAL_REFS\] From sources_detail: (\d+) references'
    sources_matches = re.findall(sources_pattern, log_content)
    for match in sources_matches:
        analysis["extracted_from_sources"] += int(match)
    
    # Content에서 추출
    content_pattern = r'\[LEGAL_REFS\] From content: (\d+) references'
    content_matches = re.findall(content_pattern, log_content)
    for match in content_matches:
        analysis["extracted_from_content"] += int(match)
    
    # Docs에서 추출
    docs_pattern = r'\[LEGAL_REFS\] From retrieved_docs: (\d+) references'
    docs_matches = re.findall(docs_pattern, log_content)
    for match in docs_matches:
        analysis["extracted_from_docs"] += int(match)
    
    return analysis

def analyze_answer_length_logs(log_content: str) -> Dict[str, Any]:
    """답변 길이 관련 로그 분석"""
    analysis = {
        "length_warnings": [],
        "length_adjustments": [],
        "too_short_count": 0,
        "too_long_count": 0,
        "adjusted_count": 0,
    }
    
    # 너무 짧은 경우
    short_pattern = r'\[ANSWER LENGTH\] ⚠️ Too short: (\d+) \(target: (\d+)-(\d+)\)'
    short_matches = re.findall(short_pattern, log_content)
    for match in short_matches:
        analysis["length_warnings"].append({
            "current": int(match[0]),
            "min_target": int(match[1]),
            "max_target": int(match[2])
        })
        analysis["too_short_count"] += 1
    
    # 너무 긴 경우
    long_pattern = r'\[ANSWER LENGTH\] Too long: (\d+), adjusting to max (\d+)'
    long_matches = re.findall(long_pattern, log_content)
    for match in long_matches:
        analysis["length_adjustments"].append({
            "original": int(match[0]),
            "max": int(match[1])
        })
        analysis["too_long_count"] += 1
        analysis["adjusted_count"] += 1
    
    return analysis

def analyze_context_usage_logs(log_content: str) -> Dict[str, Any]:
    """Context Usage 관련 로그 분석"""
    analysis = {
        "coverage_scores": [],
        "relevance_scores": [],
        "average_coverage": 0.0,
        "average_relevance": 0.0,
    }
    
    # Coverage 점수 패턴
    coverage_pattern = r'\[COVERAGE\] Coverage score: ([\d.]+)'
    coverage_matches = re.findall(coverage_pattern, log_content)
    for match in coverage_matches:
        score = float(match)
        analysis["coverage_scores"].append(score)
    
    # Relevance 점수 패턴
    relevance_pattern = r'\[RELEVANCE\] Relevance score: ([\d.]+)'
    relevance_matches = re.findall(relevance_pattern, log_content)
    for match in relevance_matches:
        score = float(match)
        analysis["relevance_scores"].append(score)
    
    if analysis["coverage_scores"]:
        analysis["average_coverage"] = sum(analysis["coverage_scores"]) / len(analysis["coverage_scores"])
    
    if analysis["relevance_scores"]:
        analysis["average_relevance"] = sum(analysis["relevance_scores"]) / len(analysis["relevance_scores"])
    
    return analysis

def identify_improvements(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """분석 결과를 바탕으로 개선 사항 식별"""
    improvements = []
    
    # Sources 변환률 개선
    sources_analysis = analysis_results.get("sources", {})
    if sources_analysis.get("total_docs", 0) > 0:
        avg_conversion_rate = (sources_analysis.get("total_conversions", 0) / 
                              sources_analysis.get("total_docs", 1)) * 100
        if avg_conversion_rate < 90:
            improvements.append({
                "category": "Sources 변환률",
                "priority": "HIGH",
                "current": f"{avg_conversion_rate:.1f}%",
                "target": "90% 이상",
                "description": f"현재 변환률이 {avg_conversion_rate:.1f}%로 목표(90%) 미만입니다.",
                "recommendation": "fallback 로직 강화, source_type 추론 개선 필요"
            })
        
        if sources_analysis.get("critical_fallbacks"):
            improvements.append({
                "category": "Critical Fallback 사용",
                "priority": "MEDIUM",
                "current": f"{len(sources_analysis['critical_fallbacks'])}건",
                "target": "0건",
                "description": f"최종 fallback이 {len(sources_analysis['critical_fallbacks'])}건 사용되었습니다.",
                "recommendation": "metadata 추출 로직 개선, content 기반 추론 강화"
            })
    
    # Legal References 생성률 개선
    legal_analysis = analysis_results.get("legal_references", {})
    if legal_analysis.get("total_extracted", 0) == 0:
        improvements.append({
            "category": "Legal References 생성",
            "priority": "HIGH",
            "current": "0개",
            "target": "statute_article 문서 수만큼",
            "description": "Legal references가 생성되지 않았습니다.",
            "recommendation": "legal_references 추출 로직 검증 및 개선 필요"
        })
    
    # 답변 길이 개선
    length_analysis = analysis_results.get("answer_length", {})
    if length_analysis.get("too_short_count", 0) > 0:
        improvements.append({
            "category": "답변 길이",
            "priority": "MEDIUM",
            "current": f"{length_analysis['too_short_count']}건 너무 짧음",
            "target": "모든 답변이 최소 길이 이상",
            "description": f"{length_analysis['too_short_count']}건의 답변이 최소 길이 미만입니다.",
            "recommendation": "프롬프트 개선, 컨텍스트 활용 강화"
        })
    
    # Context Usage 개선
    context_analysis = analysis_results.get("context_usage", {})
    avg_coverage = context_analysis.get("average_coverage", 0.0)
    if avg_coverage < 0.8:
        improvements.append({
            "category": "Context Usage",
            "priority": "MEDIUM",
            "current": f"{avg_coverage:.2f}",
            "target": "0.80 이상",
            "description": f"평균 coverage가 {avg_coverage:.2f}로 목표(0.80) 미만입니다.",
            "recommendation": "프롬프트 개선, 검색 결과 품질 향상"
        })
    
    return improvements

def main():
    """메인 함수"""
    # 프로젝트 루트 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 로그 파일 경로 (프로젝트 루트 기준)
    log_file = project_root / "logs" / "lawfirm_ai.log"
    
    if not log_file.exists():
        print(f"⚠️  로그 파일을 찾을 수 없습니다: {log_file}")
        print("실제 로그 파일 경로를 확인하거나 테스트를 실행하여 로그를 생성하세요.")
        return
    
    # 로그 파일 읽기
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
    
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
    output = {
        "analysis_results": analysis_results,
        "improvements": improvements,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = project_root / "data" / "ml_metrics" / "log_analysis_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 분석 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()

