#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그 분석 유틸리티

워크플로우 로그를 분석하는 공통 함수들
"""

import re
from typing import Dict, List, Any


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

