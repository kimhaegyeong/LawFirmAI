#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_analyzer 모듈 단위 테스트
"""

import pytest

from scripts.utils.log_analyzer import (
    analyze_sources_conversion_logs,
    analyze_legal_references_logs,
    analyze_answer_length_logs,
    analyze_context_usage_logs,
    identify_improvements
)


class TestLogAnalyzer:
    """log_analyzer 모듈 테스트"""
    
    def test_analyze_sources_conversion_logs(self):
        """Sources 변환 로그 분석 테스트"""
        log_content = "[SOURCES] 📊 Conversion statistics: 10/12 docs converted (83.33%), failed: 2"
        result = analyze_sources_conversion_logs(log_content)
        
        assert "conversion_statistics" in result
        assert result["total_conversions"] == 10
        assert result["total_docs"] == 12
        assert result["total_failed"] == 2
    
    def test_analyze_legal_references_logs(self):
        """Legal References 로그 분석 테스트"""
        log_content = "[LEGAL_REFS] Extracted 5 legal references"
        result = analyze_legal_references_logs(log_content)
        
        assert result["total_extracted"] == 5
    
    def test_analyze_answer_length_logs(self):
        """답변 길이 로그 분석 테스트"""
        log_content = "[ANSWER LENGTH] ⚠️ Too short: 50 (target: 100-200)"
        result = analyze_answer_length_logs(log_content)
        
        assert result["too_short_count"] == 1
        assert len(result["length_warnings"]) == 1
    
    def test_analyze_context_usage_logs(self):
        """Context Usage 로그 분석 테스트"""
        log_content = "[COVERAGE] Coverage score: 0.85\n[RELEVANCE] Relevance score: 0.90"
        result = analyze_context_usage_logs(log_content)
        
        assert result["average_coverage"] == 0.85
        assert result["average_relevance"] == 0.90
    
    def test_identify_improvements(self):
        """개선 사항 식별 테스트"""
        analysis_results = {
            "sources": {
                "total_docs": 100,
                "total_conversions": 80,  # 80% 변환률 (90% 미만)
                "critical_fallbacks": [{"doc_index": 1}]
            },
            "legal_references": {
                "total_extracted": 0  # 0개 추출
            },
            "answer_length": {
                "too_short_count": 5
            },
            "context_usage": {
                "average_coverage": 0.75  # 0.80 미만
            }
        }
        
        improvements = identify_improvements(analysis_results)
        
        assert isinstance(improvements, list)
        assert len(improvements) > 0
        
        # HIGH 우선순위 개선 사항 확인
        high_priority = [imp for imp in improvements if imp.get("priority") == "HIGH"]
        assert len(high_priority) > 0

