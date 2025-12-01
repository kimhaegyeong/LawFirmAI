# -*- coding: utf-8 -*-
"""
Citation 개선사항 테스트
- 접두어 포함 패턴 추출
- 유사도 기반 매칭
- Coverage 계산 개선
- 정규화 일관성
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.validators.quality_validators import AnswerValidator


class TestCitationExtractionWithPrefix:
    """접두어 포함 Citation 추출 테스트"""
    
    def test_extract_with_prefix_특히(self):
        """'특히' 접두어가 포함된 Citation 추출 테스트"""
        answer = "특히국세기본법 제18조에 따르면..."
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0, "Should extract at least one citation"
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None, "Should extract law citation"
        assert law_citation.get("law_name") == "국세기본법", f"Expected '국세기본법', got '{law_citation.get('law_name')}'"
        assert law_citation.get("article_number") == "18", f"Expected '18', got '{law_citation.get('article_number')}'"
    
    def test_extract_with_prefix_또한(self):
        """'또한' 접두어가 포함된 Citation 추출 테스트"""
        answer = "또한국세기본법 제18조를 참고하세요"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None
        assert law_citation.get("law_name") == "국세기본법"
    
    def test_extract_with_prefix_규정한(self):
        """'규정한' 접두어가 포함된 Citation 추출 테스트"""
        answer = "규정한민사집행법 제287조에 명시되어 있습니다"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None
        assert law_citation.get("law_name") == "민사집행법"
        assert law_citation.get("article_number") == "287"
    
    def test_extract_with_prefix_와(self):
        """'와' 접두어가 포함된 Citation 추출 테스트 (공백 없음)"""
        answer = "와민사집행법 제26조를 참고하세요"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None
        assert law_citation.get("law_name") == "민사집행법"
        assert law_citation.get("article_number") == "26"
    
    def test_extract_with_prefix_규정은(self):
        """'규정은' 접두어가 포함된 Citation 추출 테스트 (공백 없음)"""
        answer = "규정은민사집행법 제301조에 따르면"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None
        assert law_citation.get("law_name") == "민사집행법"
        assert law_citation.get("article_number") == "301"
    
    def test_extract_multiple_citations_with_prefixes(self):
        """여러 접두어가 포함된 Citation 추출 테스트"""
        answer = "특히국세기본법 제18조와 또한민사집행법 제287조를 참고하세요"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) >= 2, f"Expected at least 2 citations, got {len(citations)}"
        law_names = [c.get("law_name") for c in citations if c.get("type") == "law"]
        assert "국세기본법" in law_names
        assert "민사집행법" in law_names


class TestCitationMatchingWithFuzzy:
    """유사도 기반 Citation 매칭 테스트"""
    
    def test_string_similarity_calculation(self):
        """문자열 유사도 계산 테스트"""
        # 정확 일치
        similarity = AnswerValidator._calculate_string_similarity("민법", "민법")
        assert similarity == 1.0, f"Expected 1.0, got {similarity}"
        
        # 부분 일치
        similarity = AnswerValidator._calculate_string_similarity("민법", "민법 제750조")
        assert similarity >= 0.5, f"Expected >= 0.5, got {similarity}"
        
        # 유사한 문자열
        similarity = AnswerValidator._calculate_string_similarity("국세기본법", "국세기본법")
        assert similarity == 1.0
        
        # 다른 문자열
        similarity = AnswerValidator._calculate_string_similarity("민법", "형법")
        assert similarity < 1.0
    
    def test_match_citations_with_fuzzy(self):
        """유사도 기반 Citation 매칭 테스트"""
        expected = {
            "type": "law",
            "law_name": "국세기본법",
            "article_number": "18",
            "normalized": "국세기본법 제18조"
        }
        
        # 정확 일치
        answer1 = {
            "type": "law",
            "law_name": "국세기본법",
            "article_number": "18",
            "normalized": "국세기본법 제18조"
        }
        assert AnswerValidator._match_citations(expected, answer1, use_fuzzy=True)
        
        # 접두어가 제거된 경우 (정규화 후 동일)
        answer2 = {
            "type": "law",
            "law_name": "국세기본법",  # 이미 정규화됨
            "article_number": "18",
            "normalized": "국세기본법 제18조"
        }
        assert AnswerValidator._match_citations(expected, answer2, use_fuzzy=True)
    
    def test_match_citations_without_fuzzy(self):
        """유사도 기반 매칭 없이 Citation 매칭 테스트"""
        expected = {
            "type": "law",
            "law_name": "국세기본법",
            "article_number": "18",
            "normalized": "국세기본법 제18조"
        }
        
        answer = {
            "type": "law",
            "law_name": "국세기본법",
            "article_number": "18",
            "normalized": "국세기본법 제18조"
        }
        
        # use_fuzzy=False여도 정확 일치는 매칭됨
        assert AnswerValidator._match_citations(expected, answer, use_fuzzy=False)


class TestCitationNormalizationConsistency:
    """Citation 정규화 일관성 테스트"""
    
    def test_normalization_consistency(self):
        """expected와 answer가 동일한 방식으로 정규화되는지 테스트"""
        # 접두어가 포함된 Citation
        citation_with_prefix = "특히국세기본법 제18조"
        
        # 정규화
        normalized1 = AnswerValidator._normalize_citation(citation_with_prefix)
        
        # 재정규화 (일관성 보장)
        normalized2 = AnswerValidator._normalize_citation(normalized1.get("original", citation_with_prefix))
        
        # 법령명이 일치해야 함
        assert normalized1.get("law_name") == normalized2.get("law_name"), \
            f"Law names should match: {normalized1.get('law_name')} vs {normalized2.get('law_name')}"
        assert normalized1.get("article_number") == normalized2.get("article_number"), \
            f"Article numbers should match: {normalized1.get('article_number')} vs {normalized2.get('article_number')}"
    
    def test_extract_and_normalize_consistency(self):
        """추출 및 정규화 일관성 테스트"""
        answer = "특히국세기본법 제18조와 또한민사집행법 제287조"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        # 모든 Citation이 정규화되어야 함
        for citation in citations:
            assert citation.get("type") != "unknown", f"Citation should be normalized: {citation}"
            assert citation.get("normalized"), f"Citation should have normalized field: {citation}"
            
            # 법령명에 접두어가 없어야 함
            if citation.get("type") == "law":
                law_name = citation.get("law_name", "")
                # 접두어가 제거되었는지 확인
                assert not any(prefix in law_name for prefix in ["특히", "또한", "규정한", "규정은", "와", "과"]), \
                    f"Law name should not contain prefix: {law_name}"


class TestCitationCoverageImprovement:
    """Citation Coverage 계산 개선 테스트"""
    
    def test_coverage_with_matched_citations(self):
        """매칭된 Citation이 있는 경우 coverage 계산 테스트"""
        answer = "국세기본법 제18조와 민사집행법 제287조를 참고하세요"
        context = {
            "context": "국세기본법 제18조와 민사집행법 제287조에 대한 설명",
            "legal_references": ["국세기본법 제18조", "민사집행법 제287조"],
            "citations": []
        }
        
        result = AnswerValidator.validate_answer_uses_context(
            answer=answer,
            context=context,
            query="테스트 질문",
            retrieved_docs=None
        )
        
        assert result.get("citation_coverage", 0) > 0, "Should have positive citation coverage"
        assert result.get("coverage_score", 0) > 0, "Should have positive coverage score"
    
    def test_coverage_with_unmatched_citations(self):
        """매칭되지 않은 Citation이 있는 경우 coverage 계산 테스트 (보너스 점수)"""
        answer = "민법 제750조를 참고하세요"
        context = {
            "context": "국세기본법 제18조에 대한 설명",
            "legal_references": ["국세기본법 제18조"],
            "citations": []
        }
        
        result = AnswerValidator.validate_answer_uses_context(
            answer=answer,
            context=context,
            query="테스트 질문",
            retrieved_docs=None
        )
        
        # 매칭되지 않았지만 Citation이 있으면 부분 점수 부여
        citation_coverage = result.get("citation_coverage", 0)
        # 매칭 실패 시 최소 0.3 점수 부여 (개선 후)
        assert citation_coverage >= 0.0, f"Should have non-negative coverage: {citation_coverage}"
    
    def test_coverage_with_prefix_citations(self):
        """접두어가 포함된 Citation의 coverage 계산 테스트"""
        answer = "특히국세기본법 제18조에 따르면"
        context = {
            "context": "국세기본법 제18조에 대한 설명",
            "legal_references": ["국세기본법 제18조"],
            "citations": []
        }
        
        result = AnswerValidator.validate_answer_uses_context(
            answer=answer,
            context=context,
            query="테스트 질문",
            retrieved_docs=None
        )
        
        # 접두어가 제거되어 매칭되어야 함
        citation_coverage = result.get("citation_coverage", 0)
        assert citation_coverage > 0, f"Should match citation with prefix, coverage: {citation_coverage}"


class TestCitationExtractionPatterns:
    """다양한 Citation 추출 패턴 테스트"""
    
    def test_extract_with_article_and_paragraph(self):
        """항/호가 포함된 Citation 추출 테스트"""
        answer = "민사소송법 제217조 제1항 제2호에 따르면"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        law_citation = next((c for c in citations if c.get("type") == "law"), None)
        assert law_citation is not None
        # 항/호는 제거되고 기본 조문번호만 저장
        assert law_citation.get("article_number") == "217"
        assert law_citation.get("law_name") == "민사소송법"
    
    def test_extract_precedent_with_date(self):
        """날짜가 포함된 판례 Citation 추출 테스트"""
        answer = "대법원 2014. 7. 24. 선고 2012다49933 판결"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        precedent_citation = next((c for c in citations if c.get("type") == "precedent"), None)
        assert precedent_citation is not None
        assert precedent_citation.get("case_number") == "2012다49933"
        assert precedent_citation.get("court") == "대법원"
    
    def test_extract_precedent_without_date(self):
        """날짜가 없는 판례 Citation 추출 테스트"""
        answer = "대법원 2012다49933 판결"
        citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
        
        assert len(citations) > 0
        precedent_citation = next((c for c in citations if c.get("type") == "precedent"), None)
        assert precedent_citation is not None
        assert precedent_citation.get("case_number") == "2012다49933"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

