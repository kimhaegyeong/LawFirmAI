# -*- coding: utf-8 -*-
"""
Processing Extractors 테스트
langgraph_core/processing/extractors.py 단위 테스트
"""

import pytest
from typing import List, Dict

from lawfirm_langgraph.langgraph_core.processing.extractors import DocumentExtractor


class TestDocumentExtractor:
    """DocumentExtractor 테스트"""
    
    def test_extract_terms_from_documents(self):
        """문서에서 법률 용어 추출 테스트"""
        docs = [
            {"content": "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."},
            {"content": "민법 제543조에 따르면 계약 해제가 가능합니다."}
        ]
        
        terms = DocumentExtractor.extract_terms_from_documents(docs)
        
        assert isinstance(terms, list)
        assert len(terms) > 0
    
    def test_extract_terms_from_documents_empty_or_no_content(self):
        """빈 문서 또는 내용 없는 문서에서 용어 추출 테스트"""
        empty_docs = []
        terms_empty = DocumentExtractor.extract_terms_from_documents(empty_docs)
        assert isinstance(terms_empty, list)
        assert len(terms_empty) == 0
        
        no_content_docs = [{"content": ""}, {"metadata": "test"}]
        terms_no_content = DocumentExtractor.extract_terms_from_documents(no_content_docs)
        assert isinstance(terms_no_content, list)
    
    def test_extract_key_insights(self):
        """핵심 정보 추출 테스트"""
        docs = [
            {"content": "계약 해지는 계약서에 명시된 조건에 따라 가능합니다. 민법 제543조에 따르면 계약 해제가 가능합니다."}
        ]
        query = "계약 해지 방법"
        
        insights = DocumentExtractor.extract_key_insights(docs, query)
        
        assert isinstance(insights, list)
        assert len(insights) <= 15
    
    def test_extract_key_insights_edge_cases(self):
        """빈 문서 또는 매칭되지 않는 문서에서 핵심 정보 추출 테스트"""
        empty_docs = []
        query = "계약 해지 방법"
        insights_empty = DocumentExtractor.extract_key_insights(empty_docs, query)
        assert isinstance(insights_empty, list)
        assert len(insights_empty) == 0
        
        no_match_docs = [{"content": "완전히 다른 주제에 대한 내용입니다."}]
        insights_no_match = DocumentExtractor.extract_key_insights(no_match_docs, query)
        assert isinstance(insights_no_match, list)
    
    def test_extract_legal_citations(self):
        """법률 인용 정보 추출 테스트"""
        docs = [
            {"content": "민법 제543조에 따르면 계약 해제가 가능합니다."},
            {"content": "형법 제250조 살인죄에 대한 내용입니다."}
        ]
        
        citations = DocumentExtractor.extract_legal_citations(docs)
        
        assert isinstance(citations, list)
    
    def test_extract_legal_citations_edge_cases(self):
        """인용이 없거나 내용이 없는 문서에서 법률 인용 정보 추출 테스트"""
        no_citation_docs = [{"content": "일반적인 내용입니다."}]
        citations_no_citation = DocumentExtractor.extract_legal_citations(no_citation_docs)
        assert isinstance(citations_no_citation, list)
        
        no_content_docs = [{"content": ""}]
        citations_no_content = DocumentExtractor.extract_legal_citations(no_content_docs)
        assert isinstance(citations_no_content, list)

