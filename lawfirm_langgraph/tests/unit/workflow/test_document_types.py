# -*- coding: utf-8 -*-
"""
DocumentType 및 DocumentTypeConfig 단위 테스트
"""

import pytest
from lawfirm_langgraph.core.workflow.constants.document_types import (
    DocumentType,
    DocumentTypeConfig,
)


class TestDocumentType:
    """DocumentType Enum 테스트"""
    
    def test_from_string_exact_match(self):
        """정확한 문자열 매칭 테스트"""
        assert DocumentType.from_string("statute_article") == DocumentType.STATUTE_ARTICLE
        assert DocumentType.from_string("case_paragraph") == DocumentType.CASE_PARAGRAPH
        assert DocumentType.from_string("precedent_content") == DocumentType.PRECEDENT_CONTENT
        assert DocumentType.from_string("decision_paragraph") == DocumentType.DECISION_PARAGRAPH
        assert DocumentType.from_string("interpretation_paragraph") == DocumentType.INTERPRETATION_PARAGRAPH
        assert DocumentType.from_string("unknown") == DocumentType.UNKNOWN
    
    def test_from_string_case_insensitive(self):
        """대소문자 무시 테스트"""
        assert DocumentType.from_string("STATUTE_ARTICLE") == DocumentType.STATUTE_ARTICLE
        assert DocumentType.from_string("Case_Paragraph") == DocumentType.CASE_PARAGRAPH
    
    def test_from_string_legacy_compatibility(self):
        """레거시 호환성 테스트"""
        assert DocumentType.from_string("case") == DocumentType.CASE_PARAGRAPH
        assert DocumentType.from_string("precedent") == DocumentType.CASE_PARAGRAPH
    
    def test_from_string_empty_or_none(self):
        """빈 문자열 또는 None 테스트"""
        assert DocumentType.from_string("") == DocumentType.UNKNOWN
        assert DocumentType.from_string(None) == DocumentType.UNKNOWN
    
    def test_all_types(self):
        """모든 타입 리스트 반환 테스트"""
        all_types = DocumentType.all_types()
        assert "statute_article" in all_types
        assert "case_paragraph" in all_types
        assert "precedent_content" in all_types
        assert "decision_paragraph" in all_types
        assert "interpretation_paragraph" in all_types
        assert "unknown" not in all_types
    
    def test_from_metadata_with_type_field(self):
        """type 필드가 있는 경우 테스트"""
        doc = {"type": "statute_article"}
        assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
        
        doc = {"source_type": "case_paragraph"}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
    
    def test_from_metadata_with_metadata_source_type(self):
        """metadata.source_type이 있는 경우 테스트"""
        doc = {"metadata": {"source_type": "decision_paragraph"}}
        assert DocumentType.from_metadata(doc) == DocumentType.DECISION_PARAGRAPH
    
    def test_from_metadata_statute_article(self):
        """statute_article 메타데이터 필드 기반 추론 테스트"""
        # statute_name 필드
        doc = {"metadata": {"statute_name": "민법"}}
        assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
        
        # law_name 필드
        doc = {"metadata": {"law_name": "형법"}}
        assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
        
        # article_no 필드
        doc = {"metadata": {"article_no": "750"}}
        assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
    
    def test_from_metadata_case_paragraph(self):
        """case_paragraph 메타데이터 필드 기반 추론 테스트"""
        # case_id 필드
        doc = {"metadata": {"case_id": "12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
        
        # court 필드
        doc = {"metadata": {"court": "대법원"}}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
        
        # casenames 필드
        doc = {"metadata": {"casenames": "손해배상청구 사건"}}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
        
        # doc_id 필드
        doc = {"metadata": {"doc_id": "2021다12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
        
        # precedent_id 필드
        doc = {"metadata": {"precedent_id": "12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.CASE_PARAGRAPH
    
    def test_from_metadata_interpretation_paragraph(self):
        """interpretation_paragraph 메타데이터 필드 기반 추론 테스트"""
        # interpretation_id 필드
        doc = {"metadata": {"interpretation_id": "12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.INTERPRETATION_PARAGRAPH
        
        # interpretation_number 필드
        doc = {"metadata": {"interpretation_number": "2021-001"}}
        assert DocumentType.from_metadata(doc) == DocumentType.INTERPRETATION_PARAGRAPH
    
    def test_from_metadata_decision_paragraph(self):
        """decision_paragraph 메타데이터 필드 기반 추론 테스트"""
        # decision_id 필드
        doc = {"metadata": {"decision_id": "12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.DECISION_PARAGRAPH
        
        # org 필드 (interpretation_id 없음)
        doc = {"metadata": {"org": "행정심판위원회"}}
        assert DocumentType.from_metadata(doc) == DocumentType.DECISION_PARAGRAPH
        
        # org 필드가 있지만 interpretation_id도 있으면 interpretation_paragraph
        doc = {"metadata": {"org": "행정심판위원회", "interpretation_id": "12345"}}
        assert DocumentType.from_metadata(doc) == DocumentType.INTERPRETATION_PARAGRAPH
    
    def test_from_metadata_priority_order(self):
        """우선순위 순서 테스트 (interpretation_id가 있으면 interpretation_paragraph 우선)"""
        # interpretation_id와 case_id가 모두 있으면 interpretation_paragraph 우선
        doc = {"metadata": {"interpretation_id": "12345", "case_id": "67890"}}
        assert DocumentType.from_metadata(doc) == DocumentType.INTERPRETATION_PARAGRAPH
    
    def test_from_metadata_unknown(self):
        """알 수 없는 타입 테스트"""
        doc = {}
        assert DocumentType.from_metadata(doc) == DocumentType.UNKNOWN
        
        doc = {"metadata": {}}
        assert DocumentType.from_metadata(doc) == DocumentType.UNKNOWN
        
        doc = {"metadata": {"some_other_field": "value"}}
        assert DocumentType.from_metadata(doc) == DocumentType.UNKNOWN
    
    def test_from_metadata_invalid_input(self):
        """잘못된 입력 테스트"""
        assert DocumentType.from_metadata(None) == DocumentType.UNKNOWN
        assert DocumentType.from_metadata("not a dict") == DocumentType.UNKNOWN
        assert DocumentType.from_metadata([]) == DocumentType.UNKNOWN


class TestDocumentTypeConfig:
    """DocumentTypeConfig 클래스 테스트"""
    
    def test_get_min_count(self):
        """최소 개수 반환 테스트"""
        assert DocumentTypeConfig.get_min_count(DocumentType.STATUTE_ARTICLE) == 1
        assert DocumentTypeConfig.get_min_count(DocumentType.CASE_PARAGRAPH) == 2
        assert DocumentTypeConfig.get_min_count(DocumentType.PRECEDENT_CONTENT) == 2
        assert DocumentTypeConfig.get_min_count(DocumentType.DECISION_PARAGRAPH) == 1
        assert DocumentTypeConfig.get_min_count(DocumentType.INTERPRETATION_PARAGRAPH) == 1
        assert DocumentTypeConfig.get_min_count(DocumentType.UNKNOWN) == 0
    
    def test_get_priority_boost(self):
        """우선순위 부스팅 값 반환 테스트"""
        assert DocumentTypeConfig.get_priority_boost(DocumentType.STATUTE_ARTICLE) == 1.0
        assert DocumentTypeConfig.get_priority_boost(DocumentType.CASE_PARAGRAPH) == 0.9
        assert DocumentTypeConfig.get_priority_boost(DocumentType.PRECEDENT_CONTENT) == 0.9
        assert DocumentTypeConfig.get_priority_boost(DocumentType.DECISION_PARAGRAPH) == 0.8
        assert DocumentTypeConfig.get_priority_boost(DocumentType.INTERPRETATION_PARAGRAPH) == 0.7
        assert DocumentTypeConfig.get_priority_boost(DocumentType.UNKNOWN) == 0.5
    
    def test_get_required_metadata_keys(self):
        """필수 메타데이터 키 반환 테스트"""
        keys = DocumentTypeConfig.get_required_metadata_keys(DocumentType.STATUTE_ARTICLE)
        assert "statute_name" in keys
        assert "law_name" in keys
        assert "article_no" in keys
        
        keys = DocumentTypeConfig.get_required_metadata_keys(DocumentType.CASE_PARAGRAPH)
        assert "doc_id" in keys
        
        keys = DocumentTypeConfig.get_required_metadata_keys(DocumentType.INTERPRETATION_PARAGRAPH)
        assert "interpretation_id" in keys
    
    def test_get_optional_metadata_keys(self):
        """선택적 메타데이터 키 반환 테스트"""
        keys = DocumentTypeConfig.get_optional_metadata_keys(DocumentType.STATUTE_ARTICLE)
        assert "clause_no" in keys
        assert "item_no" in keys
        assert "statute_id" in keys
        
        keys = DocumentTypeConfig.get_optional_metadata_keys(DocumentType.CASE_PARAGRAPH)
        assert "case_id" in keys
        assert "court" in keys
        assert "casenames" in keys
    
    def test_is_valid_metadata(self):
        """메타데이터 유효성 검증 테스트"""
        # statute_article: 필수 필드 중 하나라도 있으면 유효
        metadata = {"statute_name": "민법"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == True
        
        metadata = {"law_name": "형법"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == True
        
        metadata = {"article_no": "750"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == True
        
        # 필수 필드가 없으면 유효하지 않음
        metadata = {"some_other_field": "value"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == False
        
        # case_paragraph: doc_id가 있으면 유효
        metadata = {"doc_id": "2021다12345"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.CASE_PARAGRAPH, metadata) == True
        
        # interpretation_paragraph: interpretation_id가 있으면 유효
        metadata = {"interpretation_id": "12345"}
        assert DocumentTypeConfig.is_valid_metadata(DocumentType.INTERPRETATION_PARAGRAPH, metadata) == True


class TestDocumentTypeIntegration:
    """DocumentType 통합 테스트"""
    
    def test_real_world_statute_document(self):
        """실제 법령 문서 예시 테스트"""
        doc = {
            "type": "statute_article",
            "metadata": {
                "statute_name": "민법",
                "article_no": "750",
                "clause_no": "1",
                "item_no": "1"
            },
            "content": "불법행위로 인한 손해배상의무..."
        }
        doc_type = DocumentType.from_metadata(doc)
        assert doc_type == DocumentType.STATUTE_ARTICLE
        assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True
    
    def test_real_world_case_document(self):
        """실제 판례 문서 예시 테스트"""
        doc = {
            "source_type": "case_paragraph",
            "metadata": {
                "doc_id": "2021다12345",
                "court": "대법원",
                "casenames": "손해배상청구 사건",
                "case_id": "12345"
            },
            "content": "원고는 피고의 불법행위로 인한 손해를 입었다고 주장한다..."
        }
        doc_type = DocumentType.from_metadata(doc)
        assert doc_type == DocumentType.CASE_PARAGRAPH
        assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True
    
    def test_real_world_decision_document(self):
        """실제 결정례 문서 예시 테스트"""
        doc = {
            "metadata": {
                "doc_id": "2021결정12345",
                "org": "행정심판위원회",
                "decision_id": "12345"
            },
            "content": "본 사건은 행정처분에 대한 심판청구 사건이다..."
        }
        doc_type = DocumentType.from_metadata(doc)
        assert doc_type == DocumentType.DECISION_PARAGRAPH
        assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True
    
    def test_real_world_interpretation_document(self):
        """실제 해석례 문서 예시 테스트"""
        doc = {
            "metadata": {
                "interpretation_id": "2021-001",
                "org": "법제처",
                "title": "법령 해석 질의"
            },
            "content": "법령 해석에 관한 질의에 대한 답변입니다..."
        }
        doc_type = DocumentType.from_metadata(doc)
        assert doc_type == DocumentType.INTERPRETATION_PARAGRAPH
        assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True

