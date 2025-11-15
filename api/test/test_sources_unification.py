"""
Sources 통일 기능 테스트
sources_by_type 및 legal_references 추출 기능 테스트
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.services.sources_extractor import SourcesExtractor


class TestSourcesByType:
    """sources_by_type 기능 테스트"""
    
    def test_get_sources_by_type_basic(self):
        """기본 타입별 그룹화 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {"type": "statute_article", "statute_name": "민법", "article_no": "123"},
            {"type": "case_paragraph", "case_number": "2021다123"},
            {"type": "decision_paragraph", "decision_number": "19-진정-0404100"},
            {"type": "interpretation_paragraph", "interpretation_number": "doc_id"},
        ]
        
        result = extractor._get_sources_by_type(sources_detail)
        
        assert len(result["statute_article"]) == 1
        assert len(result["case_paragraph"]) == 1
        assert len(result["decision_paragraph"]) == 1
        assert len(result["interpretation_paragraph"]) == 1
        assert result["statute_article"][0]["statute_name"] == "민법"
        assert result["case_paragraph"][0]["case_number"] == "2021다123"
    
    def test_get_sources_by_type_multiple(self):
        """여러 개의 동일 타입 문서 그룹화 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {"type": "statute_article", "statute_name": "민법", "article_no": "123"},
            {"type": "statute_article", "statute_name": "형법", "article_no": "234"},
            {"type": "case_paragraph", "case_number": "2021다123"},
            {"type": "case_paragraph", "case_number": "2022다456"},
        ]
        
        result = extractor._get_sources_by_type(sources_detail)
        
        assert len(result["statute_article"]) == 2
        assert len(result["case_paragraph"]) == 2
        assert len(result["decision_paragraph"]) == 0
        assert len(result["interpretation_paragraph"]) == 0
    
    def test_get_sources_by_type_empty(self):
        """빈 sources_detail 테스트"""
        extractor = SourcesExtractor(None, None)
        
        result = extractor._get_sources_by_type([])
        
        assert len(result["statute_article"]) == 0
        assert len(result["case_paragraph"]) == 0
        assert len(result["decision_paragraph"]) == 0
        assert len(result["interpretation_paragraph"]) == 0
    
    def test_get_sources_by_type_invalid_type(self):
        """알 수 없는 타입 필터링 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {"type": "statute_article", "statute_name": "민법"},
            {"type": "unknown_type", "name": "알 수 없는 타입"},
            {"type": "", "name": "타입 없음"},
        ]
        
        result = extractor._get_sources_by_type(sources_detail)
        
        assert len(result["statute_article"]) == 1
        assert len(result["case_paragraph"]) == 0
        assert len(result["decision_paragraph"]) == 0
        assert len(result["interpretation_paragraph"]) == 0
    
    def test_get_sources_by_type_invalid_dict(self):
        """유효하지 않은 딕셔너리 필터링 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {"type": "statute_article", "statute_name": "민법"},
            None,
            "invalid",
            123,
            [],
        ]
        
        result = extractor._get_sources_by_type(sources_detail)
        
        assert len(result["statute_article"]) == 1


class TestFormatLegalReferenceFromDetail:
    """legal_reference 포맷팅 테스트"""
    
    def test_format_legal_reference_complete(self):
        """완전한 법령 정보 포맷팅 테스트"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "statute_article",
            "statute_name": "민법",
            "article_no": "123",
            "clause_no": "1",
            "item_no": "2"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result == "민법 123 제1항 제2호"
    
    def test_format_legal_reference_statute_and_article(self):
        """법령명과 조문만 있는 경우"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "statute_article",
            "statute_name": "형법",
            "article_no": "234"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result == "형법 234"
    
    def test_format_legal_reference_article_only(self):
        """조문만 있는 경우"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "statute_article",
            "article_no": "123"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result == "123"
    
    def test_format_legal_reference_statute_only(self):
        """법령명만 있는 경우"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "statute_article",
            "statute_name": "민법"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result == "민법"
    
    def test_format_legal_reference_empty(self):
        """법령명과 조문이 모두 없는 경우"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "statute_article"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result is None
    
    def test_format_legal_reference_non_statute(self):
        """statute_article이 아닌 경우"""
        extractor = SourcesExtractor(None, None)
        
        detail = {
            "type": "case_paragraph",
            "case_number": "2021다123"
        }
        
        result = extractor._format_legal_reference_from_detail(detail)
        assert result is None


class TestExtractLegalReferencesFromSourcesDetailOnly:
    """sources_detail에서 legal_references 추출 테스트"""
    
    def test_extract_legal_references_basic(self):
        """기본 legal_references 추출 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123"
            },
            {
                "type": "statute_article",
                "statute_name": "형법",
                "article_no": "234"
            },
            {
                "type": "case_paragraph",
                "case_number": "2021다123"
            }
        ]
        
        result = extractor._extract_legal_references_from_sources_detail_only(sources_detail)
        
        assert len(result) == 2
        assert "민법 123" in result
        assert "형법 234" in result
    
    def test_extract_legal_references_with_clause_and_item(self):
        """항과 호가 있는 경우"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123",
                "clause_no": "1",
                "item_no": "2"
            }
        ]
        
        result = extractor._extract_legal_references_from_sources_detail_only(sources_detail)
        
        assert len(result) == 1
        assert result[0] == "민법 123 제1항 제2호"
    
    def test_extract_legal_references_duplicate_removal(self):
        """중복 제거 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123"
            },
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123"
            }
        ]
        
        result = extractor._extract_legal_references_from_sources_detail_only(sources_detail)
        
        assert len(result) == 1
        assert result[0] == "민법 123"
    
    def test_extract_legal_references_empty(self):
        """빈 sources_detail 테스트"""
        extractor = SourcesExtractor(None, None)
        
        result = extractor._extract_legal_references_from_sources_detail_only([])
        
        assert len(result) == 0
    
    def test_extract_legal_references_no_statute(self):
        """법령이 없는 경우"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "case_paragraph",
                "case_number": "2021다123"
            },
            {
                "type": "decision_paragraph",
                "decision_number": "19-진정-0404100"
            }
        ]
        
        result = extractor._extract_legal_references_from_sources_detail_only(sources_detail)
        
        assert len(result) == 0
    
    def test_extract_legal_references_invalid_dict(self):
        """유효하지 않은 딕셔너리 필터링 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123"
            },
            None,
            "invalid",
            {}
        ]
        
        result = extractor._extract_legal_references_from_sources_detail_only(sources_detail)
        
        assert len(result) == 1
        assert result[0] == "민법 123"


class TestSourcesUnificationIntegration:
    """Sources 통일 통합 테스트"""
    
    def test_extract_from_message_with_sources_by_type(self):
        """메시지에서 sources_by_type 추출 테스트"""
        extractor = SourcesExtractor(None, None)
        
        msg = {
            "metadata": {
                "sources_detail": [
                    {"type": "statute_article", "statute_name": "민법", "article_no": "123"},
                    {"type": "case_paragraph", "case_number": "2021다123"},
                ]
            }
        }
        
        result = extractor._extract_from_message(msg)
        
        assert "sources_by_type" in result
        assert len(result["sources_by_type"]["statute_article"]) == 1
        assert len(result["sources_by_type"]["case_paragraph"]) == 1
        assert "legal_references" in result  # 하위 호환성
        assert len(result["legal_references"]) > 0
        assert "민법 123" in result["legal_references"]
    
    def test_extract_from_message_legal_references_auto_extract(self):
        """legal_references 자동 추출 테스트"""
        extractor = SourcesExtractor(None, None)
        
        msg = {
            "metadata": {
                "sources_detail": [
                    {"type": "statute_article", "statute_name": "민법", "article_no": "123"},
                    {"type": "statute_article", "statute_name": "형법", "article_no": "234"},
                ],
                "legal_references": ["기존 법령"]  # 기존 legal_references와 병합
            }
        }
        
        result = extractor._extract_from_message(msg)
        
        assert "legal_references" in result
        assert len(result["legal_references"]) >= 2
        assert "민법 123" in result["legal_references"]
        assert "형법 234" in result["legal_references"]
        assert "기존 법령" in result["legal_references"]  # 기존 값 유지
    
    def test_sources_by_type_structure(self):
        """sources_by_type 구조 검증"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {"type": "statute_article", "statute_name": "민법"},
            {"type": "case_paragraph", "case_number": "2021다123"},
            {"type": "decision_paragraph", "decision_number": "19-진정-0404100"},
            {"type": "interpretation_paragraph", "interpretation_number": "doc_id"},
        ]
        
        result = extractor._get_sources_by_type(sources_detail)
        
        # 구조 검증
        assert isinstance(result, dict)
        assert "statute_article" in result
        assert "case_paragraph" in result
        assert "decision_paragraph" in result
        assert "interpretation_paragraph" in result
        
        # 타입 검증
        assert isinstance(result["statute_article"], list)
        assert isinstance(result["case_paragraph"], list)
        assert isinstance(result["decision_paragraph"], list)
        assert isinstance(result["interpretation_paragraph"], list)


class TestCaseNumberExtraction:
    """case_number 추출 및 설정 테스트"""
    
    def test_normalize_sources_detail_case_number_from_metadata(self):
        """metadata에서 case_number 추출 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "url": "",
                "content": "판례 내용",
                "metadata": {
                    "doc_id": "case_2021다123",
                    "court": "대법원",
                    "casenames": "테스트 사건"
                }
            }
        ]
        
        normalized = extractor._normalize_sources_detail(sources_detail)
        
        assert len(normalized) == 1
        assert normalized[0]["type"] == "case_paragraph"
        assert normalized[0]["case_number"] == "case_2021다123"
        assert normalized[0]["court"] == "대법원"
        assert normalized[0]["case_name"] == "테스트 사건"
    
    def test_normalize_sources_detail_case_number_from_case_id(self):
        """case_id에서 case_number 추출 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "url": "",
                "content": "판례 내용",
                "metadata": {
                    "case_id": "2021다123",
                    "court": "대법원"
                }
            }
        ]
        
        normalized = extractor._normalize_sources_detail(sources_detail)
        
        assert len(normalized) == 1
        assert normalized[0]["type"] == "case_paragraph"
        assert normalized[0]["case_number"] == "2021다123"
    
    def test_normalize_sources_detail_case_number_empty_when_missing(self):
        """doc_id와 case_id가 모두 없을 때 빈 문자열로 설정되는지 테스트"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "url": "",
                "content": "판례 내용",
                "metadata": {
                    "court": "대법원"
                }
            }
        ]
        
        normalized = extractor._normalize_sources_detail(sources_detail)
        
        assert len(normalized) == 1
        assert normalized[0]["type"] == "case_paragraph"
        assert normalized[0]["case_number"] == ""  # 빈 문자열로 설정되어야 함
        assert "case_number" in normalized[0]  # 필드는 존재해야 함
    
    def test_normalize_sources_detail_case_number_priority(self):
        """case_number 우선순위 테스트 (case_number > doc_id > case_id)"""
        extractor = SourcesExtractor(None, None)
        
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "url": "",
                "content": "판례 내용",
                "case_number": "우선순위값",
                "metadata": {
                    "doc_id": "case_2021다123",
                    "case_id": "2021다123"
                }
            }
        ]
        
        normalized = extractor._normalize_sources_detail(sources_detail)
        
        assert len(normalized) == 1
        assert normalized[0]["case_number"] == "우선순위값"  # 최상위 레벨의 case_number가 우선
