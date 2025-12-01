"""
Sources 통일 기능 테스트
sources_by_type 및 legal_references 추출 기능 테스트
"""
import pytest
from api.services.sources_extractor import SourcesExtractor


@pytest.mark.integration
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

