"""
Sources 데이터 개선 로직 테스트
"""
import pytest
from api.services.sources_extractor import SourcesExtractor


@pytest.mark.integration
class TestSourcesEnhancement:
    """Sources 데이터 개선 로직 테스트"""
    
    def test_parse_source_string(self):
        """sources 배열 파싱 테스트"""
        extractor = SourcesExtractor(None, None)
        
        test_cases = [
            ("위자료 (case_69므37)", {"doc_id": "case_69므37", "casenames": "위자료"}),
            ("손해배상(기) (case_2011므2997)", {"doc_id": "case_2011므2997", "casenames": "손해배상(기)"}),
            ("손해배상(이혼) (case_2013므2441)", {"doc_id": "case_2013므2441", "casenames": "손해배상(이혼)"}),
        ]
        
        for source_str, expected in test_cases:
            result = extractor._parse_source_string(source_str)
            
            assert result.get("doc_id") == expected.get("doc_id"), f"doc_id 불일치: {result.get('doc_id')} != {expected.get('doc_id')}"
            assert result.get("casenames") == expected.get("casenames"), f"casenames 불일치: {result.get('casenames')} != {expected.get('casenames')}"
    
    def test_generate_case_url(self):
        """판례 URL 생성 테스트"""
        extractor = SourcesExtractor(None, None)
        
        test_cases = [
            ("case_2000르125", "http://www.law.go.kr/DRF/lawService.do?target=prec&ID=case_2000르125&type=HTML"),
            ("case_69므37", "http://www.law.go.kr/DRF/lawService.do?target=prec&ID=case_69므37&type=HTML"),
            ("", ""),
        ]
        
        for case_id, expected_url in test_cases:
            result = extractor._generate_case_url(case_id)
            assert result == expected_url

