"""
Source 이벤트 최적화 테스트
"""
import pytest
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.services.sources_extractor import SourcesExtractor


class MockWorkflowService:
    """Mock workflow service"""
    pass


class MockSessionService:
    """Mock session service"""
    pass


@pytest.fixture
def sources_extractor():
    """SourcesExtractor 인스턴스 생성"""
    workflow_service = MockWorkflowService()
    session_service = MockSessionService()
    return SourcesExtractor(workflow_service, session_service)


class TestNormalizeContent:
    """content 필드 정규화 테스트"""
    
    def test_normalize_string_content(self, sources_extractor):
        """문자열 content 정규화"""
        content = "일반 텍스트 내용"
        result = sources_extractor._normalize_content(content)
        assert result == "일반 텍스트 내용"
    
    def test_normalize_json_string_content(self, sources_extractor):
        """JSON 문자열 content 정규화"""
        content = '{"text": "실제 텍스트", "score": 0.5}'
        result = sources_extractor._normalize_content(content)
        assert result == "실제 텍스트"
    
    def test_normalize_dict_content(self, sources_extractor):
        """딕셔너리 content 정규화"""
        content = {"text": "딕셔너리 텍스트", "score": 0.5}
        result = sources_extractor._normalize_content(content)
        assert result == "딕셔너리 텍스트"
    
    def test_normalize_empty_content(self, sources_extractor):
        """빈 content 처리"""
        assert sources_extractor._normalize_content("") is None
        assert sources_extractor._normalize_content(None) is None
        assert sources_extractor._normalize_content("   ") is None


class TestRemoveEmptyFields:
    """빈 필드 제거 테스트"""
    
    def test_remove_empty_string_fields(self, sources_extractor):
        """빈 문자열 필드 제거"""
        data = {
            "name": "테스트",
            "url": "",
            "content": "내용",
            "case_number": "   "
        }
        result = sources_extractor._remove_empty_fields(data)
        assert "name" in result
        assert "content" in result
        assert "url" not in result
        assert "case_number" not in result
    
    def test_remove_none_fields(self, sources_extractor):
        """None 필드 제거"""
        data = {
            "name": "테스트",
            "url": None,
            "content": "내용"
        }
        result = sources_extractor._remove_empty_fields(data)
        assert "name" in result
        assert "content" in result
        assert "url" not in result


class TestCleanSourceForClient:
    """클라이언트용 source 정리 테스트"""
    
    def test_clean_case_paragraph(self, sources_extractor):
        """판례 source 정리"""
        source_item = {
            "type": "case_paragraph",
            "name": "판례",
            "content": '{"text": "판례 내용", "score": 0.5}',
            "url": "",
            "case_number": "",
            "score": 0.629,
            "relevance_score": 0.63,
            "similarity": 0.629,
            "cross_encoder_score": 0.642,
            "chunk_id": 103351,
            "embedding_version_id": 6,
            "query": "계약서 작성 시 주의사항은 무엇인가요?",
            "metadata": {
                "chunk_id": 103351,
                "doc_id": "서울중앙지방법원-2017가단5137630",
                "court": "서울중앙지방법원",
                "casenames": "손해배상(기)",
                "announce_date": "2018-07-26T09:00:00.000+09:00",
                "query": "계약서 작성 시 주의사항은 무엇인가요?"
            }
        }
        
        result = sources_extractor._clean_source_for_client(source_item)
        
        # 필수 필드 확인
        assert result["type"] == "case_paragraph"
        assert result["name"] == "판례"
        assert result["content"] == "판례 내용"
        assert result["relevance_score"] == 0.63
        
        # 제거된 필드 확인
        assert "score" not in result
        assert "similarity" not in result
        assert "cross_encoder_score" not in result
        assert "chunk_id" not in result
        assert "embedding_version_id" not in result
        assert "query" not in result
        
        # 빈 필드 제거 확인
        assert "url" not in result or result.get("url") is None
        
        # metadata에서 추출된 필드 확인
        assert "case_number" in result
        assert result["case_number"] == "서울중앙지방법원-2017가단5137630"
        assert "court" in result
        assert "case_name" in result
    
    def test_clean_statute_article(self, sources_extractor):
        """법령 source 정리"""
        source_item = {
            "type": "statute_article",
            "name": "민법",
            "content": "법령 조문 내용",
            "statute_name": "민법",
            "article_no": "제535조",
            "score": 0.8,
            "relevance_score": 0.75,
            "chunk_id": 12345
        }
        
        result = sources_extractor._clean_source_for_client(source_item)
        
        assert result["type"] == "statute_article"
        assert result["statute_name"] == "민법"
        assert result["article_no"] == "제535조"
        assert result["relevance_score"] == 0.75
        assert "score" not in result
        assert "chunk_id" not in result


class TestNormalizeSourcesDetail:
    """sources_detail 정규화 테스트"""
    
    def test_normalize_with_json_content(self, sources_extractor):
        """JSON 문자열 content가 포함된 sources_detail 정규화"""
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "content": '{"text": "판례 내용", "score": 0.5}',
                "url": "",
                "metadata": {
                    "doc_id": "서울중앙지방법원-2017가단5137630",
                    "court": "서울중앙지방법원"
                }
            }
        ]
        
        result = sources_extractor._normalize_sources_detail(sources_detail)
        
        assert len(result) == 1
        assert result[0]["content"] == "판례 내용"
        assert "url" not in result[0] or result[0].get("url") is None
        assert result[0]["case_number"] == "서울중앙지방법원-2017가단5137630"
        assert result[0]["court"] == "서울중앙지방법원"


class TestGetSourcesByType:
    """sources_by_type 생성 테스트"""
    
    def test_get_sources_by_type_with_cleaning(self, sources_extractor):
        """정리된 sources_by_type 생성"""
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "content": "판례 내용",
                "score": 0.629,
                "relevance_score": 0.63,
                "chunk_id": 103351,
                "metadata": {
                    "doc_id": "서울중앙지방법원-2017가단5137630",
                    "court": "서울중앙지방법원"
                }
            },
            {
                "type": "statute_article",
                "name": "민법",
                "content": "법령 내용",
                "statute_name": "민법",
                "article_no": "제535조",
                "score": 0.8,
                "relevance_score": 0.75
            }
        ]
        
        result = sources_extractor._get_sources_by_type(sources_detail)
        
        assert "case_paragraph" in result
        assert "statute_article" in result
        assert len(result["case_paragraph"]) == 1
        assert len(result["statute_article"]) == 1
        
        # 정리 확인
        case = result["case_paragraph"][0]
        assert "relevance_score" in case
        assert "score" not in case
        assert "chunk_id" not in case
        
        statute = result["statute_article"][0]
        assert "relevance_score" in statute
        assert "score" not in statute


class TestIntegrationSourceEvent:
    """실제 API 플로우 통합 테스트"""
    
    def test_create_sources_event_with_real_data(self, sources_extractor):
        """실제 retrieved_docs 데이터로 source 이벤트 생성 테스트"""
        # 실제 검색 결과와 유사한 데이터 구조
        retrieved_docs = [
            {
                "type": "case_paragraph",
                "content": '{"text": "그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.", "score": 0.629}',
                "text": "그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.",
                "score": 0.6290625790771995,
                "relevance_score": 0.6290625790771995,
                "similarity": 0.6290625790771995,
                "cross_encoder_score": 0.6428248609148348,
                "original_score": 0.6198877245187759,
                "keyword_match_score": 0.3,
                "combined_relevance_score": 0.5303438053540397,
                "source": "exact_semantic",
                "metadata": {
                    "chunk_id": 103351,
                    "source_type": "case_paragraph",
                    "source_id": 737,
                    "text": "그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.",
                    "content": "그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.",
                    "embedding_version_id": 6,
                    "ml_confidence_score": 0.5,
                    "quality_score": 0.5,
                    "chunk_size_category": "small",
                    "chunking_strategy": "standard",
                    "id": 737,
                    "doc_id": "서울중앙지방법원-2017가단5137630",
                    "court": "서울중앙지방법원",
                    "case_type": "civil",
                    "casenames": "손해배상(기)",
                    "announce_date": "2018-07-26T09:00:00.000+09:00",
                    "query": "계약서 작성 시 주의사항은 무엇인가요?",
                    "source_type_weight": 1.0
                }
            },
            {
                "type": "case_paragraph",
                "content": '{"text": "1) 계약당사자 일방이 자신이 부담하는 계약상 채무를 이행하는 데 장애가 될 수 있는 사유를 계약을 체결할 당시에 알았거나 예견할 수 있었음에도 이를 상대방에게 고지하지 아니한 경우에는...", "score": 0.774}',
                "text": "1) 계약당사자 일방이 자신이 부담하는 계약상 채무를 이행하는 데 장애가 될 수 있는 사유를 계약을 체결할 당시에 알았거나 예견할 수 있었음에도 이를 상대방에게 고지하지 아니한 경우에는...",
                "score": 0.7745553433895112,
                "relevance_score": 0.7745553433895112,
                "similarity": 0.7745553433895112,
                "cross_encoder_score": 1.0,
                "original_score": 0.6242589056491852,
                "keyword_match_score": 0.2,
                "combined_relevance_score": 0.6021887403726578,
                "source": "exact_semantic",
                "metadata": {
                    "chunk_id": 106317,
                    "source_type": "case_paragraph",
                    "source_id": 1014,
                    "text": "1) 계약당사자 일방이 자신이 부담하는 계약상 채무를 이행하는 데 장애가 될 수 있는 사유를 계약을 체결할 당시에 알았거나 예견할 수 있었음에도 이를 상대방에게 고지하지 아니한 경우에는...",
                    "content": "1) 계약당사자 일방이 자신이 부담하는 계약상 채무를 이행하는 데 장애가 될 수 있는 사유를 계약을 체결할 당시에 알았거나 예견할 수 있었음에도 이를 상대방에게 고지하지 아니한 경우에는...",
                    "embedding_version_id": 6,
                    "ml_confidence_score": 0.5,
                    "quality_score": 0.5,
                    "chunk_size_category": "small",
                    "chunking_strategy": "standard",
                    "id": 1014,
                    "doc_id": "수원지방법원-2019가합11756",
                    "court": "수원지방법원",
                    "case_type": "civil",
                    "casenames": "손해배상등",
                    "announce_date": "2020-10-21T09:00:00.000+09:00",
                    "query": "계약서 작성 시 주의사항은 무엇인가요?",
                    "source_type_weight": 1.0
                }
            }
        ]
        
        # sources_detail 생성 (fallback 메서드 직접 사용)
        sources_detail = sources_extractor._generate_sources_detail_fallback(retrieved_docs)
        
        # 정규화 적용
        sources_detail = sources_extractor._normalize_sources_detail(sources_detail)
        
        # 검증: content가 문자열인지 확인
        assert len(sources_detail) >= 2, f"Expected at least 2 items, got {len(sources_detail)}"
        for detail in sources_detail:
            # content 필드 확인
            if "content" in detail:
                assert isinstance(detail.get("content"), str), f"content should be string, got {type(detail.get('content'))}"
                assert detail["content"] != "", "content should not be empty"
        
        # sources_by_type 생성
        sources_by_type = sources_extractor._get_sources_by_type(sources_detail)
        
        # 검증: sources_by_type 구조 확인
        assert "case_paragraph" in sources_by_type
        assert len(sources_by_type["case_paragraph"]) == 2
        
        # 각 항목 검증
        for case in sources_by_type["case_paragraph"]:
            # 필수 필드 확인
            assert "type" in case
            assert "name" in case
            assert "content" in case
            assert isinstance(case["content"], str)
            
            # 타입별 필드 확인
            assert "case_number" in case or "court" in case or "case_name" in case
            
            # 불필요한 필드 제거 확인
            assert "score" not in case
            assert "similarity" not in case
            assert "chunk_id" not in case
            assert "query" not in case
            
            # relevance_score 확인
            if "relevance_score" in case:
                assert isinstance(case["relevance_score"], (int, float))
    
    def test_create_sources_event_end_to_end(self, sources_extractor):
        """전체 플로우 통합 테스트 (sources_by_type 생성 및 정리)"""
        # 실제 metadata 구조
        sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "content": '{"text": "판례 내용", "score": 0.62}',
                "url": "",
                "case_number": "",
                "score": 0.629,
                "relevance_score": 0.63,
                "similarity": 0.629,
                "cross_encoder_score": 0.642,
                "chunk_id": 103351,
                "metadata": {
                    "doc_id": "서울중앙지방법원-2017가단5137630",
                    "court": "서울중앙지방법원",
                    "casenames": "손해배상(기)",
                    "announce_date": "2018-07-26T09:00:00.000+09:00",
                    "query": "계약서 작성 시 주의사항은 무엇인가요?",
                    "chunk_id": 103351
                }
            }
        ]
        
        # sources_by_type 생성 (실제 API에서 하는 것과 동일)
        sources_by_type = sources_extractor._get_sources_by_type_with_reference_statutes(sources_detail)
        
        # sources_by_type 구조 검증
        assert sources_by_type is not None
        assert "case_paragraph" in sources_by_type
        assert len(sources_by_type["case_paragraph"]) > 0
        
        # 각 항목 검증
        case = sources_by_type["case_paragraph"][0]
        
        # content가 문자열인지 확인
        assert isinstance(case.get("content"), str), f"content should be string, got {type(case.get('content'))}"
        assert case["content"] == "판례 내용", "content should be extracted from JSON"
        
        # 불필요한 필드 제거 확인
        assert "score" not in case, "score should be removed"
        assert "similarity" not in case, "similarity should be removed"
        assert "cross_encoder_score" not in case, "cross_encoder_score should be removed"
        assert "chunk_id" not in case, "chunk_id should be removed"
        assert "query" not in case, "query should be removed"
        
        # relevance_score만 유지
        assert "relevance_score" in case, "relevance_score should be kept"
        assert case["relevance_score"] == 0.63
        
        # 타입별 필드 확인
        assert "case_number" in case or "court" in case or "case_name" in case
        
        # 빈 필드 제거 확인
        assert "url" not in case or case.get("url"), "empty url should be removed"
    
    def test_content_normalization_various_formats(self, sources_extractor):
        """다양한 content 형식 정규화 테스트"""
        test_cases = [
            # (input, expected_output)
            ('{"text": "텍스트", "score": 0.5}', "텍스트"),
            ('{"content": "내용", "score": 0.5}', "내용"),
            ('{"text": "텍스트", "content": "내용"}', "텍스트"),  # text 우선
            ("일반 문자열", "일반 문자열"),
            ({"text": "딕셔너리 텍스트"}, "딕셔너리 텍스트"),
            ({"content": "딕셔너리 내용"}, "딕셔너리 내용"),
            ("", None),
            (None, None),
            ("   ", None),
        ]
        
        for input_content, expected in test_cases:
            result = sources_extractor._normalize_content(input_content)
            assert result == expected, f"Failed for input: {input_content}, got: {result}, expected: {expected}"
    
    def test_empty_fields_removal(self, sources_extractor):
        """빈 필드 제거 테스트"""
        data = {
            "name": "테스트",
            "url": "",
            "content": "내용",
            "case_number": "   ",
            "court": None,
            "metadata": {},
            "empty_list": [],
            "valid_field": "값"
        }
        
        result = sources_extractor._remove_empty_fields(data)
        
        assert "name" in result
        assert "content" in result
        assert "valid_field" in result
        assert "url" not in result
        assert "case_number" not in result
        assert "court" not in result
        assert "metadata" not in result
        assert "empty_list" not in result


class TestContentNewlineAndName:
    """content 줄바꿈 및 name 필드 테스트"""
    
    def test_content_newline_conversion(self, sources_extractor):
        """content의 \\n을 실제 줄바꿈으로 변환"""
        content = "첫 번째 줄\\n두 번째 줄\\n세 번째 줄"
        result = sources_extractor._normalize_content(content)
        assert result == "첫 번째 줄\n두 번째 줄\n세 번째 줄"
        assert "\n" in result
        assert "\\n" not in result
    
    def test_case_paragraph_name_with_doc_id(self, sources_extractor):
        """판례 name이 doc_id로 설정되는지 확인"""
        source_item = {
            "type": "case_paragraph",
            "name": "판례",
            "metadata": {
                "doc_id": "서울중앙지방법원-2017가단5137630",
                "court": "서울중앙지방법원"
            }
        }
        
        result = sources_extractor._clean_source_for_client(source_item)
        
        assert result["name"] == "서울중앙지방법원-2017가단5137630"
        assert result["name"] != "판례"
    
    def test_case_paragraph_name_with_case_number(self, sources_extractor):
        """판례 name이 case_number로 설정되는지 확인"""
        source_item = {
            "type": "case_paragraph",
            "name": "판례",
            "case_number": "서울중앙지방법원-2017가단5137630"
        }
        
        result = sources_extractor._clean_source_for_client(source_item)
        
        assert result["name"] == "서울중앙지방법원-2017가단5137630"
        assert result["name"] != "판례"
    
    def test_content_with_newline_in_json_string(self, sources_extractor):
        """JSON 문자열 content의 \\n이 실제 줄바꿈으로 변환되는지 확인"""
        content = '{"text": "첫 줄\\n두 번째 줄", "score": 0.5}'
        result = sources_extractor._normalize_content(content)
        assert result == "첫 줄\n두 번째 줄"
        assert "\n" in result
        assert "\\n" not in result
    
    def test_content_with_newline_in_python_dict_string(self, sources_extractor):
        """Python 딕셔너리 문자열 content의 \\n이 실제 줄바꿈으로 변환되는지 확인"""
        import ast
        content = "{'text': '첫 줄\\n두 번째 줄', 'content': '내용', 'score': 0.5}"
        result = sources_extractor._normalize_content(content)
        assert result == "첫 줄\n두 번째 줄"
        assert "\n" in result
        assert "\\n" not in result
    
    def test_real_case_data_format(self, sources_extractor):
        """실제 데이터 형식으로 전체 플로우 테스트"""
        source_item = {
            "type": "case_paragraph",
            "name": "판례",
            "content": "{'text': '그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.\\n\\n나. 예비적 청구원인\\n\\n설사 원, 피고 사이에 계약이 체결되지 않았다 하더라도, 피고는 계약 교섭단계에서 계약이 확실히 체결되리라는 정당한 기대 내지 신뢰를 원고에게 부여한 뒤 상당한 이유 없이 계약 체결을 거부하였으므로, 원고에게 그에 따른 정신적 손해를 배상할 의무가 있다.', 'content': '그런데 피고는 원고로부터 계약금 60,000,000원을 지급받고도 일방적으로 계약을 해제하였으므로, 원고에게 위약금 60,000,000원을 지급할 의무가 있다.\\n\\n나. 예비적 청구원인\\n\\n설사 원, 피고 사이에 계약이 체결되지 않았다 하더라도, 피고는 계약 교섭단계에서 계약이 확실히 체결되리라는 정당한 기대 내지 신뢰를 원고에게 부여한 뒤 상당한 이유 없이 계약 체결을 거부하였으므로, 원고에게 그에 따른 정신적 손해를 배상할 의무가 있다.', 'score': 0.6266303000864539}",
            "metadata": {
                "doc_id": "서울중앙지방법원-2017가단5137630",
                "court": "서울중앙지방법원",
                "casenames": "손해배상(기)"
            }
        }
        
        result = sources_extractor._clean_source_for_client(source_item)
        
        # name이 doc_id로 설정되었는지 확인
        assert result["name"] == "서울중앙지방법원-2017가단5137630"
        assert result["name"] != "판례"
        
        # content가 문자열로 변환되고 줄바꿈이 적용되었는지 확인
        assert "content" in result
        assert isinstance(result["content"], str)
        assert "\n" in result["content"], "content should have actual newlines"
        assert "\\n" not in result["content"], "content should not have escaped newlines"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

