# -*- coding: utf-8 -*-
"""
Sources 데이터 개선 로직 테스트
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.services.sources_extractor import SourcesExtractor


def test_parse_source_string():
    """sources 배열 파싱 테스트"""
    extractor = SourcesExtractor(None, None)
    
    test_cases = [
        ("위자료 (case_69므37)", {"doc_id": "case_69므37", "casenames": "위자료"}),
        ("손해배상(기) (case_2011므2997)", {"doc_id": "case_2011므2997", "casenames": "손해배상(기)"}),
        ("손해배상(이혼) (case_2013므2441)", {"doc_id": "case_2013므2441", "casenames": "손해배상(이혼)"}),
    ]
    
    print("=" * 80)
    print("sources 배열 파싱 테스트")
    print("=" * 80)
    
    for source_str, expected in test_cases:
        result = extractor._parse_source_string(source_str)
        print(f"\n입력: {source_str}")
        print(f"예상: {expected}")
        print(f"결과: {result}")
        
        assert result.get("doc_id") == expected.get("doc_id"), f"doc_id 불일치: {result.get('doc_id')} != {expected.get('doc_id')}"
        assert result.get("casenames") == expected.get("casenames"), f"casenames 불일치: {result.get('casenames')} != {expected.get('casenames')}"
        print("✅ 통과")
    
    print("\n" + "=" * 80)
    print("모든 파싱 테스트 통과!")
    print("=" * 80)


def test_generate_case_url():
    """판례 URL 생성 테스트"""
    extractor = SourcesExtractor(None, None)
    
    test_cases = [
        ("case_2000르125", "http://www.law.go.kr/DRF/lawService.do?target=prec&ID=case_2000르125&type=HTML"),
        ("case_69므37", "http://www.law.go.kr/DRF/lawService.do?target=prec&ID=case_69므37&type=HTML"),
        ("", ""),
        ("invalid", ""),
    ]
    
    print("\n" + "=" * 80)
    print("판례 URL 생성 테스트")
    print("=" * 80)
    
    for doc_id, expected_url in test_cases:
        result = extractor._generate_case_url(doc_id)
        print(f"\n입력: {doc_id}")
        print(f"예상: {expected_url}")
        print(f"결과: {result}")
        
        assert result == expected_url, f"URL 불일치: {result} != {expected_url}"
        print("✅ 통과")
    
    print("\n" + "=" * 80)
    print("모든 URL 생성 테스트 통과!")
    print("=" * 80)


def test_enhance_sources_detail():
    """sources_detail 보완 테스트"""
    extractor = SourcesExtractor(None, None)
    
    sources = [
        "위자료 (case_69므37)",
        "손해배상(기) (case_2011므2997)",
        "손해배상(이혼) (case_2013므2441)",
    ]
    
    sources_detail = [
        {
            "name": "판례",
            "type": "case_paragraph",
            "url": "",
            "metadata": {
                "court": "",
                "doc_id": "case_69므37",
                "casenames": "",
                "announce_date": "",
                "case_type": None
            },
            "content": "민법 제750조"
        },
        {
            "name": "판례",
            "type": "case_paragraph",
            "url": "",
            "metadata": {
                "court": "",
                "doc_id": "",
                "casenames": "",
                "announce_date": "",
                "case_type": None
            },
            "content": "피고는 원고 B이 원고 C의 단독 대리행위를 추인하였다거나..."
        }
    ]
    
    print("\n" + "=" * 80)
    print("sources_detail 보완 테스트")
    print("=" * 80)
    
    print("\n[Before]")
    print(f"Sources: {len(sources)}개")
    print(f"Sources Detail: {len(sources_detail)}개")
    for i, detail in enumerate(sources_detail):
        print(f"  [{i}] name={detail.get('name')}, doc_id={detail.get('metadata', {}).get('doc_id')}, url={detail.get('url')}")
    
    enhanced = extractor._enhance_sources_detail_with_sources(sources, sources_detail)
    
    print("\n[After]")
    print(f"Sources Detail: {len(enhanced)}개")
    for i, detail in enumerate(enhanced):
        print(f"  [{i}] name={detail.get('name')}, doc_id={detail.get('metadata', {}).get('doc_id')}, url={detail.get('url')}, case_number={detail.get('case_number')}, case_name={detail.get('case_name')}")
    
    assert len(enhanced) >= len(sources_detail), "sources_detail이 줄어들면 안 됩니다"
    
    for detail in enhanced:
        if detail.get("type") == "case_paragraph":
            doc_id = detail.get("case_number") or detail.get("metadata", {}).get("doc_id")
            if doc_id:
                assert detail.get("url"), f"doc_id가 있으면 URL이 있어야 합니다: {doc_id}"
                assert detail.get("name") == doc_id, f"제목은 case_number만 표시해야 합니다: {detail.get('name')} != {doc_id}"
                print(f"✅ doc_id {doc_id}에 대한 URL 생성 및 제목 확인")
    
    print("\n" + "=" * 80)
    print("sources_detail 보완 테스트 통과!")
    print("=" * 80)


def test_parse_source_string_all_types():
    """모든 타입의 sources 파싱 테스트"""
    extractor = SourcesExtractor(None, None)
    
    test_cases = [
        ("위자료 (case_69므37)", {"doc_id": "case_69므37", "source_type": "case_paragraph"}),
        ("결정례 (decision_2023-001)", {"doc_id": "decision_2023-001", "source_type": "decision_paragraph"}),
        ("해석례 (interpretation_2023-002)", {"doc_id": "interpretation_2023-002", "source_type": "interpretation_paragraph"}),
        ("해석례 (expc_2023-003)", {"doc_id": "expc_2023-003", "source_type": "interpretation_paragraph"}),
    ]
    
    print("\n" + "=" * 80)
    print("모든 타입의 sources 파싱 테스트")
    print("=" * 80)
    
    for source_str, expected in test_cases:
        result = extractor._parse_source_string(source_str)
        print(f"\n입력: {source_str}")
        print(f"예상 doc_id: {expected.get('doc_id')}")
        print(f"예상 source_type: {expected.get('source_type')}")
        print(f"결과: {result}")
        
        assert result.get("doc_id") == expected.get("doc_id"), f"doc_id 불일치: {result.get('doc_id')} != {expected.get('doc_id')}"
        assert result.get("source_type") == expected.get("source_type"), f"source_type 불일치: {result.get('source_type')} != {expected.get('source_type')}"
        print("✅ 통과")
    
    print("\n" + "=" * 80)
    print("모든 타입 파싱 테스트 통과!")
    print("=" * 80)


def test_extract_legal_references_from_sources_detail():
    """sources_detail에서 legal_references 추출 테스트"""
    extractor = SourcesExtractor(None, None)
    
    sources_detail = [
        {
            "type": "case_paragraph",
            "content": "민법 제750조에 따르면 손해배상 청구가 가능합니다."
        },
        {
            "type": "case_paragraph",
            "content": "형법 제250조와 민법 제750조를 참고하세요."
        },
        {
            "type": "statute_article",
            "content": "민법 제751조는 공동불법행위에 대한 규정입니다."
        }
    ]
    
    existing_legal_refs = ["민법 제750조"]
    
    print("\n" + "=" * 80)
    print("sources_detail에서 legal_references 추출 테스트")
    print("=" * 80)
    
    print(f"\n[Before] Legal References: {existing_legal_refs}")
    
    result = extractor._extract_legal_references_from_sources_detail(sources_detail, existing_legal_refs)
    
    print(f"\n[After] Legal References: {result}")
    
    assert "민법 제750조" in result, "기존 legal_reference가 유지되어야 합니다"
    assert "형법 제250조" in result, "sources_detail에서 추출된 legal_reference가 있어야 합니다"
    assert "민법 제751조" in result, "sources_detail에서 추출된 legal_reference가 있어야 합니다"
    
    print("\n" + "=" * 80)
    print("legal_references 추출 테스트 통과!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Sources 데이터 개선 로직 테스트 시작")
    print("=" * 80)
    
    try:
        test_parse_source_string()
        test_generate_case_url()
        test_enhance_sources_detail()
        test_extract_legal_references_from_sources_detail()
        test_parse_source_string_all_types()
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

