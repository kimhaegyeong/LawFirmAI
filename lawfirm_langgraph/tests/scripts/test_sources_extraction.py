# -*- coding: utf-8 -*-
"""
Sources 추출 로직 테스트 스크립트
개선된 sources_detail 생성 로직을 테스트합니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sources_extraction():
    """sources 추출 로직 테스트"""
    print("\n" + "=" * 80)
    print("Sources 추출 로직 테스트 시작")
    print("=" * 80)
    
    try:
        from core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
        from core.agents.handlers.answer_formatter import AnswerFormatterHandler
        
        formatter = UnifiedSourceFormatter()
        answer_formatter = AnswerFormatterHandler(logger=logger)
        
        # 테스트 케이스 1: metadata가 비어있는 경우
        print("\n[테스트 1] metadata가 비어있는 판례")
        empty_metadata = {
            "court": "",
            "doc_id": "",
            "casenames": "",
            "announce_date": "",
            "case_type": None
        }
        source_info = formatter.format_source("case_paragraph", empty_metadata)
        print(f"  결과 name: {source_info.name}")
        print(f"  결과 type: {source_info.type}")
        print(f"  결과 metadata: {source_info.metadata}")
        assert source_info.name == "판례", f"예상: '판례', 실제: '{source_info.name}'"
        
        # 테스트 케이스 2: casenames가 있는 경우
        print("\n[테스트 2] casenames가 있는 판례")
        metadata_with_casenames = {
            "court": "",
            "doc_id": "2014두38149",
            "casenames": "등록세등부과처분취소",
            "announce_date": "",
            "case_type": None
        }
        source_info = formatter.format_source("case_paragraph", metadata_with_casenames)
        print(f"  결과 name: {source_info.name}")
        print(f"  결과 type: {source_info.type}")
        print(f"  결과 metadata: {source_info.metadata}")
        assert source_info.name == "등록세등부과처분취소", f"예상: '등록세등부과처분취소', 실제: '{source_info.name}'"
        
        # 테스트 케이스 3: doc_id만 있는 경우
        print("\n[테스트 3] doc_id만 있는 판례")
        metadata_with_doc_id = {
            "court": "",
            "doc_id": "2014두38149",
            "casenames": "",
            "announce_date": "",
            "case_type": None
        }
        source_info = formatter.format_source("case_paragraph", metadata_with_doc_id)
        print(f"  결과 name: {source_info.name}")
        print(f"  결과 type: {source_info.type}")
        assert source_info.name == "2014두38149", f"예상: '2014두38149', 실제: '{source_info.name}'"
        
        # 테스트 케이스 4: sources 배열에서 판례명 추출 테스트
        print("\n[테스트 4] sources 배열에서 판례명 추출")
        test_sources = [
            "등록세등부과처분취소 (case_2014두38149)",
            "부가가치세부과처분취소 (case_2014두13393)",
            "전기통신사업법위반·대부업등의등록및금융이용자보호에관한법률위반 (case_2019도4368)"
        ]
        test_sources_detail = [
            {
                "name": "판례",
                "type": "case_paragraph",
                "metadata": {
                    "court": "",
                    "doc_id": "",
                    "casenames": "",
                    "announce_date": "",
                    "case_type": None
                }
            },
            {
                "name": "판례",
                "type": "case_paragraph",
                "metadata": {
                    "court": "",
                    "doc_id": "",
                    "casenames": "",
                    "announce_date": "",
                    "case_type": None
                }
            },
            {
                "name": "판례",
                "type": "case_paragraph",
                "metadata": {
                    "court": "",
                    "doc_id": "",
                    "casenames": "",
                    "announce_date": "",
                    "case_type": None
                }
            }
        ]
        
        # fallback 로직 시뮬레이션
        for idx, detail in enumerate(test_sources_detail):
            if detail.get("name") in ("판례", "") and idx < len(test_sources):
                source_str = test_sources[idx]
                if source_str and source_str != "판례":
                    if "(" in source_str:
                        case_name = source_str.split("(")[0].strip()
                        doc_id_match = source_str[source_str.find("(")+1:source_str.find(")")]
                        if doc_id_match:
                            clean_doc_id = doc_id_match.replace("case_", "").strip()
                            if clean_doc_id and not detail.get("case_number"):
                                detail["case_number"] = clean_doc_id
                                if "metadata" in detail and isinstance(detail["metadata"], dict):
                                    detail["metadata"]["doc_id"] = clean_doc_id
                        
                        if case_name:
                            detail["name"] = case_name
                            if "metadata" in detail and isinstance(detail["metadata"], dict):
                                detail["metadata"]["casenames"] = case_name
                            detail["case_name"] = case_name
        
        print(f"  원본 sources: {test_sources}")
        print(f"  개선된 sources_detail:")
        for idx, detail in enumerate(test_sources_detail, 1):
            print(f"    [{idx}] name: {detail.get('name')}, case_name: {detail.get('case_name')}, case_number: {detail.get('case_number')}")
            assert detail.get("name") != "판례", f"판례 {idx}: name이 '판례'로 남아있음"
            assert detail.get("case_name") is not None, f"판례 {idx}: case_name이 없음"
            assert detail.get("case_number") is not None, f"판례 {idx}: case_number가 없음"
        
        # 테스트 케이스 5: casenames를 case_name으로 변환 테스트
        print("\n[테스트 5] casenames를 case_name으로 변환")
        test_detail = {
            "name": "판례",
            "type": "case_paragraph",
            "metadata": {
                "court": "대법원",
                "doc_id": "2014두38149",
                "casenames": "등록세등부과처분취소",
                "announce_date": "",
                "case_type": None
            }
        }
        
        # _create_source_detail_dict 로직 시뮬레이션
        if test_detail.get("metadata", {}).get("casenames"):
            test_detail["case_name"] = test_detail["metadata"]["casenames"]
        
        print(f"  결과 case_name: {test_detail.get('case_name')}")
        assert test_detail.get("case_name") == "등록세등부과처분취소", f"예상: '등록세등부과처분취소', 실제: '{test_detail.get('case_name')}'"
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_sources_extraction()

