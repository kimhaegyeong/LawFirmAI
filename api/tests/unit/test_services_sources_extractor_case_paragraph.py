# -*- coding: utf-8 -*-
"""
case_paragraph name 필드 설정 테스트
sources_extractor의 _clean_source_for_client에서 case_paragraph의 name이 제대로 설정되는지 확인
"""

import sys
import os
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'api'))

from api.services.sources_extractor import SourcesExtractor

# SourcesExtractor 초기화 (간단한 Mock 사용)
def get_sources_extractor():
    """SourcesExtractor 인스턴스 생성"""
    try:
        # SourcesExtractor는 workflow_service와 session_service가 필요하지만,
        # _clean_source_for_client는 이들을 사용하지 않으므로 Mock으로 처리
        class MockWorkflowService:
            pass
        
        class MockSessionService:
            pass
        
        workflow_service = MockWorkflowService()
        session_service = MockSessionService()
        
        extractor = SourcesExtractor(workflow_service, session_service)
        return extractor
    except Exception as e:
        print(f"⚠️ SourcesExtractor 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_case_paragraph_name():
    """case_paragraph의 name 필드 설정 테스트"""
    print("\n" + "=" * 60)
    print("case_paragraph name 필드 설정 테스트 시작")
    print("=" * 60)
    
    sources_extractor = get_sources_extractor()
    if not sources_extractor:
        print("❌ SourcesExtractor를 초기화할 수 없습니다.")
        return 1
    
    test_results = {
        "case_number_exists": False,
        "doc_id_in_metadata": False,
        "doc_id_in_top_level": False,
        "no_identifiers": False,
        "case_number_in_metadata": False,
    }
    
    # === 테스트 1: case_number가 최상위 레벨에 있는 경우 ===
    print("\n=== 테스트 1: case_number가 최상위 레벨에 있는 경우 ===")
    case_item_with_case_number = {
        "type": "case_paragraph",
        "case_number": "2020다12345",
        "content": "판례 내용",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(case_item_with_case_number)
    if cleaned.get("name") == "2020다12345" and cleaned.get("case_number") == "2020다12345":
        print("✅ case_number가 최상위 레벨에 있는 경우: name='2020다12345', case_number='2020다12345'")
        test_results["case_number_exists"] = True
    else:
        print(f"❌ case_number가 최상위 레벨에 있는 경우: name='{cleaned.get('name')}', case_number='{cleaned.get('case_number')}' (예상: '2020다12345')")
    
    # === 테스트 2: case_number가 metadata에 있는 경우 ===
    print("\n=== 테스트 2: case_number가 metadata에 있는 경우 ===")
    case_item_case_number_in_metadata = {
        "type": "case_paragraph",
        "content": "판례 내용",
        "metadata": {
            "case_number": "2020다12345"
        }
    }
    cleaned = sources_extractor._clean_source_for_client(case_item_case_number_in_metadata)
    if cleaned.get("name") == "2020다12345" or cleaned.get("case_number") == "2020다12345":
        print("✅ case_number가 metadata에 있는 경우: name 또는 case_number='2020다12345'")
        test_results["case_number_in_metadata"] = True
    else:
        print(f"❌ case_number가 metadata에 있는 경우: name='{cleaned.get('name')}', case_number='{cleaned.get('case_number')}' (예상: '2020다12345')")
    
    # === 테스트 3: doc_id가 metadata에 있는 경우 ===
    print("\n=== 테스트 3: doc_id가 metadata에 있는 경우 ===")
    case_item_doc_id_in_metadata = {
        "type": "case_paragraph",
        "content": "판례 내용",
        "metadata": {
            "doc_id": "2020다12345"
        }
    }
    cleaned = sources_extractor._clean_source_for_client(case_item_doc_id_in_metadata)
    if cleaned.get("name") == "2020다12345":
        print("✅ doc_id가 metadata에 있는 경우: name='2020다12345'")
        test_results["doc_id_in_metadata"] = True
    else:
        print(f"❌ doc_id가 metadata에 있는 경우: name='{cleaned.get('name')}' (예상: '2020다12345')")
    
    # === 테스트 4: doc_id가 최상위 레벨에 있는 경우 ===
    print("\n=== 테스트 4: doc_id가 최상위 레벨에 있는 경우 ===")
    case_item_doc_id_top_level = {
        "type": "case_paragraph",
        "doc_id": "2020다12345",
        "content": "판례 내용",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(case_item_doc_id_top_level)
    if cleaned.get("name") == "2020다12345":
        print("✅ doc_id가 최상위 레벨에 있는 경우: name='2020다12345'")
        test_results["doc_id_in_top_level"] = True
    else:
        print(f"❌ doc_id가 최상위 레벨에 있는 경우: name='{cleaned.get('name')}' (예상: '2020다12345')")
    
    # === 테스트 5: case_number와 doc_id가 모두 없는 경우 ===
    print("\n=== 테스트 5: case_number와 doc_id가 모두 없는 경우 ===")
    case_item_no_identifiers = {
        "type": "case_paragraph",
        "content": "판례 내용",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(case_item_no_identifiers)
    # name이 "판례"로 설정되어야 함 (최소한 표시는 되도록)
    if cleaned.get("name") == "판례":
        print("✅ case_number와 doc_id가 모두 없는 경우: name='판례' (기본값)")
        test_results["no_identifiers"] = True
    elif cleaned.get("name"):
        print(f"⚠️ case_number와 doc_id가 모두 없는 경우: name='{cleaned.get('name')}' (예상: '판례', 하지만 name이 있으면 통과)")
        test_results["no_identifiers"] = True
    else:
        print(f"❌ case_number와 doc_id가 모두 없는 경우: name이 없음 (예상: '판례' 또는 다른 값)")
    
    # === 테스트 6: 실제 sources_by_type 생성 테스트 ===
    print("\n=== 테스트 6: 실제 sources_by_type 생성 테스트 ===")
    sources_detail = [
        {
            "type": "case_paragraph",
            "case_number": "2020다12345",
            "content": "판례 내용",
            "metadata": {}
        },
        {
            "type": "case_paragraph",
            "content": "판례 내용",
            "metadata": {
                "doc_id": "2020다67890"
            }
        },
        {
            "type": "case_paragraph",
            "content": "판례 내용",
            "metadata": {}
        }
    ]
    
    sources_by_type = sources_extractor._get_sources_by_type(sources_detail)
    case_paragraphs = sources_by_type.get("case_paragraph", [])
    
    if len(case_paragraphs) == 3:
        print(f"✅ sources_by_type에 3개의 case_paragraph가 포함됨")
        
        # 각 case_paragraph에 name이 있는지 확인
        all_have_name = True
        for i, case_para in enumerate(case_paragraphs, 1):
            if "name" not in case_para or not case_para.get("name"):
                print(f"❌ case_paragraph {i}에 name 필드가 없음: {list(case_para.keys())}")
                all_have_name = False
            else:
                print(f"✅ case_paragraph {i}: name='{case_para.get('name')}'")
        
        if all_have_name:
            print("✅ 모든 case_paragraph에 name 필드가 설정됨")
            test_results["sources_by_type"] = True
        else:
            test_results["sources_by_type"] = False
    else:
        print(f"❌ sources_by_type에 case_paragraph가 {len(case_paragraphs)}개만 포함됨 (예상: 3개)")
        test_results["sources_by_type"] = False
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed_count = sum(1 for passed in test_results.values() if passed)
    total_count = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name.replace('_', ' ').capitalize()}: {status}")
    
    print(f"\n총 {total_count}개 테스트 중 {passed_count}개 통과")
    
    if passed_count == total_count:
        print("🎉 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(test_case_paragraph_name())


