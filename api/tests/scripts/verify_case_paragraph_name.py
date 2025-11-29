# -*- coding: utf-8 -*-
"""
case_paragraph name 필드 검증 스크립트
실제 API 응답이나 sources_by_type에서 name 필드가 제대로 설정되었는지 확인
"""

import sys
import os
import json
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
api_dir = tests_dir.parent
project_root = api_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "api") not in sys.path:
    sys.path.insert(0, str(project_root / "api"))


def verify_sources_by_type(sources_by_type: dict) -> bool:
    """sources_by_type에서 case_paragraph의 name 필드 검증"""
    print("\n" + "=" * 60)
    print("case_paragraph name 필드 검증")
    print("=" * 60)
    
    case_paragraphs = sources_by_type.get("case_paragraph", [])
    
    if not case_paragraphs:
        print("⚠️ case_paragraph가 없습니다.")
        return True  # case_paragraph가 없으면 검증할 것이 없으므로 통과
    
    print(f"\n총 {len(case_paragraphs)}개의 case_paragraph 발견")
    
    all_have_name = True
    for i, case_para in enumerate(case_paragraphs, 1):
        name = case_para.get("name", "")
        case_number = case_para.get("case_number", "")
        doc_id = case_para.get("metadata", {}).get("doc_id", "") if isinstance(case_para.get("metadata"), dict) else ""
        
        if not name:
            print(f"❌ case_paragraph {i}: name 필드 없음")
            print(f"   keys: {list(case_para.keys())}")
            if case_number:
                print(f"   case_number: '{case_number}'")
            if doc_id:
                print(f"   doc_id: '{doc_id}'")
            all_have_name = False
        else:
            print(f"✅ case_paragraph {i}: name='{name}'")
            if case_number and name != case_number:
                print(f"   ⚠️ case_number='{case_number}' (name과 다름)")
    
    if all_have_name:
        print("\n✅ 모든 case_paragraph에 name 필드가 설정되었습니다!")
        return True
    else:
        print("\n❌ 일부 case_paragraph에 name 필드가 없습니다!")
        return False


def main():
    """메인 함수"""
    # 사용자가 제공한 JSON 형식의 예시
    example_sources_by_type = {
        "statute_article": [],
        "case_paragraph": [
            {
                "type": "case_paragraph",
                "content": "1. 제1심판결 중 아래에서 지급을 명하는 금액에 해당하는 원고 패소 부분을 취소한다..."
            }
        ],
        "decision_paragraph": [],
        "interpretation_paragraph": [],
        "regulation_paragraph": []
    }
    
    print("=" * 60)
    print("case_paragraph name 필드 검증 테스트")
    print("=" * 60)
    
    # 예시 데이터로 테스트
    print("\n[테스트 1] 예시 데이터 (name 필드 없음)")
    result1 = verify_sources_by_type(example_sources_by_type)
    
    # 개선된 데이터 (name 필드 있음)
    improved_sources_by_type = {
        "statute_article": [],
        "case_paragraph": [
            {
                "type": "case_paragraph",
                "name": "2020다12345",  # name 필드 추가
                "case_number": "2020다12345",
                "content": "1. 제1심판결 중 아래에서 지급을 명하는 금액에 해당하는 원고 패소 부분을 취소한다..."
            }
        ],
        "decision_paragraph": [],
        "interpretation_paragraph": [],
        "regulation_paragraph": []
    }
    
    print("\n[테스트 2] 개선된 데이터 (name 필드 있음)")
    result2 = verify_sources_by_type(improved_sources_by_type)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    
    if result1 and result2:
        print("✅ 모든 테스트 통과!")
        print("\n💡 실제 API 응답에서도 _clean_source_for_client가 호출되면")
        print("   name 필드가 자동으로 설정됩니다.")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())

