# -*- coding: utf-8 -*-
"""
실제 사용자 데이터 형식으로 case_paragraph name 필드 테스트
사용자가 제공한 JSON 형식과 유사한 데이터로 테스트
"""

import sys
import os
import json
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
        return None


def test_real_user_data_format():
    """실제 사용자 데이터 형식으로 테스트"""
    print("\n" + "=" * 60)
    print("실제 사용자 데이터 형식 테스트")
    print("=" * 60)
    
    sources_extractor = get_sources_extractor()
    if not sources_extractor:
        print("❌ SourcesExtractor를 초기화할 수 없습니다.")
        return 1
    
    # 사용자가 제공한 JSON과 유사한 형식의 데이터
    # 실제로는 name 필드가 없었음
    user_case_data = {
        "type": "case_paragraph",
        "content": "1. 제1심판결 중 아래에서 지급을 명하는 금액에 해당하는 원고 패소 부분을 취소한다.\n피고는 원고에게 31,692,461원과 이에 대하여 2016. 6. 23.부터 2022. 1. 20.까지는 연 5%, 그 다음날부터 다 갚는 날까지는 연 12%의 각 비율로 계산한 돈을 지급하라..."
    }
    
    print("\n=== 테스트: 사용자 데이터 형식 (name 필드 없음) ===")
    print(f"입력 데이터: type={user_case_data.get('type')}, content 길이={len(user_case_data.get('content', ''))}")
    print(f"입력 데이터 keys: {list(user_case_data.keys())}")
    
    cleaned = sources_extractor._clean_source_for_client(user_case_data)
    
    print(f"\n출력 데이터 keys: {list(cleaned.keys())}")
    print(f"출력 데이터 name: '{cleaned.get('name')}'")
    print(f"출력 데이터 case_number: '{cleaned.get('case_number')}'")
    
    if "name" in cleaned and cleaned.get("name"):
        print(f"✅ name 필드가 설정됨: '{cleaned.get('name')}'")
        return 0
    else:
        print("❌ name 필드가 설정되지 않음")
        return 1


def test_sources_by_type_with_user_data():
    """sources_by_type 생성 테스트 (사용자 데이터 형식)"""
    print("\n" + "=" * 60)
    print("sources_by_type 생성 테스트 (사용자 데이터 형식)")
    print("=" * 60)
    
    sources_extractor = get_sources_extractor()
    if not sources_extractor:
        print("❌ SourcesExtractor를 초기화할 수 없습니다.")
        return 1
    
    # 사용자가 제공한 JSON과 유사한 sources_detail
    sources_detail = [
        {
            "type": "case_paragraph",
            "content": "1. 제1심판결 중 아래에서 지급을 명하는 금액에 해당하는 원고 패소 부분을 취소한다..."
        }
    ]
    
    print(f"\n입력 sources_detail: {len(sources_detail)}개")
    for i, detail in enumerate(sources_detail, 1):
        print(f"  {i}. type={detail.get('type')}, keys={list(detail.keys())}")
    
    sources_by_type = sources_extractor._get_sources_by_type(sources_detail)
    case_paragraphs = sources_by_type.get("case_paragraph", [])
    
    print(f"\n출력 sources_by_type['case_paragraph']: {len(case_paragraphs)}개")
    
    all_have_name = True
    for i, case_para in enumerate(case_paragraphs, 1):
        print(f"\n  {i}. keys: {list(case_para.keys())}")
        if "name" in case_para:
            print(f"      name: '{case_para.get('name')}'")
        else:
            print(f"      ❌ name 필드 없음")
            all_have_name = False
        
        if "case_number" in case_para:
            print(f"      case_number: '{case_para.get('case_number')}'")
    
    if all_have_name:
        print("\n✅ 모든 case_paragraph에 name 필드가 설정됨")
        return 0
    else:
        print("\n❌ 일부 case_paragraph에 name 필드가 없음")
        return 1


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("실제 사용자 데이터 형식 테스트 시작")
    print("=" * 60)
    
    results = []
    
    # 테스트 1: 단일 case_paragraph 데이터
    print("\n[테스트 1] 단일 case_paragraph 데이터")
    try:
        result1 = test_real_user_data_format()
        results.append(("단일 case_paragraph", result1 == 0))
    except Exception as e:
        print(f"❌ 테스트 1 실패: {e}")
        import traceback
        traceback.print_exc()
        results.append(("단일 case_paragraph", False))
    
    # 테스트 2: sources_by_type 생성
    print("\n[테스트 2] sources_by_type 생성")
    try:
        result2 = test_sources_by_type_with_user_data()
        results.append(("sources_by_type 생성", result2 == 0))
    except Exception as e:
        print(f"❌ 테스트 2 실패: {e}")
        import traceback
        traceback.print_exc()
        results.append(("sources_by_type 생성", False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n총 {total_count}개 테스트 중 {passed_count}개 통과")
    
    if passed_count == total_count:
        print("🎉 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())

