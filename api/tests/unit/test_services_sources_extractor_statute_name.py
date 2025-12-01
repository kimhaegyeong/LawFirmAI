# -*- coding: utf-8 -*-
"""
statute_article의 statute_name이 content에서 추출되는지 테스트
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


def test_statute_name_from_content():
    """content에서 법령명 추출 테스트"""
    print("\n" + "=" * 60)
    print("statute_article name 필드 설정 테스트 (content에서 추출)")
    print("=" * 60)
    
    sources_extractor = get_sources_extractor()
    if not sources_extractor:
        print("❌ SourcesExtractor를 초기화할 수 없습니다.")
        return 1
    
    test_results = {
        "statute_name_from_content_민법": False,
        "statute_name_from_content_형법": False,
        "statute_name_from_content_상법": False,
        "statute_name_from_content_with_article": False,
        "statute_name_already_set": False,
        "statute_name_from_metadata": False,
    }
    
    # === 테스트 1: content에서 "민법 제750조" 추출 ===
    print("\n=== 테스트 1: content에서 '민법 제750조' 추출 ===")
    statute_item_content_민법 = {
        "type": "statute_article",
        "statute_name": "법령",  # "법령"으로 설정되어 있음
        "content": "나) 피고들은 원고에게 민법 제750조 불법행위에 기한 손해배상책임 또는 민법 제758조 공작물 소유자의 책임에 근거하여 원고가 입은 손해를 배상할 의무가 있다.",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_content_민법)
    if cleaned.get("name") == "민법" and cleaned.get("statute_name") == "민법":
        print("✅ content에서 '민법' 추출 성공: name='민법', statute_name='민법'")
        test_results["statute_name_from_content_민법"] = True
    else:
        print(f"❌ content에서 '민법' 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '민법')")
    
    # === 테스트 2: content에서 "형법 제XXX조" 추출 ===
    print("\n=== 테스트 2: content에서 '형법 제XXX조' 추출 ===")
    statute_item_content_형법 = {
        "type": "statute_article",
        "statute_name": "법령",
        "content": "따라서 특별한 사정이 없는 한, 피고는 원고들에게 형법 제250조 또는 부정경쟁방지법 제5조에 따라 그로 인한 손해를 배상할 책임이 있다.",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_content_형법)
    # content에 "형법"과 "부정경쟁방지법"이 모두 있지만, 첫 번째로 매칭되는 것을 사용
    if cleaned.get("name") in ["형법", "부정경쟁방지법"] and cleaned.get("statute_name") in ["형법", "부정경쟁방지법"]:
        print(f"✅ content에서 법령명 추출 성공: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}'")
        test_results["statute_name_from_content_형법"] = True
    else:
        print(f"❌ content에서 법령명 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}'")
    
    # === 테스트 3: content에서 "상법 제XXX조" 추출 ===
    print("\n=== 테스트 3: content에서 '상법 제XXX조' 추출 ===")
    statute_item_content_상법 = {
        "type": "statute_article",
        "statute_name": "법령",
        "content": "[1] [1] 상법 제750조, 제806조 제843조 / [2] 상법 제750조, 제806조 , 제843조 / [3] 제396조 , 제763조 , 제806조 제843조",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_content_상법)
    if cleaned.get("name") == "상법" and cleaned.get("statute_name") == "상법":
        print("✅ content에서 '상법' 추출 성공: name='상법', statute_name='상법'")
        test_results["statute_name_from_content_상법"] = True
    else:
        print(f"❌ content에서 '상법' 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '상법')")
    
    # === 테스트 4: content에 조문 번호가 있는 경우 ===
    print("\n=== 테스트 4: content에 조문 번호가 있는 경우 ===")
    statute_item_with_article = {
        "type": "statute_article",
        "statute_name": "법령",
        "content": "민법 제750조",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_with_article)
    if cleaned.get("name") == "민법" and cleaned.get("statute_name") == "민법":
        print("✅ content에서 '민법' 추출 성공 (조문 번호 포함): name='민법', statute_name='민법'")
        test_results["statute_name_from_content_with_article"] = True
    else:
        print(f"❌ content에서 '민법' 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '민법')")
    
    # === 테스트 5: statute_name이 이미 올바르게 설정된 경우 ===
    print("\n=== 테스트 5: statute_name이 이미 올바르게 설정된 경우 ===")
    statute_item_already_set = {
        "type": "statute_article",
        "statute_name": "민법",
        "content": "민법 제750조",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_already_set)
    if cleaned.get("name") == "민법" and cleaned.get("statute_name") == "민법":
        print("✅ statute_name이 이미 설정된 경우: name='민법', statute_name='민법'")
        test_results["statute_name_already_set"] = True
    else:
        print(f"❌ statute_name이 이미 설정된 경우 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '민법')")
    
    # === 테스트 6: metadata에서 law_name이 있는 경우 ===
    print("\n=== 테스트 6: metadata에서 law_name이 있는 경우 ===")
    statute_item_metadata = {
        "type": "statute_article",
        "statute_name": "법령",
        "content": "민법 제750조",
        "metadata": {
            "law_name": "민법"
        }
    }
    cleaned = sources_extractor._clean_source_for_client(statute_item_metadata)
    if cleaned.get("name") == "민법" and cleaned.get("statute_name") == "민법":
        print("✅ metadata에서 law_name 추출 성공: name='민법', statute_name='민법'")
        test_results["statute_name_from_metadata"] = True
    else:
        print(f"❌ metadata에서 law_name 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '민법')")
    
    # === 테스트 7: 실제 사용자 데이터 형식 (사용자가 제공한 JSON) ===
    print("\n=== 테스트 7: 실제 사용자 데이터 형식 ===")
    user_statute_data = {
        "type": "statute_article",
        "statute_name": "법령",
        "content": "나) 피고들은 원고에게 민법 제750조 불법행위에 기한 손해배상책임 또는 민법 제758조 공작물 소유자의 책임에 근거하여 원고가 입은 손해를 배상할 의무가 있다.",
        "metadata": {}
    }
    cleaned = sources_extractor._clean_source_for_client(user_statute_data)
    if cleaned.get("name") == "민법" and cleaned.get("statute_name") == "민법":
        print("✅ 실제 사용자 데이터 형식에서 '민법' 추출 성공: name='민법', statute_name='민법'")
        test_results["user_data_format"] = True
    else:
        print(f"❌ 실제 사용자 데이터 형식에서 '민법' 추출 실패: name='{cleaned.get('name')}', statute_name='{cleaned.get('statute_name')}' (예상: '민법')")
    
    # === 테스트 8: sources_by_type 생성 테스트 ===
    print("\n=== 테스트 8: sources_by_type 생성 테스트 ===")
    sources_detail = [
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "나) 피고들은 원고에게 민법 제750조 불법행위에 기한 손해배상책임 또는 민법 제758조 공작물 소유자의 책임에 근거하여 원고가 입은 손해를 배상할 의무가 있다.",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "따라서 특별한 사정이 없는 한, 피고는 원고들에게 민법 제750조 또는 부정경쟁방지법 제5조에 따라 그로 인한 손해를 배상할 책임이 있다.",
            "metadata": {}
        },
        {
            "type": "case_paragraph",
            "content": "1) 원고가 피고 주택도시보증공사에 대하여 갖는 하자보수보증금채권은...",
            "metadata": {}
        }
    ]
    
    sources_by_type = sources_extractor._get_sources_by_type(sources_detail)
    statute_articles = sources_by_type.get("statute_article", [])
    case_paragraphs = sources_by_type.get("case_paragraph", [])
    
    print(f"   statute_article: {len(statute_articles)}개")
    print(f"   case_paragraph: {len(case_paragraphs)}개")
    
    all_statutes_have_name = True
    for i, statute in enumerate(statute_articles, 1):
        name = statute.get("name", "")
        statute_name = statute.get("statute_name", "")
        if name and name != "법령" and statute_name and statute_name != "법령":
            print(f"   ✅ statute_article {i}: name='{name}', statute_name='{statute_name}'")
        else:
            print(f"   ❌ statute_article {i}: name='{name}', statute_name='{statute_name}' (예상: '민법' 또는 다른 법령명)")
            all_statutes_have_name = False
    
    all_cases_have_name = True
    for i, case_para in enumerate(case_paragraphs, 1):
        name = case_para.get("name", "")
        if name:
            print(f"   ✅ case_paragraph {i}: name='{name}'")
        else:
            print(f"   ❌ case_paragraph {i}: name 필드 없음")
            all_cases_have_name = False
    
    if all_statutes_have_name and all_cases_have_name:
        print("✅ 모든 sources에 name 필드가 올바르게 설정됨")
        test_results["sources_by_type"] = True
    else:
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
    sys.exit(test_statute_name_from_content())

