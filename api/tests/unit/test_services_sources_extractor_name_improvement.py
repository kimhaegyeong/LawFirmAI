# -*- coding: utf-8 -*-
"""
Source name 개선 테스트 스크립트
statute_name, case_number, decision_number, interpretation_number가 제대로 표시되는지 테스트
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from api.services.sources_extractor import SourcesExtractor
from api.services.chat_service import get_chat_service


def test_statute_article_name():
    """법령명이 제대로 표시되는지 테스트"""
    print("\n=== 법령명 표시 테스트 ===")
    
    chat_service = get_chat_service()
    if not chat_service or not hasattr(chat_service, 'sources_extractor'):
        print("❌ SourcesExtractor를 가져올 수 없습니다.")
        return False
    
    extractor = chat_service.sources_extractor
    
    # 테스트 케이스 1: statute_name이 "법령"인 경우
    test_cases = [
        {
            "name": "statute_name이 '법령'인 경우",
            "source_item": {
                "type": "statute_article",
                "statute_name": "법령",
                "article_no": "750",
                "metadata": {
                    "law_name": "민법",
                    "article_no": "750"
                }
            },
            "expected_name": "민법"
        },
        {
            "name": "statute_name이 없고 law_name만 있는 경우",
            "source_item": {
                "type": "statute_article",
                "article_no": "750",
                "metadata": {
                    "law_name": "민법",
                    "article_no": "750"
                }
            },
            "expected_name": "민법"
        },
        {
            "name": "statute_name과 law_name이 모두 없고 abbrv만 있는 경우",
            "source_item": {
                "type": "statute_article",
                "article_no": "750",
                "metadata": {
                    "abbrv": "민법",
                    "article_no": "750"
                }
            },
            "expected_name": "민법"
        },
        {
            "name": "statute_name이 정상인 경우",
            "source_item": {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "750",
                "metadata": {
                    "statute_name": "민법",
                    "article_no": "750"
                }
            },
            "expected_name": "민법"
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        cleaned = extractor._clean_source_for_client(test_case["source_item"])
        name = cleaned.get("name", "")
        statute_name = cleaned.get("statute_name", "")
        
        # 디버깅: 입력값과 출력값 상세 로그 (실패한 경우만)
        if name != test_case["expected_name"] and statute_name != test_case["expected_name"]:
            print(f"\n[DEBUG] {test_case['name']}:")
            print(f"  입력: statute_name={test_case['source_item'].get('statute_name')}, metadata.law_name={test_case['source_item'].get('metadata', {}).get('law_name')}")
            print(f"  출력: name='{name}', statute_name='{statute_name}'")
            print(f"  예상: '{test_case['expected_name']}'")
        
        if name == test_case["expected_name"] or statute_name == test_case["expected_name"]:
            print(f"✅ {test_case['name']}: name='{name}', statute_name='{statute_name}'")
        else:
            print(f"❌ {test_case['name']}: name='{name}', statute_name='{statute_name}' (예상: '{test_case['expected_name']}')")
            all_passed = False
    
    return all_passed


def test_case_paragraph_name():
    """판례 번호가 제대로 표시되는지 테스트"""
    print("\n=== 판례 번호 표시 테스트 ===")
    
    chat_service = get_chat_service()
    if not chat_service or not hasattr(chat_service, 'sources_extractor'):
        print("❌ SourcesExtractor를 가져올 수 없습니다.")
        return False
    
    extractor = chat_service.sources_extractor
    
    test_cases = [
        {
            "name": "case_number가 있는 경우",
            "source_item": {
                "type": "case_paragraph",
                "case_number": "2020다12345",
                "metadata": {
                    "doc_id": "2020다12345"
                }
            },
            "expected_name": "2020다12345"
        },
        {
            "name": "case_number가 없고 doc_id만 있는 경우",
            "source_item": {
                "type": "case_paragraph",
                "metadata": {
                    "doc_id": "2020다12345"
                }
            },
            "expected_name": "2020다12345"
        },
        {
            "name": "name이 '판례'인 경우",
            "source_item": {
                "type": "case_paragraph",
                "name": "판례",
                "case_number": "2020다12345",
                "metadata": {
                    "doc_id": "2020다12345"
                }
            },
            "expected_name": "2020다12345"
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        cleaned = extractor._clean_source_for_client(test_case["source_item"])
        name = cleaned.get("name", "")
        
        if name == test_case["expected_name"]:
            print(f"✅ {test_case['name']}: name='{name}'")
        else:
            print(f"❌ {test_case['name']}: name='{name}' (예상: '{test_case['expected_name']}')")
            all_passed = False
    
    return all_passed


def test_decision_paragraph_name():
    """결정례 번호가 제대로 표시되는지 테스트"""
    print("\n=== 결정례 번호 표시 테스트 ===")
    
    chat_service = get_chat_service()
    if not chat_service or not hasattr(chat_service, 'sources_extractor'):
        print("❌ SourcesExtractor를 가져올 수 없습니다.")
        return False
    
    extractor = chat_service.sources_extractor
    
    test_cases = [
        {
            "name": "decision_number가 있는 경우",
            "source_item": {
                "type": "decision_paragraph",
                "decision_number": "2020결정123",
                "metadata": {
                    "doc_id": "2020결정123"
                }
            },
            "expected_name": "2020결정123"
        },
        {
            "name": "decision_number가 없고 doc_id만 있는 경우",
            "source_item": {
                "type": "decision_paragraph",
                "metadata": {
                    "doc_id": "2020결정123"
                }
            },
            "expected_name": "2020결정123"
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        cleaned = extractor._clean_source_for_client(test_case["source_item"])
        name = cleaned.get("name", "")
        
        if name == test_case["expected_name"]:
            print(f"✅ {test_case['name']}: name='{name}'")
        else:
            print(f"❌ {test_case['name']}: name='{name}' (예상: '{test_case['expected_name']}')")
            all_passed = False
    
    return all_passed


def test_interpretation_paragraph_name():
    """해석례 번호가 제대로 표시되는지 테스트"""
    print("\n=== 해석례 번호 표시 테스트 ===")
    
    chat_service = get_chat_service()
    if not chat_service or not hasattr(chat_service, 'sources_extractor'):
        print("❌ SourcesExtractor를 가져올 수 없습니다.")
        return False
    
    extractor = chat_service.sources_extractor
    
    test_cases = [
        {
            "name": "interpretation_number가 있는 경우",
            "source_item": {
                "type": "interpretation_paragraph",
                "interpretation_number": "2020해석123",
                "metadata": {
                    "doc_id": "2020해석123"
                }
            },
            "expected_name": "2020해석123"
        },
        {
            "name": "interpretation_number가 없고 doc_id만 있는 경우",
            "source_item": {
                "type": "interpretation_paragraph",
                "metadata": {
                    "doc_id": "2020해석123"
                }
            },
            "expected_name": "2020해석123"
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        cleaned = extractor._clean_source_for_client(test_case["source_item"])
        name = cleaned.get("name", "")
        
        if name == test_case["expected_name"]:
            print(f"✅ {test_case['name']}: name='{name}'")
        else:
            print(f"❌ {test_case['name']}: name='{name}' (예상: '{test_case['expected_name']}')")
            all_passed = False
    
    return all_passed


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("Source Name 개선 테스트 시작")
    print("=" * 60)
    
    results = []
    
    # 각 테스트 실행
    results.append(("법령명 표시", test_statute_article_name()))
    results.append(("판례 번호 표시", test_case_paragraph_name()))
    results.append(("결정례 번호 표시", test_decision_paragraph_name()))
    results.append(("해석례 번호 표시", test_interpretation_paragraph_name()))
    
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

