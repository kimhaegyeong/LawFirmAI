#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
용어 추출 기능 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.processing.extractors.document_extractor import DocumentExtractor

def test_term_extraction():
    """용어 추출 테스트"""
    print("=" * 80)
    print("용어 추출 기능 테스트")
    print("=" * 80)
    
    # 테스트 문서 데이터
    test_docs = [
        {
            "content": """
            임대차 분쟁 시 해결 방법은 무엇인가요?
            임대차 계약에서 임대인과 임차인 간의 분쟁이 발생할 수 있습니다.
            민법 제618조에 따르면 임대인은 임차인에게 목적물을 사용하게 할 의무가 있습니다.
            또한 임차인은 임대인에게 차임을 지급할 의무가 있습니다.
            계약 해지 시 손해배상 책임이 발생할 수 있으며, 이에 대한 판례가 있습니다.
            대법원 2020다12345 판결에 따르면 임대차 계약 해지 시 적절한 통지가 필요합니다.
            """
        },
        {
            "content": """
            계약 해지에 대한 법률적 근거를 알려주세요.
            계약 해지는 민법 제543조에 규정되어 있으며, 계약 당사자는 상당한 기간을 정하여
            이행을 최고하고 그 기간 내에 이행하지 아니한 때에는 계약을 해지할 수 있습니다.
            손해배상 청구는 불법행위에 기한 손해배상과 계약 위반에 기한 손해배상으로 구분됩니다.
            """
        },
        {
            "content": """
            판례에서 인정하는 손해배상 범위는 어떻게 되나요?
            대법원 판례에 따르면 손해배상의 범위는 통상의 손해와 특별 손해로 구분됩니다.
            민법 제393조에 따르면 채무불이행으로 인한 손해배상은 통상의 손해를 그 범위로 합니다.
            계약 위반 시 계약금 반환 문제도 발생할 수 있습니다.
            """
        }
    ]
    
    print(f"\n테스트 문서 수: {len(test_docs)}개")
    print(f"각 문서의 평균 길이: {sum(len(doc.get('content', '')) for doc in test_docs) / len(test_docs):.0f}자\n")
    
    # 테스트 1: 기본 용어 추출 (최대 1000개)
    print("-" * 80)
    print("테스트 1: 기본 용어 추출 (최대 1000개)")
    print("-" * 80)
    
    try:
        terms = DocumentExtractor.extract_terms_from_documents(test_docs)
        print(f"✅ 추출 성공: {len(terms)}개 용어")
        print(f"\n상위 20개 용어:")
        for i, term in enumerate(terms[:20], 1):
            print(f"  {i:2d}. {term}")
        
        if len(terms) > 20:
            print(f"  ... (총 {len(terms)}개)")
    except Exception as e:
        print(f"❌ 추출 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 테스트 2: 최대 용어 수 제한 (100개)
    print("\n" + "-" * 80)
    print("테스트 2: 최대 용어 수 제한 (100개)")
    print("-" * 80)
    
    try:
        terms_limited = DocumentExtractor.extract_terms_from_documents(test_docs, max_terms=100)
        print(f"✅ 추출 성공: {len(terms_limited)}개 용어")
        print(f"\n추출된 용어 (최대 100개):")
        for i, term in enumerate(terms_limited[:30], 1):
            print(f"  {i:2d}. {term}")
        
        if len(terms_limited) > 30:
            print(f"  ... (총 {len(terms_limited)}개)")
    except Exception as e:
        print(f"❌ 추출 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 테스트 3: 불용어 제거 확인
    print("\n" + "-" * 80)
    print("테스트 3: 불용어 제거 확인")
    print("-" * 80)
    
    common_stopwords = ['것', '이', '그', '및', '또한', '따라서', '그러나', '하지만', 
                        '때문', '위해', '대해', '관련', '등', '또는', '무엇인가요', 
                        '알려주세요', '설명해주세요']
    
    found_stopwords = [term for term in terms if term in common_stopwords]
    
    if found_stopwords:
        print(f"⚠️ 불용어가 추출됨: {found_stopwords}")
    else:
        print("✅ 불용어가 제대로 제거됨")
    
    # 테스트 4: 중복 제거 확인
    print("\n" + "-" * 80)
    print("테스트 4: 중복 제거 확인")
    print("-" * 80)
    
    unique_terms = set(terms)
    if len(terms) == len(unique_terms):
        print(f"✅ 중복 없음: {len(terms)}개 모두 고유")
    else:
        print(f"⚠️ 중복 발견: {len(terms)}개 중 {len(unique_terms)}개만 고유")
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)

if __name__ == "__main__":
    test_term_extraction()

