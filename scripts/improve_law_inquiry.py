#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법률 문의 분류 정확도 개선 스크립트
키워드 가중치 강화 및 패턴 개선
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.database_keyword_manager import DatabaseKeywordManager


def improve_law_inquiry_keywords():
    """법률 문의 키워드 가중치 강화"""
    print("=" * 60)
    print("법률 문의 키워드 가중치 강화")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 기존 키워드 가중치 업데이트
    print("\n1. 기존 키워드 가중치 업데이트")
    existing_keywords = [
        {"keyword": "제", "weight_value": 5.0},  # 기존 3.0에서 상향
        {"keyword": "조", "weight_value": 5.0},
        {"keyword": "항", "weight_value": 5.0},
        {"keyword": "호", "weight_value": 5.0},
        {"keyword": "민법", "weight_value": 4.5},
        {"keyword": "형법", "weight_value": 4.5},
        {"keyword": "근로기준법", "weight_value": 4.5},
        {"keyword": "상법", "weight_value": 4.5},
        {"keyword": "행정법", "weight_value": 4.5},
        {"keyword": "내용", "weight_value": 3.5},  # 중간에서 고로 상향
        {"keyword": "규정", "weight_value": 3.5},
        {"keyword": "기준", "weight_value": 3.5},
        {"keyword": "처벌", "weight_value": 3.5},
        {"keyword": "최저임금", "weight_value": 3.5},
    ]
    
    for kw_data in existing_keywords:
        # 기존 키워드의 가중치 업데이트
        keywords = db_manager.get_keywords_for_type("law_inquiry")
        for kw in keywords:
            if kw['keyword'] == kw_data['keyword']:
                # 가중치 업데이트를 위해 키워드 삭제 후 재추가
                db_manager.delete_keyword(kw['id'])
                success = db_manager.add_keyword(
                    "law_inquiry",
                    kw_data['keyword'],
                    "high",
                    kw_data['weight_value'],
                    kw.get('category'),
                    kw.get('description')
                )
                if success:
                    print(f"   ✅ {kw_data['keyword']} 가중치 업데이트: {kw_data['weight_value']}")
                break
    
    # 새로운 키워드 추가
    print("\n2. 새로운 키워드 추가")
    new_keywords = [
        {"keyword": "얼마", "weight_level": "high", "weight_value": 3.0, "category": "question", "description": "금액/수량 문의"},
        {"keyword": "몇", "weight_level": "high", "weight_value": 3.0, "category": "question", "description": "수량 문의"},
        {"keyword": "언제", "weight_level": "high", "weight_value": 3.0, "category": "question", "description": "시기 문의"},
        {"keyword": "법령", "weight_level": "high", "weight_value": 4.0, "category": "law", "description": "법령 일반"},
        {"keyword": "법규", "weight_level": "high", "weight_value": 4.0, "category": "law", "description": "법규 일반"},
        {"keyword": "법조문", "weight_level": "high", "weight_value": 4.5, "category": "article", "description": "법조문 일반"},
        {"keyword": "조문", "weight_level": "high", "weight_value": 4.0, "category": "article", "description": "조문 일반"},
        {"keyword": "법정", "weight_level": "medium", "weight_value": 3.0, "category": "legal", "description": "법정 관련"},
        {"keyword": "법적", "weight_level": "medium", "weight_value": 3.0, "category": "legal", "description": "법적 관련"},
        {"keyword": "법률적", "weight_level": "medium", "weight_value": 3.0, "category": "legal", "description": "법률적 관련"},
        {"keyword": "법적근거", "weight_level": "high", "weight_value": 4.0, "category": "legal", "description": "법적 근거"},
        {"keyword": "법적기준", "weight_level": "high", "weight_value": 4.0, "category": "legal", "description": "법적 기준"},
    ]
    
    for kw_data in new_keywords:
        success = db_manager.add_keyword(
            "law_inquiry",
            kw_data["keyword"],
            kw_data["weight_level"],
            kw_data["weight_value"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}, {kw_data['weight_value']}) 추가")


def improve_law_inquiry_patterns():
    """법률 문의 패턴 강화"""
    print("\n" + "=" * 60)
    print("법률 문의 패턴 강화")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 새로운 패턴 추가
    print("\n새로운 패턴 추가")
    new_patterns = [
        # 법조문 번호 패턴 강화
        {"pattern": r'.*제\d+조.*내용|.*제\d+항.*내용|.*제\d+호.*내용', "pattern_type": "regex", "priority": 1, "description": "법조문 내용 문의 패턴"},
        {"pattern": r'.*민법.*제\d+조|.*형법.*제\d+조|.*근로기준법.*제\d+조', "pattern_type": "regex", "priority": 1, "description": "법령별 조문 패턴"},
        
        # 법령별 특화 패턴
        {"pattern": r'.*근로기준법.*정한.*최저임금|.*최저임금.*얼마', "pattern_type": "regex", "priority": 1, "description": "최저임금 문의 패턴"},
        {"pattern": r'.*형법.*제\d+조.*처벌.*기준|.*처벌.*기준.*어떻게', "pattern_type": "regex", "priority": 1, "description": "처벌 기준 문의 패턴"},
        
        # 질문 패턴 강화
        {"pattern": r'.*내용.*알려|.*규정.*알려|.*기준.*알려|.*얼마.*알려', "pattern_type": "regex", "priority": 1, "description": "내용 문의 패턴"},
        {"pattern": r'.*몇.*알려|.*언제.*알려|.*어떻게.*알려', "pattern_type": "regex", "priority": 1, "description": "질문어 문의 패턴"},
        
        # 법령 이름 + 질문어 조합
        {"pattern": r'.*민법.*어떻게|.*형법.*어떻게|.*근로기준법.*어떻게', "pattern_type": "regex", "priority": 1, "description": "법령별 질문 패턴"},
        
        # 추가 패턴
        {"pattern": r'.*법령.*내용|.*법규.*내용|.*법조문.*내용', "pattern_type": "regex", "priority": 1, "description": "법령 내용 문의 패턴"},
        {"pattern": r'.*법적.*근거|.*법적.*기준|.*법률적.*근거', "pattern_type": "regex", "priority": 1, "description": "법적 근거 문의 패턴"},
        {"pattern": r'.*제\d+조.*어떻게|.*제\d+항.*어떻게', "pattern_type": "regex", "priority": 1, "description": "법조문 질문 패턴"},
        {"pattern": r'.*법정.*어떻게|.*법적.*어떻게', "pattern_type": "regex", "priority": 1, "description": "법정/법적 질문 패턴"},
    ]
    
    for pattern_data in new_patterns:
        success = db_manager.add_pattern(
            "law_inquiry",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:50]}...")


def show_law_inquiry_statistics():
    """법률 문의 통계 표시"""
    print("\n" + "=" * 60)
    print("법률 문의 개선 후 통계")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 법률 문의 키워드 통계
    keywords = db_manager.get_keywords_for_type("law_inquiry")
    high_keywords = [kw for kw in keywords if kw['weight_level'] == 'high']
    medium_keywords = [kw for kw in keywords if kw['weight_level'] == 'medium']
    low_keywords = [kw for kw in keywords if kw['weight_level'] == 'low']
    
    print(f"\n📊 법률 문의 키워드 통계:")
    print(f"   전체 키워드: {len(keywords)}개")
    print(f"   고가중치: {len(high_keywords)}개")
    print(f"   중가중치: {len(medium_keywords)}개")
    print(f"   저가중치: {len(low_keywords)}개")
    
    # 고가중치 키워드 목록
    print(f"\n🔑 고가중치 키워드 목록:")
    for kw in sorted(high_keywords, key=lambda x: x['weight_value'], reverse=True):
        print(f"   {kw['keyword']:10s}: {kw['weight_value']:4.1f} ({kw.get('category', 'N/A')})")
    
    # 패턴 통계
    patterns = db_manager.get_patterns_for_type("law_inquiry")
    print(f"\n📋 법률 문의 패턴 통계:")
    print(f"   전체 패턴: {len(patterns)}개")
    
    for pattern in patterns:
        print(f"   {pattern['pattern'][:60]}...")


def main():
    """메인 함수"""
    print("법률 문의 분류 정확도 개선 작업")
    
    try:
        # 1. 키워드 가중치 강화
        improve_law_inquiry_keywords()
        
        # 2. 패턴 강화
        improve_law_inquiry_patterns()
        
        # 3. 통계 표시
        show_law_inquiry_statistics()
        
        print(f"\n" + "=" * 60)
        print("✅ 법률 문의 키워드 및 패턴 개선 완료!")
        print("이제 하이브리드 매핑 로직을 개선하겠습니다.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 개선 작업 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
