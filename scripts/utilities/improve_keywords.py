#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
키워드 데이터베이스 개선 스크립트
테스트 결과를 바탕으로 키워드와 패턴을 개선
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.database_keyword_manager import DatabaseKeywordManager


def improve_keywords():
    """키워드 개선"""
    print("=" * 60)
    print("키워드 데이터베이스 개선")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 1. 판례 검색 키워드 강화
    print("\n1. 판례 검색 키워드 강화")
    precedent_keywords = [
        {"keyword": "찾아주세요", "weight_level": "high", "category": "action", "description": "판례 검색 요청"},
        {"keyword": "찾아", "weight_level": "high", "category": "action", "description": "판례 검색 요청"},
        {"keyword": "검색", "weight_level": "high", "category": "action", "description": "판례 검색 요청"},
        {"keyword": "유사", "weight_level": "high", "category": "comparison", "description": "유사 판례 검색"},
        {"keyword": "관련", "weight_level": "medium", "category": "relation", "description": "관련 판례"},
        {"keyword": "최근", "weight_level": "medium", "category": "time", "description": "최근 판례"},
        {"keyword": "최신", "weight_level": "medium", "category": "time", "description": "최신 판례"},
        {"keyword": "참고", "weight_level": "medium", "category": "action", "description": "참고 판례"},
        {"keyword": "선례", "weight_level": "medium", "category": "core", "description": "선례 검색"},
        {"keyword": "사례", "weight_level": "medium", "category": "core", "description": "사례 검색"}
    ]
    
    for kw_data in precedent_keywords:
        success = db_manager.add_keyword(
            "precedent_search",
            kw_data["keyword"],
            kw_data["weight_level"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}) 추가")
    
    # 2. 법률 문의 키워드 강화
    print("\n2. 법률 문의 키워드 강화")
    law_inquiry_keywords = [
        {"keyword": "제", "weight_level": "high", "category": "article", "description": "법조문 번호"},
        {"keyword": "조", "weight_level": "high", "category": "article", "description": "법조문 번호"},
        {"keyword": "항", "weight_level": "high", "category": "article", "description": "법조문 번호"},
        {"keyword": "호", "weight_level": "high", "category": "article", "description": "법조문 번호"},
        {"keyword": "민법", "weight_level": "high", "category": "law", "description": "민법 관련"},
        {"keyword": "형법", "weight_level": "high", "category": "law", "description": "형법 관련"},
        {"keyword": "근로기준법", "weight_level": "high", "category": "law", "description": "근로기준법 관련"},
        {"keyword": "상법", "weight_level": "high", "category": "law", "description": "상법 관련"},
        {"keyword": "행정법", "weight_level": "high", "category": "law", "description": "행정법 관련"},
        {"keyword": "내용", "weight_level": "medium", "category": "content", "description": "법령 내용 문의"},
        {"keyword": "규정", "weight_level": "medium", "category": "regulation", "description": "법령 규정"},
        {"keyword": "기준", "weight_level": "medium", "category": "standard", "description": "법적 기준"},
        {"keyword": "처벌", "weight_level": "medium", "category": "punishment", "description": "처벌 기준"},
        {"keyword": "최저임금", "weight_level": "medium", "category": "wage", "description": "최저임금 관련"}
    ]
    
    for kw_data in law_inquiry_keywords:
        success = db_manager.add_keyword(
            "law_inquiry",
            kw_data["keyword"],
            kw_data["weight_level"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}) 추가")
    
    # 3. 형사 사건 키워드 강화
    print("\n3. 형사 사건 키워드 강화")
    criminal_keywords = [
        {"keyword": "고소", "weight_level": "high", "category": "action", "description": "고소 관련"},
        {"keyword": "고발", "weight_level": "high", "category": "action", "description": "고발 관련"},
        {"keyword": "사기죄", "weight_level": "high", "category": "crime", "description": "사기죄 관련"},
        {"keyword": "과실치상상죄", "weight_level": "high", "category": "crime", "description": "과실치상상죄 관련"},
        {"keyword": "교통사고", "weight_level": "high", "category": "accident", "description": "교통사고 관련"},
        {"keyword": "적용", "weight_level": "medium", "category": "application", "description": "법령 적용"},
        {"keyword": "피의자", "weight_level": "medium", "category": "person", "description": "피의자 관련"},
        {"keyword": "피고인", "weight_level": "medium", "category": "person", "description": "피고인 관련"},
        {"keyword": "변호인", "weight_level": "medium", "category": "person", "description": "변호인 관련"},
        {"keyword": "선임", "weight_level": "medium", "category": "action", "description": "변호인 선임"}
    ]
    
    for kw_data in criminal_keywords:
        success = db_manager.add_keyword(
            "criminal_case",
            kw_data["keyword"],
            kw_data["weight_level"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}) 추가")
    
    # 4. 법률 조언 키워드 강화
    print("\n4. 법률 조언 키워드 강화")
    legal_advice_keywords = [
        {"keyword": "어떻게", "weight_level": "high", "category": "question", "description": "방법 문의"},
        {"keyword": "대처", "weight_level": "high", "category": "action", "description": "대처 방법"},
        {"keyword": "해결", "weight_level": "high", "category": "solution", "description": "해결 방법"},
        {"keyword": "방법", "weight_level": "high", "category": "method", "description": "해결 방법"},
        {"keyword": "손해", "weight_level": "medium", "category": "damage", "description": "손해 관련"},
        {"keyword": "위반", "weight_level": "medium", "category": "violation", "description": "계약 위반"},
        {"keyword": "분쟁", "weight_level": "medium", "category": "dispute", "description": "분쟁 해결"},
        {"keyword": "소음", "weight_level": "medium", "category": "nuisance", "description": "소음 분쟁"},
        {"keyword": "성희롱", "weight_level": "medium", "category": "harassment", "description": "성희롱 관련"},
        {"keyword": "직장", "weight_level": "medium", "category": "workplace", "description": "직장 관련"}
    ]
    
    for kw_data in legal_advice_keywords:
        success = db_manager.add_keyword(
            "legal_advice",
            kw_data["keyword"],
            kw_data["weight_level"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}) 추가")
    
    # 5. 일반 질문 키워드 강화
    print("\n5. 일반 질문 키워드 강화")
    general_keywords = [
        {"keyword": "어디서", "weight_level": "high", "category": "location", "description": "장소 문의"},
        {"keyword": "얼마나", "weight_level": "high", "category": "amount", "description": "금액 문의"},
        {"keyword": "비용", "weight_level": "high", "category": "cost", "description": "비용 문의"},
        {"keyword": "상담", "weight_level": "medium", "category": "consultation", "description": "상담 관련"},
        {"keyword": "변호사", "weight_level": "medium", "category": "lawyer", "description": "변호사 관련"},
        {"keyword": "소송", "weight_level": "medium", "category": "lawsuit", "description": "소송 관련"},
        {"keyword": "제기", "weight_level": "medium", "category": "action", "description": "소송 제기"}
    ]
    
    for kw_data in general_keywords:
        success = db_manager.add_keyword(
            "general_question",
            kw_data["keyword"],
            kw_data["weight_level"],
            category=kw_data["category"],
            description=kw_data["description"]
        )
        if success:
            print(f"   ✅ {kw_data['keyword']} ({kw_data['weight_level']}) 추가")


def improve_patterns():
    """패턴 개선"""
    print("\n" + "=" * 60)
    print("패턴 개선")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 1. 판례 검색 패턴 강화
    print("\n1. 판례 검색 패턴 강화")
    precedent_patterns = [
        {"pattern": r'.*판례.*찾|.*판례.*검색|.*판례.*찾아', "pattern_type": "regex", "priority": 1, "description": "판례 검색 요청 패턴"},
        {"pattern": r'.*유사.*판례|.*관련.*판례|.*최근.*판례', "pattern_type": "regex", "priority": 1, "description": "특정 판례 검색 패턴"},
        {"pattern": r'.*대법원.*판례|.*하급심.*판례', "pattern_type": "regex", "priority": 1, "description": "법원별 판례 검색 패턴"}
    ]
    
    for pattern_data in precedent_patterns:
        success = db_manager.add_pattern(
            "precedent_search",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:30]}...")
    
    # 2. 법률 문의 패턴 강화
    print("\n2. 법률 문의 패턴 강화")
    law_inquiry_patterns = [
        {"pattern": r'.*제\d+조|.*제\d+항|.*제\d+호', "pattern_type": "regex", "priority": 1, "description": "법조문 번호 패턴"},
        {"pattern": r'.*민법.*제|.*형법.*제|.*근로기준법.*제', "pattern_type": "regex", "priority": 1, "description": "법령별 조문 패턴"},
        {"pattern": r'.*내용.*알려|.*규정.*알려|.*기준.*알려', "pattern_type": "regex", "priority": 1, "description": "법령 내용 문의 패턴"}
    ]
    
    for pattern_data in law_inquiry_patterns:
        success = db_manager.add_pattern(
            "law_inquiry",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:30]}...")
    
    # 3. 형사 사건 패턴 강화
    print("\n3. 형사 사건 패턴 강화")
    criminal_patterns = [
        {"pattern": r'.*고소.*당|.*고발.*당|.*피의자.*되', "pattern_type": "regex", "priority": 1, "description": "피의자 관련 패턴"},
        {"pattern": r'.*사기죄|.*과실치상상죄|.*교통사고.*죄', "pattern_type": "regex", "priority": 1, "description": "범죄 유형 패턴"},
        {"pattern": r'.*변호인.*선임|.*변호인.*필수', "pattern_type": "regex", "priority": 1, "description": "변호인 관련 패턴"}
    ]
    
    for pattern_data in criminal_patterns:
        success = db_manager.add_pattern(
            "criminal_case",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:30]}...")
    
    # 4. 법률 조언 패턴 강화
    print("\n4. 법률 조언 패턴 강화")
    legal_advice_patterns = [
        {"pattern": r'.*어떻게.*해야|.*어떻게.*대처|.*어떻게.*해결', "pattern_type": "regex", "priority": 1, "description": "해결 방법 문의 패턴"},
        {"pattern": r'.*손해.*입.*어떻게|.*위반.*어떻게', "pattern_type": "regex", "priority": 1, "description": "손해/위반 관련 패턴"},
        {"pattern": r'.*분쟁.*해결|.*조언.*해주|.*방법.*조언', "pattern_type": "regex", "priority": 1, "description": "조언 요청 패턴"}
    ]
    
    for pattern_data in legal_advice_patterns:
        success = db_manager.add_pattern(
            "legal_advice",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:30]}...")
    
    # 5. 일반 질문 패턴 강화
    print("\n5. 일반 질문 패턴 강화")
    general_patterns = [
        {"pattern": r'.*어디서.*받|.*어디서.*받을', "pattern_type": "regex", "priority": 1, "description": "장소 문의 패턴"},
        {"pattern": r'.*얼마나.*드|.*비용.*얼마', "pattern_type": "regex", "priority": 1, "description": "비용 문의 패턴"},
        {"pattern": r'.*소송.*제기.*어떻게', "pattern_type": "regex", "priority": 1, "description": "소송 제기 문의 패턴"}
    ]
    
    for pattern_data in general_patterns:
        success = db_manager.add_pattern(
            "general_question",
            pattern_data["pattern"],
            pattern_data["pattern_type"],
            pattern_data["priority"],
            pattern_data["description"]
        )
        if success:
            print(f"   ✅ 패턴 추가: {pattern_data['pattern'][:30]}...")


def show_statistics():
    """개선 후 통계 표시"""
    print("\n" + "=" * 60)
    print("개선 후 통계")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 전체 통계
    stats = db_manager.get_keyword_statistics()
    print(f"\n📊 전체 키워드 통계:")
    print(f"   전체 키워드: {stats.get('total_keywords', 0)}개")
    print(f"   고가중치: {stats.get('high_weight_count', 0)}개")
    print(f"   중가중치: {stats.get('medium_weight_count', 0)}개")
    print(f"   저가중치: {stats.get('low_weight_count', 0)}개")
    
    # 질문 유형별 통계
    question_types = db_manager.get_all_question_types()
    print(f"\n📋 질문 유형별 키워드 수:")
    
    for qt in question_types:
        keywords = db_manager.get_keywords_for_type(qt['type_name'])
        print(f"   {qt['type_name']:20s}: {len(keywords):3d}개")


def main():
    """메인 함수"""
    print("키워드 데이터베이스 개선 작업")
    
    try:
        # 1. 키워드 개선
        improve_keywords()
        
        # 2. 패턴 개선
        improve_patterns()
        
        # 3. 통계 표시
        show_statistics()
        
        print(f"\n" + "=" * 60)
        print("✅ 키워드 데이터베이스 개선 완료!")
        print("이제 다시 테스트를 실행해보세요.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 개선 작업 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
