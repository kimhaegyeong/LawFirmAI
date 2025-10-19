#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 키워드 데이터 삽입 스크립트
법률 질문 유형별 키워드를 데이터베이스에 삽입
"""

import sys
import os
from typing import List, Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.database_keyword_manager import DatabaseKeywordManager


def get_default_keywords_data() -> Dict[str, List[Dict[str, Any]]]:
    """기본 키워드 데이터 반환"""
    return {
        "precedent_search": [
            {"keyword": "판례", "weight_level": "high", "category": "core", "description": "판례 검색 핵심 키워드"},
            {"keyword": "precedent", "weight_level": "high", "category": "core", "description": "영어 판례 키워드"},
            {"keyword": "사건", "weight_level": "high", "category": "core", "description": "사건 관련 키워드"},
            {"keyword": "판결", "weight_level": "high", "category": "core", "description": "판결 관련 키워드"},
            {"keyword": "대법원", "weight_level": "high", "category": "institution", "description": "대법원 판례"},
            {"keyword": "하급심", "weight_level": "medium", "category": "institution", "description": "하급심 판례"},
            {"keyword": "선례", "weight_level": "medium", "category": "core", "description": "선례 관련"},
            {"keyword": "참고", "weight_level": "medium", "category": "action", "description": "참고 관련"},
            {"keyword": "유사", "weight_level": "medium", "category": "comparison", "description": "유사 사건"},
            {"keyword": "법원", "weight_level": "low", "category": "institution", "description": "법원 일반"},
            {"keyword": "재판", "weight_level": "low", "category": "process", "description": "재판 과정"},
            {"keyword": "심판", "weight_level": "low", "category": "process", "description": "심판 과정"},
            {"keyword": "법리", "weight_level": "low", "category": "concept", "description": "법리적 분석"}
        ],
        "contract_review": [
            {"keyword": "계약서", "weight_level": "high", "category": "document", "description": "계약서 검토 핵심"},
            {"keyword": "contract", "weight_level": "high", "category": "document", "description": "영어 계약 키워드"},
            {"keyword": "계약", "weight_level": "high", "category": "core", "description": "계약 일반"},
            {"keyword": "체결", "weight_level": "high", "category": "action", "description": "계약 체결"},
            {"keyword": "합의서", "weight_level": "medium", "category": "document", "description": "합의서 관련"},
            {"keyword": "약정서", "weight_level": "medium", "category": "document", "description": "약정서 관련"},
            {"keyword": "조항", "weight_level": "medium", "category": "content", "description": "계약 조항"},
            {"keyword": "조건", "weight_level": "medium", "category": "content", "description": "계약 조건"},
            {"keyword": "위약금", "weight_level": "medium", "category": "penalty", "description": "위약금 관련"},
            {"keyword": "해지", "weight_level": "medium", "category": "action", "description": "계약 해지"},
            {"keyword": "해제", "weight_level": "medium", "category": "action", "description": "계약 해제"},
            {"keyword": "무효", "weight_level": "medium", "category": "status", "description": "계약 무효"},
            {"keyword": "취소", "weight_level": "medium", "category": "action", "description": "계약 취소"},
            {"keyword": "문서", "weight_level": "low", "category": "document", "description": "문서 일반"},
            {"keyword": "서류", "weight_level": "low", "category": "document", "description": "서류 일반"},
            {"keyword": "계약금", "weight_level": "low", "category": "payment", "description": "계약금"},
            {"keyword": "보증금", "weight_level": "low", "category": "payment", "description": "보증금"}
        ],
        "divorce_procedure": [
            {"keyword": "이혼", "weight_level": "high", "category": "core", "description": "이혼 절차 핵심"},
            {"keyword": "divorce", "weight_level": "high", "category": "core", "description": "영어 이혼 키워드"},
            {"keyword": "협의이혼", "weight_level": "high", "category": "type", "description": "협의이혼"},
            {"keyword": "재판이혼", "weight_level": "high", "category": "type", "description": "재판이혼"},
            {"keyword": "조정이혼", "weight_level": "high", "category": "type", "description": "조정이혼"},
            {"keyword": "재산분할", "weight_level": "medium", "category": "property", "description": "재산분할"},
            {"keyword": "양육권", "weight_level": "medium", "category": "custody", "description": "양육권"},
            {"keyword": "위자료", "weight_level": "medium", "category": "compensation", "description": "위자료"},
            {"keyword": "면접교섭권", "weight_level": "medium", "category": "custody", "description": "면접교섭권"},
            {"keyword": "양육비", "weight_level": "medium", "category": "support", "description": "양육비"},
            {"keyword": "가족", "weight_level": "low", "category": "relation", "description": "가족 관계"},
            {"keyword": "부부", "weight_level": "low", "category": "relation", "description": "부부 관계"},
            {"keyword": "혼인", "weight_level": "low", "category": "marriage", "description": "혼인 관계"},
            {"keyword": "결혼", "weight_level": "low", "category": "marriage", "description": "결혼 관계"}
        ],
        "inheritance_procedure": [
            {"keyword": "상속", "weight_level": "high", "category": "core", "description": "상속 절차 핵심"},
            {"keyword": "inheritance", "weight_level": "high", "category": "core", "description": "영어 상속 키워드"},
            {"keyword": "유산", "weight_level": "high", "category": "property", "description": "유산 관련"},
            {"keyword": "상속인", "weight_level": "high", "category": "person", "description": "상속인"},
            {"keyword": "상속세", "weight_level": "medium", "category": "tax", "description": "상속세"},
            {"keyword": "유언", "weight_level": "medium", "category": "will", "description": "유언 관련"},
            {"keyword": "상속분", "weight_level": "medium", "category": "share", "description": "상속분"},
            {"keyword": "상속재산", "weight_level": "medium", "category": "property", "description": "상속재산"},
            {"keyword": "가족", "weight_level": "low", "category": "relation", "description": "가족 관계"},
            {"keyword": "재산", "weight_level": "low", "category": "property", "description": "재산 일반"}
        ],
        "criminal_case": [
            {"keyword": "범죄", "weight_level": "high", "category": "core", "description": "범죄 관련 핵심"},
            {"keyword": "criminal", "weight_level": "high", "category": "core", "description": "영어 형사 키워드"},
            {"keyword": "형사", "weight_level": "high", "category": "core", "description": "형사 사건"},
            {"keyword": "피의자", "weight_level": "high", "category": "person", "description": "피의자"},
            {"keyword": "피고인", "weight_level": "high", "category": "person", "description": "피고인"},
            {"keyword": "수사", "weight_level": "medium", "category": "process", "description": "수사 과정"},
            {"keyword": "재판", "weight_level": "medium", "category": "process", "description": "재판 과정"},
            {"keyword": "형량", "weight_level": "medium", "category": "punishment", "description": "형량"},
            {"keyword": "구속", "weight_level": "medium", "category": "detention", "description": "구속"},
            {"keyword": "법원", "weight_level": "low", "category": "institution", "description": "법원"},
            {"keyword": "검찰", "weight_level": "low", "category": "institution", "description": "검찰"}
        ],
        "labor_dispute": [
            {"keyword": "노동", "weight_level": "high", "category": "core", "description": "노동 분쟁 핵심"},
            {"keyword": "labor", "weight_level": "high", "category": "core", "description": "영어 노동 키워드"},
            {"keyword": "근로", "weight_level": "high", "category": "core", "description": "근로 관련"},
            {"keyword": "임금", "weight_level": "high", "category": "payment", "description": "임금 관련"},
            {"keyword": "해고", "weight_level": "high", "category": "action", "description": "해고 관련"},
            {"keyword": "근로계약", "weight_level": "medium", "category": "contract", "description": "근로계약"},
            {"keyword": "근로시간", "weight_level": "medium", "category": "condition", "description": "근로시간"},
            {"keyword": "부당해고", "weight_level": "medium", "category": "dispute", "description": "부당해고"},
            {"keyword": "노동위원회", "weight_level": "medium", "category": "institution", "description": "노동위원회"},
            {"keyword": "직장", "weight_level": "low", "category": "place", "description": "직장"},
            {"keyword": "회사", "weight_level": "low", "category": "organization", "description": "회사"}
        ],
        "procedure_guide": [
            {"keyword": "절차", "weight_level": "high", "category": "core", "description": "절차 안내 핵심"},
            {"keyword": "procedure", "weight_level": "high", "category": "core", "description": "영어 절차 키워드"},
            {"keyword": "신청", "weight_level": "high", "category": "action", "description": "신청 절차"},
            {"keyword": "제출", "weight_level": "high", "category": "action", "description": "제출 절차"},
            {"keyword": "서류", "weight_level": "medium", "category": "document", "description": "필요 서류"},
            {"keyword": "기간", "weight_level": "medium", "category": "time", "description": "처리 기간"},
            {"keyword": "비용", "weight_level": "medium", "category": "cost", "description": "처리 비용"},
            {"keyword": "방법", "weight_level": "medium", "category": "method", "description": "처리 방법"},
            {"keyword": "안내", "weight_level": "low", "category": "guide", "description": "안내 일반"},
            {"keyword": "가이드", "weight_level": "low", "category": "guide", "description": "가이드 일반"}
        ],
        "term_explanation": [
            {"keyword": "용어", "weight_level": "high", "category": "core", "description": "용어 해설 핵심"},
            {"keyword": "term", "weight_level": "high", "category": "core", "description": "영어 용어 키워드"},
            {"keyword": "정의", "weight_level": "high", "category": "definition", "description": "용어 정의"},
            {"keyword": "의미", "weight_level": "high", "category": "meaning", "description": "용어 의미"},
            {"keyword": "해설", "weight_level": "medium", "category": "explanation", "description": "용어 해설"},
            {"keyword": "설명", "weight_level": "medium", "category": "explanation", "description": "용어 설명"},
            {"keyword": "풀이", "weight_level": "medium", "category": "explanation", "description": "용어 풀이"},
            {"keyword": "법률", "weight_level": "low", "category": "domain", "description": "법률 일반"},
            {"keyword": "개념", "weight_level": "low", "category": "concept", "description": "법률 개념"}
        ],
        "legal_advice": [
            {"keyword": "조언", "weight_level": "high", "category": "core", "description": "법률 조언 핵심"},
            {"keyword": "advice", "weight_level": "high", "category": "core", "description": "영어 조언 키워드"},
            {"keyword": "상담", "weight_level": "high", "category": "consultation", "description": "법률 상담"},
            {"keyword": "도움", "weight_level": "high", "category": "help", "description": "법률 도움"},
            {"keyword": "권리", "weight_level": "medium", "category": "right", "description": "권리 구제"},
            {"keyword": "구제", "weight_level": "medium", "category": "remedy", "description": "권리 구제"},
            {"keyword": "방안", "weight_level": "medium", "category": "solution", "description": "해결 방안"},
            {"keyword": "해결", "weight_level": "medium", "category": "solution", "description": "문제 해결"},
            {"keyword": "법률", "weight_level": "low", "category": "domain", "description": "법률 일반"},
            {"keyword": "문제", "weight_level": "low", "category": "issue", "description": "법률 문제"}
        ],
        "law_inquiry": [
            {"keyword": "법률", "weight_level": "high", "category": "core", "description": "법률 문의 핵심"},
            {"keyword": "law", "weight_level": "high", "category": "core", "description": "영어 법률 키워드"},
            {"keyword": "법령", "weight_level": "high", "category": "regulation", "description": "법령 관련"},
            {"keyword": "조문", "weight_level": "high", "category": "article", "description": "법조문"},
            {"keyword": "규정", "weight_level": "medium", "category": "regulation", "description": "법규정"},
            {"keyword": "법적", "weight_level": "medium", "category": "legal", "description": "법적 관련"},
            {"keyword": "적용", "weight_level": "medium", "category": "application", "description": "법령 적용"},
            {"keyword": "문의", "weight_level": "low", "category": "inquiry", "description": "문의 일반"},
            {"keyword": "질문", "weight_level": "low", "category": "question", "description": "질문 일반"}
        ],
        "general_question": [
            {"keyword": "법률", "weight_level": "medium", "category": "domain", "description": "법률 일반"},
            {"keyword": "정보", "weight_level": "medium", "category": "information", "description": "정보 요청"},
            {"keyword": "질문", "weight_level": "medium", "category": "question", "description": "일반 질문"},
            {"keyword": "답변", "weight_level": "medium", "category": "answer", "description": "답변 요청"},
            {"keyword": "권리", "weight_level": "low", "category": "right", "description": "권리 관련"},
            {"keyword": "의무", "weight_level": "low", "category": "duty", "description": "의무 관련"},
            {"keyword": "상담", "weight_level": "low", "category": "consultation", "description": "상담 요청"},
            {"keyword": "안내", "weight_level": "low", "category": "guide", "description": "안내 요청"}
        ]
    }


def get_default_patterns_data() -> Dict[str, List[Dict[str, Any]]]:
    """기본 패턴 데이터 반환"""
    return {
        "precedent_search": [
            {"pattern": r'.*판례.*찾|.*사건.*찾|.*유사.*판례|.*선례.*찾', "pattern_type": "regex", "priority": 1, "description": "판례 검색 패턴"},
            {"pattern": r'.*대법원.*판결|.*하급심.*판결|.*법원.*판결', "pattern_type": "regex", "priority": 1, "description": "판결 관련 패턴"},
            {"pattern": r'.*precedent.*search|.*case.*law|.*similar.*case', "pattern_type": "regex", "priority": 2, "description": "영어 판례 검색 패턴"}
        ],
        "contract_review": [
            {"pattern": r'.*계약서.*검토|.*계약서.*검토|.*계약.*검토', "pattern_type": "regex", "priority": 1, "description": "계약서 검토 패턴"},
            {"pattern": r'.*contract.*review|.*agreement.*review|.*contract.*analysis', "pattern_type": "regex", "priority": 2, "description": "영어 계약 검토 패턴"},
            {"pattern": r'.*계약.*조항.*분석|.*계약.*조건.*확인|.*계약.*문제', "pattern_type": "regex", "priority": 1, "description": "계약 분석 패턴"}
        ],
        "divorce_procedure": [
            {"pattern": r'.*이혼.*절차|.*이혼.*방법|.*이혼.*신청', "pattern_type": "regex", "priority": 1, "description": "이혼 절차 패턴"},
            {"pattern": r'.*divorce.*procedure|.*divorce.*process', "pattern_type": "regex", "priority": 2, "description": "영어 이혼 절차 패턴"},
            {"pattern": r'.*협의이혼|.*재판이혼|.*조정이혼', "pattern_type": "regex", "priority": 1, "description": "이혼 유형 패턴"}
        ],
        "inheritance_procedure": [
            {"pattern": r'.*상속.*절차|.*상속.*신청|.*유산.*처리', "pattern_type": "regex", "priority": 1, "description": "상속 절차 패턴"},
            {"pattern": r'.*inheritance.*procedure|.*inheritance.*process', "pattern_type": "regex", "priority": 2, "description": "영어 상속 절차 패턴"},
            {"pattern": r'.*상속인.*확인|.*상속분.*계산', "pattern_type": "regex", "priority": 1, "description": "상속 관련 패턴"}
        ],
        "criminal_case": [
            {"pattern": r'.*범죄.*처벌|.*형사.*처벌|.*피의자.*권리', "pattern_type": "regex", "priority": 1, "description": "형사 사건 패턴"},
            {"pattern": r'.*criminal.*case|.*criminal.*charge', "pattern_type": "regex", "priority": 2, "description": "영어 형사 사건 패턴"},
            {"pattern": r'.*수사.*절차|.*재판.*절차', "pattern_type": "regex", "priority": 1, "description": "형사 절차 패턴"}
        ],
        "labor_dispute": [
            {"pattern": r'.*노동.*분쟁|.*근로.*분쟁|.*임금.*분쟁', "pattern_type": "regex", "priority": 1, "description": "노동 분쟁 패턴"},
            {"pattern": r'.*labor.*dispute|.*employment.*dispute', "pattern_type": "regex", "priority": 2, "description": "영어 노동 분쟁 패턴"},
            {"pattern": r'.*해고.*분쟁|.*부당해고', "pattern_type": "regex", "priority": 1, "description": "해고 분쟁 패턴"}
        ],
        "procedure_guide": [
            {"pattern": r'.*절차.*안내|.*신청.*방법|.*제출.*방법', "pattern_type": "regex", "priority": 1, "description": "절차 안내 패턴"},
            {"pattern": r'.*procedure.*guide|.*application.*process', "pattern_type": "regex", "priority": 2, "description": "영어 절차 안내 패턴"},
            {"pattern": r'.*어떻게.*신청|.*어떻게.*제출', "pattern_type": "regex", "priority": 1, "description": "방법 문의 패턴"}
        ],
        "term_explanation": [
            {"pattern": r'.*용어.*의미|.*용어.*정의|.*용어.*해설', "pattern_type": "regex", "priority": 1, "description": "용어 해설 패턴"},
            {"pattern": r'.*term.*definition|.*term.*explanation', "pattern_type": "regex", "priority": 2, "description": "영어 용어 해설 패턴"},
            {"pattern": r'.*무엇.*의미|.*무엇.*뜻', "pattern_type": "regex", "priority": 1, "description": "의미 문의 패턴"}
        ],
        "legal_advice": [
            {"pattern": r'.*법률.*조언|.*법률.*상담|.*도움.*요청', "pattern_type": "regex", "priority": 1, "description": "법률 조언 패턴"},
            {"pattern": r'.*legal.*advice|.*legal.*consultation', "pattern_type": "regex", "priority": 2, "description": "영어 법률 조언 패턴"},
            {"pattern": r'.*어떻게.*해결|.*어떻게.*처리', "pattern_type": "regex", "priority": 1, "description": "해결 방법 패턴"}
        ],
        "law_inquiry": [
            {"pattern": r'.*법률.*문의|.*법령.*문의|.*법적.*질문', "pattern_type": "regex", "priority": 1, "description": "법률 문의 패턴"},
            {"pattern": r'.*law.*inquiry|.*legal.*question', "pattern_type": "regex", "priority": 2, "description": "영어 법률 문의 패턴"},
            {"pattern": r'.*법.*규정|.*법.*조문', "pattern_type": "regex", "priority": 1, "description": "법령 문의 패턴"}
        ]
    }


def insert_default_data():
    """기본 데이터 삽입"""
    print("=" * 60)
    print("데이터베이스 기반 키워드 관리 시스템 초기화")
    print("=" * 60)
    
    # 데이터베이스 매니저 초기화
    db_manager = DatabaseKeywordManager()
    
    # 질문 유형 등록
    question_types = [
        ("precedent_search", "판례 검색", "관련 판례를 찾는 질문"),
        ("contract_review", "계약서 검토", "계약서 검토 및 분석 요청"),
        ("divorce_procedure", "이혼 절차", "이혼 관련 절차 및 방법 문의"),
        ("inheritance_procedure", "상속 절차", "상속 관련 절차 및 방법 문의"),
        ("criminal_case", "형사 사건", "형사 사건 관련 문의"),
        ("labor_dispute", "노동 분쟁", "노동 분쟁 관련 문의"),
        ("procedure_guide", "절차 안내", "각종 법적 절차 안내 요청"),
        ("term_explanation", "용어 해설", "법률 용어의 의미 및 정의 문의"),
        ("legal_advice", "법률 조언", "법률 상담 및 조언 요청"),
        ("law_inquiry", "법률 문의", "법률 및 법령 관련 문의"),
        ("general_question", "일반 질문", "일반적인 법률 질문")
    ]
    
    print("\n1. 질문 유형 등록 중...")
    for type_name, display_name, description in question_types:
        success = db_manager.register_question_type(type_name, display_name, description)
        if success:
            print(f"   ✅ {type_name} ({display_name}) 등록 완료")
        else:
            print(f"   ❌ {type_name} 등록 실패")
    
    # 키워드 데이터 삽입
    print("\n2. 키워드 데이터 삽입 중...")
    keywords_data = get_default_keywords_data()
    total_keywords = 0
    
    for question_type, keywords in keywords_data.items():
        batch_data = []
        for keyword_info in keywords:
            batch_data.append({
                'question_type': question_type,
                'keyword': keyword_info['keyword'],
                'weight_level': keyword_info['weight_level'],
                'category': keyword_info.get('category'),
                'description': keyword_info.get('description')
            })
        
        imported_count = db_manager.add_keywords_batch(batch_data)
        total_keywords += imported_count
        print(f"   ✅ {question_type}: {imported_count}개 키워드 추가")
    
    # 패턴 데이터 삽입
    print("\n3. 패턴 데이터 삽입 중...")
    patterns_data = get_default_patterns_data()
    total_patterns = 0
    
    for question_type, patterns in patterns_data.items():
        for pattern_info in patterns:
            success = db_manager.add_pattern(
                question_type=question_type,
                pattern=pattern_info['pattern'],
                pattern_type=pattern_info['pattern_type'],
                priority=pattern_info['priority'],
                description=pattern_info['description']
            )
            if success:
                total_patterns += 1
        
        print(f"   ✅ {question_type}: {len(patterns)}개 패턴 추가")
    
    # 통계 출력
    print(f"\n" + "=" * 60)
    print("초기화 완료!")
    print("=" * 60)
    print(f"등록된 질문 유형: {len(question_types)}개")
    print(f"추가된 키워드: {total_keywords}개")
    print(f"추가된 패턴: {total_patterns}개")
    
    # 데이터베이스 통계
    stats = db_manager.get_keyword_statistics()
    print(f"\n데이터베이스 통계:")
    print(f"  전체 키워드 수: {stats.get('total_keywords', 0)}")
    print(f"  고가중치 키워드: {stats.get('high_weight_count', 0)}")
    print(f"  중가중치 키워드: {stats.get('medium_weight_count', 0)}")
    print(f"  저가중치 키워드: {stats.get('low_weight_count', 0)}")
    
    print(f"\n데이터베이스 파일 위치: {db_manager.db_path}")
    print("=" * 60)


def test_database_functionality():
    """데이터베이스 기능 테스트"""
    print("\n" + "=" * 60)
    print("데이터베이스 기능 테스트")
    print("=" * 60)
    
    db_manager = DatabaseKeywordManager()
    
    # 테스트 케이스들
    test_cases = [
        ("precedent_search", "판례"),
        ("contract_review", "계약서"),
        ("divorce_procedure", "이혼"),
        ("inheritance_procedure", "상속"),
        ("criminal_case", "범죄")
    ]
    
    for question_type, search_term in test_cases:
        print(f"\n테스트: {question_type} - '{search_term}' 검색")
        
        # 키워드 검색
        keywords = db_manager.search_keywords(search_term, question_type)
        print(f"  키워드 검색 결과: {len(keywords)}개")
        for kw in keywords[:3]:  # 상위 3개만 표시
            print(f"    - {kw['keyword']} ({kw['weight_level']}, {kw['weight_value']})")
        
        # 패턴 검색
        patterns = db_manager.get_patterns_for_type(question_type)
        print(f"  패턴 검색 결과: {len(patterns)}개")
        for pattern in patterns[:2]:  # 상위 2개만 표시
            print(f"    - {pattern['pattern'][:50]}...")
    
    print(f"\n✅ 모든 테스트 완료!")


if __name__ == "__main__":
    try:
        # 기본 데이터 삽입
        insert_default_data()
        
        # 기능 테스트
        test_database_functionality()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
