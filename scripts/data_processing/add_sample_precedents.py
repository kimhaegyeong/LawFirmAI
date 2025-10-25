#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 테이블에 샘플 데이터 추가
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.database import DatabaseManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_sample_precedents():
    """판례 테이블에 샘플 데이터 추가"""
    
    # 데이터베이스 매니저 초기화
    db_manager = DatabaseManager("data/lawfirm.db")
    
    # 샘플 판례 데이터
    sample_precedents = [
        {
            "case_id": "CIVIL_750_001",
            "category": "민사",
            "case_name": "교통사고 과실비율 인정 사건",
            "case_number": "대법원 2018다12345",
            "decision_date": "2018-03-15",
            "field": "불법행위",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2018다12345",
            "full_text": "교통사고에서 과실비율을 7:3으로 인정한 사건. 원고의 과실을 30%로 인정하여 손해배상액을 감액하였다.",
            "searchable_text": "교통사고 과실비율 7:3 인정 손해배상 감액 원고 과실 30%"
        },
        {
            "case_id": "CIVIL_750_002", 
            "category": "민사",
            "case_name": "의료사고 입증책임 사건",
            "case_number": "대법원 2019다67890",
            "decision_date": "2019-07-20",
            "field": "불법행위",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2019다67890",
            "full_text": "의료사고에서 의사의 과실 입증책임은 원고(환자)가 부담한다는 원칙을 확인한 사건.",
            "searchable_text": "의료사고 입증책임 원고 부담 의사 과실 환자"
        },
        {
            "case_id": "CIVIL_750_003",
            "category": "민사", 
            "case_name": "정신적 피해 위자료 인정 사건",
            "case_number": "대법원 2020다11111",
            "decision_date": "2020-11-10",
            "field": "불법행위",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2020다11111",
            "full_text": "불법행위로 인한 정신적 피해에 대해 위자료 500만원을 인정한 사건.",
            "searchable_text": "정신적 피해 위자료 500만원 불법행위 정신적 손해"
        },
        {
            "case_id": "CIVIL_751_001",
            "category": "민사",
            "case_name": "정신적 피해 배상 범위 사건", 
            "case_number": "대법원 2021다22222",
            "decision_date": "2021-05-15",
            "field": "불법행위",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2021다22222",
            "full_text": "민법 제751조에 따른 정신적 피해의 배상 범위와 산정 기준을 명시한 사건.",
            "searchable_text": "정신적 피해 배상 범위 산정 기준 민법 751조"
        },
        {
            "case_id": "CIVIL_752_001",
            "category": "민사",
            "case_name": "생명침해로 인한 재산이외 손해 사건",
            "case_number": "대법원 2022다33333", 
            "decision_date": "2022-09-20",
            "field": "불법행위",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2022다33333",
            "full_text": "민법 제752조에 따른 생명침해로 인한 재산이외의 손해 배상에 관한 사건.",
            "searchable_text": "생명침해 재산이외 손해 배상 민법 752조"
        },
        {
            "case_id": "CIVIL_753_001",
            "category": "민사",
            "case_name": "미성년자 책임능력 사건",
            "case_number": "대법원 2023다44444",
            "decision_date": "2023-02-28",
            "field": "불법행위", 
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2023다44444",
            "full_text": "민법 제753조에 따른 미성년자의 책임능력과 손해배상책임에 관한 사건.",
            "searchable_text": "미성년자 책임능력 손해배상책임 민법 753조"
        },
        {
            "case_id": "CONTRACT_001",
            "category": "민사",
            "case_name": "계약 해제 사건",
            "case_number": "대법원 2023다55555",
            "decision_date": "2023-06-15",
            "field": "계약",
            "court": "대법원", 
            "detail_url": "https://example.com/precedent/2023다55555",
            "full_text": "계약의 해제 요건과 효과에 관한 판례. 계약 위반 시 해제권 행사 조건을 명시.",
            "searchable_text": "계약 해제 요건 효과 계약 위반 해제권"
        },
        {
            "case_id": "PROPERTY_001",
            "category": "민사",
            "case_name": "소유권 확인 사건",
            "case_number": "대법원 2023다66666",
            "decision_date": "2023-08-10",
            "field": "물권",
            "court": "대법원",
            "detail_url": "https://example.com/precedent/2023다66666", 
            "full_text": "부동산 소유권 확인에 관한 판례. 등기부상의 추정력과 실제 권리관계의 불일치 문제.",
            "searchable_text": "소유권 확인 부동산 등기부 추정력 권리관계"
        }
    ]
    
    try:
        # 판례 데이터 삽입
        for precedent in sample_precedents:
            query = """
                INSERT OR REPLACE INTO precedent_cases 
                (case_id, category, case_name, case_number, decision_date, field, court, detail_url, full_text, searchable_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                precedent["case_id"],
                precedent["category"], 
                precedent["case_name"],
                precedent["case_number"],
                precedent["decision_date"],
                precedent["field"],
                precedent["court"],
                precedent["detail_url"],
                precedent["full_text"],
                precedent["searchable_text"]
            )
            
            db_manager.execute_update(query, params)
            logger.info(f"판례 데이터 삽입 완료: {precedent['case_name']}")
        
        # 판례 섹션 데이터도 추가
        sample_sections = [
            {
                "section_id": "CIVIL_750_001_001",
                "case_id": "CIVIL_750_001",
                "section_type": "판시사항",
                "section_type_korean": "판시사항",
                "section_content": "교통사고에서 과실비율을 7:3으로 인정한 사건",
                "section_length": 25,
                "has_content": True
            },
            {
                "section_id": "CIVIL_750_001_002", 
                "case_id": "CIVIL_750_001",
                "section_type": "판결요지",
                "section_type_korean": "판결요지",
                "section_content": "원고의 과실을 30%로 인정하여 손해배상액을 감액하였다",
                "section_length": 30,
                "has_content": True
            },
            {
                "section_id": "CIVIL_750_002_001",
                "case_id": "CIVIL_750_002",
                "section_type": "판시사항", 
                "section_type_korean": "판시사항",
                "section_content": "의료사고에서 의사의 과실 입증책임은 원고가 부담한다",
                "section_length": 28,
                "has_content": True
            },
            {
                "section_id": "CIVIL_750_003_001",
                "case_id": "CIVIL_750_003",
                "section_type": "판시사항",
                "section_type_korean": "판시사항", 
                "section_content": "불법행위로 인한 정신적 피해에 대해 위자료 500만원을 인정",
                "section_length": 32,
                "has_content": True
            }
        ]
        
        for section in sample_sections:
            query = """
                INSERT OR REPLACE INTO precedent_sections
                (section_id, case_id, section_type, section_type_korean, section_content, section_length, has_content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                section["section_id"],
                section["case_id"],
                section["section_type"],
                section["section_type_korean"],
                section["section_content"],
                section["section_length"],
                section["has_content"]
            )
            
            db_manager.execute_update(query, params)
            logger.info(f"판례 섹션 데이터 삽입 완료: {section['section_id']}")
        
        logger.info(f"총 {len(sample_precedents)}개의 판례 데이터와 {len(sample_sections)}개의 섹션 데이터가 추가되었습니다.")
        
        # 데이터 확인
        count_query = "SELECT COUNT(*) as count FROM precedent_cases"
        result = db_manager.execute_query(count_query)
        logger.info(f"현재 판례 테이블의 총 레코드 수: {result[0]['count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"판례 데이터 추가 실패: {e}")
        return False

if __name__ == "__main__":
    success = add_sample_precedents()
    if success:
        print("✅ 판례 샘플 데이터 추가 완료")
    else:
        print("❌ 판례 샘플 데이터 추가 실패")
        sys.exit(1)
