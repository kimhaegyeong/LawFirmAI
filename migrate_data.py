#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 마이그레이션 스크립트
기존 laws, precedents 테이블의 데이터를 새로운 documents 테이블로 마이그레이션
"""

import sqlite3
import logging
from pathlib import Path
import sys
import os

# source 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))

from data.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_laws_data():
    """laws 테이블 데이터를 documents 테이블로 마이그레이션"""
    logger.info("법령 데이터 마이그레이션 시작...")
    
    try:
        # 기존 데이터베이스 연결
        conn = sqlite3.connect('data/lawfirm.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 새로운 데이터베이스 매니저
        db_manager = DatabaseManager()
        
        # laws 테이블에서 데이터 조회
        cursor.execute("SELECT * FROM laws")
        laws = cursor.fetchall()
        
        migrated_count = 0
        for law in laws:
            # SQLite Row 객체에서 데이터 추출
            law_id, law_name, article_number, content, category, promulgation_date, created_at = law
            
            # 문서 데이터 준비
            doc_data = {
                'id': f"law_{law_id}",
                'document_type': 'law',
                'title': f"{law_name} {article_number}",
                'content': content,
                'source_url': None
            }
            
            # 법령 메타데이터 준비
            law_meta = {
                'law_name': law_name,
                'article_number': article_number,
                'promulgation_date': promulgation_date,
                'enforcement_date': promulgation_date,  # 시행일자가 없으므로 공포일자 사용
                'department': None
            }
            
            # documents 테이블에 추가
            success = db_manager.add_document(doc_data, law_meta=law_meta)
            if success:
                migrated_count += 1
                logger.info(f"마이그레이션 완료: {law['law_name']}")
            else:
                logger.error(f"마이그레이션 실패: {law['law_name']}")
        
        conn.close()
        logger.info(f"법령 데이터 마이그레이션 완료: {migrated_count}/{len(laws)}개")
        return migrated_count
        
    except Exception as e:
        logger.error(f"법령 데이터 마이그레이션 오류: {e}")
        return 0

def migrate_precedents_data():
    """precedents 테이블 데이터를 documents 테이블로 마이그레이션"""
    logger.info("판례 데이터 마이그레이션 시작...")
    
    try:
        # 기존 데이터베이스 연결
        conn = sqlite3.connect('data/lawfirm.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 새로운 데이터베이스 매니저
        db_manager = DatabaseManager()
        
        # precedents 테이블에서 데이터 조회
        cursor.execute("SELECT * FROM precedents")
        precedents = cursor.fetchall()
        
        migrated_count = 0
        for precedent in precedents:
            # SQLite Row 객체에서 데이터 추출
            prec_id, case_number, court_name, decision_date, case_name, content, case_type, created_at = precedent
            
            # 문서 데이터 준비
            doc_data = {
                'id': f"precedent_{prec_id}",
                'document_type': 'precedent',
                'title': case_name,
                'content': content,
                'source_url': None
            }
            
            # 판례 메타데이터 준비
            prec_meta = {
                'case_number': case_number,
                'court_name': court_name,
                'decision_date': decision_date,
                'case_type': case_type
            }
            
            # documents 테이블에 추가
            success = db_manager.add_document(doc_data, prec_meta=prec_meta)
            if success:
                migrated_count += 1
                logger.info(f"마이그레이션 완료: {precedent['case_name']}")
            else:
                logger.error(f"마이그레이션 실패: {precedent['case_name']}")
        
        conn.close()
        logger.info(f"판례 데이터 마이그레이션 완료: {migrated_count}/{len(precedents)}개")
        return migrated_count
        
    except Exception as e:
        logger.error(f"판례 데이터 마이그레이션 오류: {e}")
        return 0

def migrate_constitutional_decisions_data():
    """constitutional_decisions 테이블 데이터를 documents 테이블로 마이그레이션"""
    logger.info("헌재결정례 데이터 마이그레이션 시작...")
    
    try:
        # 기존 데이터베이스 연결
        conn = sqlite3.connect('data/lawfirm.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 새로운 데이터베이스 매니저
        db_manager = DatabaseManager()
        
        # constitutional_decisions 테이블에서 데이터 조회
        cursor.execute("SELECT * FROM constitutional_decisions")
        decisions = cursor.fetchall()
        
        migrated_count = 0
        for decision in decisions:
            # 문서 데이터 준비
            doc_data = {
                'id': f"constitutional_{decision['id']}",
                'document_type': 'constitutional_decision',
                'title': decision.get('case_name', ''),
                'content': decision.get('content', ''),
                'source_url': decision.get('source_url', '')
            }
            
            # 헌재결정례 메타데이터 준비
            const_meta = {
                'case_number': decision.get('case_number', ''),
                'decision_date': decision.get('decision_date', ''),
                'case_type': decision.get('case_type', '')
            }
            
            # documents 테이블에 추가
            success = db_manager.add_document(doc_data, const_meta=const_meta)
            if success:
                migrated_count += 1
                logger.info(f"마이그레이션 완료: {decision['case_name']}")
            else:
                logger.error(f"마이그레이션 실패: {decision['case_name']}")
        
        conn.close()
        logger.info(f"헌재결정례 데이터 마이그레이션 완료: {migrated_count}/{len(decisions)}개")
        return migrated_count
        
    except Exception as e:
        logger.error(f"헌재결정례 데이터 마이그레이션 오류: {e}")
        return 0

def verify_migration():
    """마이그레이션 결과 확인"""
    logger.info("마이그레이션 결과 확인...")
    
    try:
        conn = sqlite3.connect('data/lawfirm.db')
        cursor = conn.cursor()
        
        # documents 테이블 통계
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT document_type, COUNT(*) FROM documents GROUP BY document_type")
        doc_types = cursor.fetchall()
        
        logger.info(f"총 문서 수: {total_docs}개")
        for doc_type, count in doc_types:
            logger.info(f"  - {doc_type}: {count}개")
        
        conn.close()
        return total_docs
        
    except Exception as e:
        logger.error(f"마이그레이션 확인 오류: {e}")
        return 0

if __name__ == "__main__":
    print("🔄 LawFirmAI 데이터베이스 마이그레이션 시작")
    print("=" * 50)
    
    # 각 테이블별 마이그레이션 수행
    laws_count = migrate_laws_data()
    precedents_count = migrate_precedents_data()
    constitutional_count = migrate_constitutional_decisions_data()
    
    # 결과 확인
    total_migrated = verify_migration()
    
    print("\n📊 마이그레이션 결과:")
    print(f"  - 법령: {laws_count}개")
    print(f"  - 판례: {precedents_count}개")
    print(f"  - 헌재결정례: {constitutional_count}개")
    print(f"  - 총 문서: {total_migrated}개")
    
    if total_migrated > 0:
        print("\n✅ 마이그레이션 완료! 이제 임베딩을 다시 생성할 수 있습니다.")
    else:
        print("\n❌ 마이그레이션 실패. 로그를 확인해주세요.")
