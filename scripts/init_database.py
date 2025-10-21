#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 초기화 스크립트
"""

import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.data.database import DatabaseManager

def initialize_database():
    """데이터베이스 초기화"""
    print("데이터베이스 초기화 시작...")
    
    try:
        # 데이터베이스 매니저 생성
        db_manager = DatabaseManager()
        print("데이터베이스 매니저 생성 완료")
        
        # 데이터베이스 연결 테스트
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # 테이블 목록 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"생성된 테이블 수: {len(tables)}")
            
            for table in tables:
                print(f"- {table[0]}")
            
            # law_metadata 테이블 구조 확인
            cursor.execute("PRAGMA table_info(law_metadata)")
            columns = cursor.fetchall()
            print(f"\nlaw_metadata 테이블 컬럼:")
            for col in columns:
                print(f"- {col[1]} ({col[2]})")
        
        print("데이터베이스 초기화 완료!")
        
    except Exception as e:
        print(f"데이터베이스 초기화 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    initialize_database()

