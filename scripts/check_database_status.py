#!/usr/bin/env python3
"""
데이터베이스 상태 확인 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

def check_database_status():
    """데이터베이스 상태 확인"""
    try:
        db_manager = DatabaseManager()
        
        print("📊 데이터베이스 상태 확인")
        print("="*50)
        
        # 테이블별 레코드 수 확인
        tables = [
            'assembly_laws',
            'assembly_articles', 
            'precedent_cases',
            'precedent_sections',
            'processed_files'
        ]
        
        for table in tables:
            try:
                result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                count = result[0]['count'] if result else 0
                print(f"  {table}: {count:,}개 레코드")
            except Exception as e:
                print(f"  {table}: 오류 - {e}")
        
        # assembly_laws 테이블 샘플 데이터 확인
        print("\n📋 assembly_laws 테이블 샘플 데이터:")
        try:
            sample = db_manager.execute_query("SELECT law_name, category, ministry FROM assembly_laws LIMIT 5")
            if sample:
                for row in sample:
                    print(f"  - {row['law_name']} ({row['category']}, {row['ministry']})")
            else:
                print("  데이터가 없습니다.")
        except Exception as e:
            print(f"  오류: {e}")
        
        # 데이터베이스 파일 경로 확인
        db_path = "data/lawfirm.db"
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"\n💾 데이터베이스 파일: {db_path} ({file_size:,} bytes)")
        else:
            print(f"\n❌ 데이터베이스 파일이 존재하지 않습니다: {db_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_status()
