#!/usr/bin/env python3
"""
precedent_cases 테이블 분석 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

def analyze_precedent_cases():
    """precedent_cases 테이블 분석"""
    try:
        db_manager = DatabaseManager()
        
        print("📊 precedent_cases 테이블 분석")
        print("="*50)
        
        # 기본 통계
        print("1. 기본 통계:")
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM precedent_cases")
        total_count = result[0]['count'] if result else 0
        print(f"  총 판례 수: {total_count:,}개")
        
        # 텍스트 필드 상태 확인
        print("\n2. 텍스트 필드 상태:")
        result = db_manager.execute_query("""
            SELECT 
                COUNT(CASE WHEN full_text IS NOT NULL AND full_text != '' THEN 1 END) as has_full_text,
                COUNT(CASE WHEN searchable_text IS NOT NULL AND searchable_text != '' THEN 1 END) as has_searchable_text
            FROM precedent_cases
        """)
        
        if result:
            row = result[0]
            print(f"  full_text 있음: {row['has_full_text']:,}")
            print(f"  searchable_text 있음: {row['has_searchable_text']:,}")
        
        # 카테고리별 분포
        print("\n3. 카테고리별 분포:")
        result = db_manager.execute_query("""
            SELECT category, COUNT(*) as count
            FROM precedent_cases
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        
        for row in result:
            print(f"  {row['category']}: {row['count']:,}개")
        
        # 분야별 분포
        print("\n4. 분야별 분포:")
        result = db_manager.execute_query("""
            SELECT field, COUNT(*) as count
            FROM precedent_cases
            WHERE field IS NOT NULL AND field != ''
            GROUP BY field
            ORDER BY count DESC
            LIMIT 10
        """)
        
        for row in result:
            print(f"  {row['field']}: {row['count']:,}개")
        
        # 법원별 분포
        print("\n5. 법원별 분포:")
        result = db_manager.execute_query("""
            SELECT court, COUNT(*) as count
            FROM precedent_cases
            WHERE court IS NOT NULL AND court != ''
            GROUP BY court
            ORDER BY count DESC
            LIMIT 10
        """)
        
        for row in result:
            print(f"  {row['court']}: {row['count']:,}개")
        
        # 샘플 데이터 확인
        print("\n6. 샘플 데이터:")
        sample = db_manager.execute_query("""
            SELECT case_name, case_number, category, field, court, full_text
            FROM precedent_cases 
            WHERE full_text IS NOT NULL AND full_text != ''
            LIMIT 3
        """)
        
        for i, row in enumerate(sample, 1):
            print(f"\n  샘플 {i}: {row['case_name']}")
            print(f"    사건번호: {row['case_number']}")
            print(f"    카테고리: {row['category']}")
            print(f"    분야: {row['field']}")
            print(f"    법원: {row['court']}")
            if row['full_text']:
                text_preview = row['full_text'][:200] + "..." if len(row['full_text']) > 200 else row['full_text']
                print(f"    내용: {text_preview}")
        
        # 판례 섹션 데이터 확인
        print("\n7. precedent_sections 테이블 통계:")
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM precedent_sections")
        sections_count = result[0]['count'] if result else 0
        print(f"  총 섹션 수: {sections_count:,}개")
        
        # 섹션 타입별 분포
        result = db_manager.execute_query("""
            SELECT section_type, COUNT(*) as count
            FROM precedent_sections
            WHERE section_type IS NOT NULL AND section_type != ''
            GROUP BY section_type
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("  섹션 타입별 분포:")
        for row in result:
            print(f"    {row['section_type']}: {row['count']:,}개")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_precedent_cases()
