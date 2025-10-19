#!/usr/bin/env python3
"""
법령 데이터 분석 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

def analyze_law_data():
    """법령 데이터 분석"""
    try:
        db_manager = DatabaseManager()
        
        print("📊 법령 데이터 분석")
        print("="*50)
        
        # full_text 필드 상태 확인
        print("1. full_text 필드 상태:")
        result = db_manager.execute_query("""
            SELECT 
                COUNT(*) as total,
                COUNT(full_text) as has_full_text,
                COUNT(CASE WHEN full_text IS NOT NULL AND full_text != '' THEN 1 END) as non_empty_full_text
            FROM assembly_laws
        """)
        
        if result:
            row = result[0]
            print(f"  총 레코드: {row['total']:,}")
            print(f"  full_text 있음: {row['has_full_text']:,}")
            print(f"  full_text 비어있지 않음: {row['non_empty_full_text']:,}")
        
        # 다른 텍스트 필드 확인
        print("\n2. 다른 텍스트 필드 상태:")
        result = db_manager.execute_query("""
            SELECT 
                COUNT(CASE WHEN searchable_text IS NOT NULL AND searchable_text != '' THEN 1 END) as has_searchable_text,
                COUNT(CASE WHEN html_clean_text IS NOT NULL AND html_clean_text != '' THEN 1 END) as has_html_clean_text,
                COUNT(CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 END) as has_summary
            FROM assembly_laws
        """)
        
        if result:
            row = result[0]
            print(f"  searchable_text 있음: {row['has_searchable_text']:,}")
            print(f"  html_clean_text 있음: {row['has_html_clean_text']:,}")
            print(f"  summary 있음: {row['has_summary']:,}")
        
        # 샘플 데이터 확인
        print("\n3. 샘플 데이터 (searchable_text):")
        sample = db_manager.execute_query("""
            SELECT law_name, searchable_text, html_clean_text, summary
            FROM assembly_laws 
            WHERE searchable_text IS NOT NULL AND searchable_text != ''
            LIMIT 3
        """)
        
        for i, row in enumerate(sample, 1):
            print(f"\n  샘플 {i}: {row['law_name']}")
            searchable_text = row['searchable_text'][:200] + "..." if len(row['searchable_text']) > 200 else row['searchable_text']
            print(f"    searchable_text: {searchable_text}")
            
            if row['html_clean_text']:
                html_text = row['html_clean_text'][:200] + "..." if len(row['html_clean_text']) > 200 else row['html_clean_text']
                print(f"    html_clean_text: {html_text}")
            
            if row['summary']:
                summary = row['summary'][:200] + "..." if len(row['summary']) > 200 else row['summary']
                print(f"    summary: {summary}")
        
        # 카테고리별 분포
        print("\n4. 카테고리별 분포:")
        result = db_manager.execute_query("""
            SELECT category, COUNT(*) as count
            FROM assembly_laws
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        
        for row in result:
            print(f"  {row['category']}: {row['count']:,}개")
        
        # 소관부처별 분포
        print("\n5. 소관부처별 분포:")
        result = db_manager.execute_query("""
            SELECT ministry, COUNT(*) as count
            FROM assembly_laws
            WHERE ministry IS NOT NULL AND ministry != ''
            GROUP BY ministry
            ORDER BY count DESC
            LIMIT 10
        """)
        
        for row in result:
            print(f"  {row['ministry']}: {row['count']:,}개")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_law_data()
