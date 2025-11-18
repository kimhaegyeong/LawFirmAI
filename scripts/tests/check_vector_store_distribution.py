# -*- coding: utf-8 -*-
"""
벡터 스토어 데이터 분포 확인 스크립트
데이터베이스에서 문서 타입별 분포를 확인합니다.
"""

import sys
import os
import sqlite3
from pathlib import Path
from typing import Dict, Any

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

def check_database_distribution():
    """데이터베이스에서 문서 타입별 분포 확인"""
    print("\n" + "=" * 80)
    print("벡터 스토어 데이터 분포 확인")
    print("=" * 80)
    
    try:
        from core.utils.config import Config
        config = Config()
        db_path = config.database_path
        
        if not os.path.exists(db_path):
            print(f"\n❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
            return
        
        print(f"\n📁 데이터베이스 경로: {db_path}")
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. text_chunks 테이블 확인
        print("\n1️⃣ text_chunks 테이블 문서 타입 분포:")
        print("-" * 80)
        try:
            cursor.execute("""
                SELECT source_type, COUNT(*) as count 
                FROM text_chunks 
                GROUP BY source_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            
            if results:
                total = sum(row['count'] for row in results)
                print(f"   총 문서 수: {total:,}개")
                print()
                for row in results:
                    doc_type = row['source_type'] or 'unknown'
                    count = row['count']
                    percentage = (count / total * 100) if total > 0 else 0
                    print(f"   - {doc_type}: {count:,}개 ({percentage:.1f}%)")
            else:
                print("   ⚠️  데이터가 없습니다.")
        except sqlite3.OperationalError as e:
            print(f"   ❌ 오류: {e}")
            print("   text_chunks 테이블이 없거나 source_type 컬럼이 없을 수 있습니다.")
        
        # 2. embeddings 테이블 확인
        print("\n2️⃣ embeddings 테이블 문서 타입 분포:")
        print("-" * 80)
        try:
            cursor.execute("""
                SELECT source_type, COUNT(*) as count 
                FROM embeddings 
                GROUP BY source_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            
            if results:
                total = sum(row['count'] for row in results)
                print(f"   총 임베딩 수: {total:,}개")
                print()
                for row in results:
                    doc_type = row['source_type'] or 'unknown'
                    count = row['count']
                    percentage = (count / total * 100) if total > 0 else 0
                    print(f"   - {doc_type}: {count:,}개 ({percentage:.1f}%)")
            else:
                print("   ⚠️  데이터가 없습니다.")
        except sqlite3.OperationalError as e:
            print(f"   ❌ 오류: {e}")
            print("   embeddings 테이블이 없거나 source_type 컬럼이 없을 수 있습니다.")
        
        # 3. 각 소스 테이블 확인
        print("\n3️⃣ 소스 테이블별 문서 수:")
        print("-" * 80)
        source_tables = [
            ('statute_articles', '법령 조문'),
            ('case_paragraphs', '판례'),
            ('decision_paragraphs', '결정례'),
            ('interpretation_paragraphs', '해석례')
        ]
        
        for table_name, table_desc in source_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                result = cursor.fetchone()
                count = result['count'] if result else 0
                print(f"   - {table_desc} ({table_name}): {count:,}개")
            except sqlite3.OperationalError:
                print(f"   - {table_desc} ({table_name}): 테이블 없음")
        
        # 4. 최근 추가된 문서 확인
        print("\n4️⃣ 최근 추가된 문서 (최근 10개):")
        print("-" * 80)
        try:
            cursor.execute("""
                SELECT source_type, COUNT(*) as count
                FROM text_chunks
                WHERE id > (SELECT MAX(id) - 100 FROM text_chunks)
                GROUP BY source_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            
            if results:
                for row in results:
                    doc_type = row['source_type'] or 'unknown'
                    count = row['count']
                    print(f"   - {doc_type}: {count}개 (최근 100개 중)")
            else:
                print("   ⚠️  최근 데이터가 없습니다.")
        except sqlite3.OperationalError as e:
            print(f"   ❌ 오류: {e}")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ 데이터 분포 확인 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_database_distribution()

