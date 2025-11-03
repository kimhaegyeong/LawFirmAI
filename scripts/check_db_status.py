# -*- coding: utf-8 -*-
"""
데이터베이스 상태 확인 스크립트
마이그레이션은 수행하지 않고 읽기 전용으로 확인만 합니다.
"""

import sqlite3
import os
import sys
from pathlib import Path

def check_database_status():
    """데이터베이스 상태 확인"""
    print("\n" + "=" * 60)
    print("데이터베이스 상태 확인")
    print("=" * 60)
    
    # 설정에서 데이터베이스 경로 가져오기
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "lawfirm_langgraph"))
        from langgraph_core.utils.config import Config
        config = Config()
        db_path = config.database_path
    except Exception as e:
        # 기본 경로 사용
        db_path = "./data/lawfirm_v2.db"
        print(f"⚠️ Config 로드 실패, 기본 경로 사용: {e}")
    
    # 절대 경로로 변환
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    
    print(f"\n📁 데이터베이스 경로: {db_path}")
    
    # 파일 존재 확인
    if not os.path.exists(db_path):
        print(f"❌ 데이터베이스 파일이 없습니다: {db_path}")
        print("\n초기화 방법:")
        print("  python scripts/init_lawfirm_v2_db.py")
        return False
    
    file_size = os.path.getsize(db_path)
    print(f"✅ 데이터베이스 파일 존재 (크기: {file_size:,} bytes = {file_size/1024/1024:.2f} MB)")
    
    # 데이터베이스 연결
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # SQLite 버전 확인
        cursor.execute("SELECT sqlite_version()")
        sqlite_version = cursor.fetchone()[0]
        print(f"\n🔧 SQLite 버전: {sqlite_version}")
        
        # 모든 테이블 목록 조회
        cursor.execute("""
            SELECT name, type 
            FROM sqlite_master 
            WHERE type IN ('table', 'view')
            ORDER BY type, name
        """)
        all_objects = cursor.fetchall()
        
        tables = [name for name, obj_type in all_objects if obj_type == 'table']
        views = [name for name, obj_type in all_objects if obj_type == 'view']
        
        print(f"\n📊 테이블 목록: {len(tables)}개")
        for table in tables:
            print(f"  - {table}")
        
        if views:
            print(f"\n👁️ 뷰 목록: {len(views)}개")
            for view in views:
                print(f"  - {view}")
        
        # FTS5 가상 테이블 확인
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'table' AND name LIKE '%_fts'
            ORDER BY name
        """)
        fts_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n🔍 FTS5 가상 테이블: {len(fts_tables)}개")
        for fts_table in fts_tables:
            print(f"  - {fts_table}")
        
        # 주요 테이블 데이터 확인
        print("\n📈 주요 테이블 데이터 통계:")
        print("-" * 60)
        
        checks = [
            ("domains", "SELECT COUNT(*) FROM domains"),
            ("statutes", "SELECT COUNT(*) FROM statutes"),
            ("statute_articles", "SELECT COUNT(*) FROM statute_articles"),
            ("cases", "SELECT COUNT(*) FROM cases"),
            ("case_paragraphs", "SELECT COUNT(*) FROM case_paragraphs"),
            ("decision_paragraphs", "SELECT COUNT(*) FROM decision_paragraphs"),
            ("interpretation_paragraphs", "SELECT COUNT(*) FROM interpretation_paragraphs"),
            ("text_chunks", "SELECT COUNT(*) FROM text_chunks"),
            ("embeddings", "SELECT COUNT(*) FROM embeddings"),
        ]
        
        missing_tables = []
        for name, query in checks:
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                status = "✅" if count > 0 else "⚠️"
                print(f"{status} {name:30s}: {count:,}개")
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    print(f"❌ {name:30s}: 테이블 없음")
                    missing_tables.append(name)
                else:
                    print(f"❌ {name:30s}: 오류 - {e}")
        
        # 필수 테이블 존재 여부 확인
        print("\n" + "=" * 60)
        print("필수 테이블 존재 여부 확인")
        print("=" * 60)
        
        required_tables = {
            "embeddings": "벡터 검색용 임베딩 테이블",
            "statute_articles_fts": "법령 조문 FTS5 검색 테이블",
            "case_paragraphs_fts": "판례 FTS5 검색 테이블",
            "text_chunks": "텍스트 청크 메타데이터 테이블",
        }
        
        all_present = True
        for table_name, description in required_tables.items():
            if table_name in tables or table_name in fts_tables:
                print(f"✅ {table_name:30s}: 존재 - {description}")
            else:
                print(f"❌ {table_name:30s}: 없음 - {description}")
                all_present = False
        
        # 인덱스 확인
        print("\n" + "=" * 60)
        print("인덱스 확인")
        print("=" * 60)
        
        cursor.execute("""
            SELECT name, tbl_name, type
            FROM sqlite_master 
            WHERE type = 'index' AND tbl_name IN ('embeddings', 'text_chunks', 'statute_articles', 'case_paragraphs')
            ORDER BY tbl_name, name
        """)
        indexes = cursor.fetchall()
        
        if indexes:
            print(f"\n발견된 인덱스: {len(indexes)}개")
            for idx_name, tbl_name, idx_type in indexes:
                print(f"  - {tbl_name}.{idx_name} ({idx_type})")
        else:
            print("⚠️ 관련 인덱스를 찾을 수 없습니다.")
        
        conn.close()
        
        # 최종 상태 요약
        print("\n" + "=" * 60)
        print("상태 요약")
        print("=" * 60)
        
        if all_present:
            print("✅ 모든 필수 테이블이 존재합니다.")
        else:
            print("⚠️ 일부 필수 테이블이 없습니다.")
            print("\n해결 방법:")
            print("1. 마이그레이션 스크립트 실행:")
            print("   sqlite3 data/lawfirm_v2.db < scripts/migrations/001_create_lawfirm_v2.sql")
            print("2. 인덱스 최적화:")
            print("   sqlite3 data/lawfirm_v2.db < scripts/migrations/002_optimize_indexes.sql")
            print("3. 벡터 임베딩 생성 (데이터가 있는 경우):")
            print("   python scripts/data_processing/incremental_preprocessor.py")
        
        return all_present
        
    except sqlite3.Error as e:
        print(f"❌ 데이터베이스 접근 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    check_database_status()

