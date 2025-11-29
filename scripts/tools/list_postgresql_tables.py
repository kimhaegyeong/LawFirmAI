#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL 데이터베이스의 실제 테이블 목록 조회 스크립트
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import text, inspect, create_engine

def build_database_url():
    """데이터베이스 URL 구성"""
    # DATABASE_URL 환경 변수 확인
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # PostgreSQL 환경변수 조합
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
    postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
    
    encoded_password = quote_plus(postgres_password)
    db_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    return db_url

def get_database_connection(database_url):
    """데이터베이스 연결 생성"""
    return create_engine(
        database_url,
        pool_pre_ping=True,
        echo=False
    )

def main():
    """PostgreSQL 데이터베이스의 모든 테이블과 스키마 정보 조회"""
    db_url = build_database_url()
    if not db_url:
        print("❌ DATABASE_URL을 구성할 수 없습니다.")
        print("환경 변수 DATABASE_URL 또는 POSTGRES_* 변수를 설정하세요.")
        return
    
    if not db_url.startswith('postgresql'):
        print(f"⚠️  이 스크립트는 PostgreSQL에서만 작동합니다.")
        print(f"현재 데이터베이스: {db_url.split('://')[0]}")
        return
    
    print("=" * 80)
    print("PostgreSQL 데이터베이스 스키마 조회")
    print("=" * 80)
    print(f"데이터베이스 URL: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()
    
    engine = get_database_connection(database_url=db_url)
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        
        print(f"📋 총 테이블 수: {len(tables)}개\n")
        print("테이블 목록:")
        for i, table in enumerate(sorted(tables), 1):
            print(f"  {i:2d}. {table}")
        
        print("\n" + "=" * 80)
        print("테이블별 상세 정보")
        print("=" * 80)
        
        for table_name in sorted(tables):
            print(f"\n📊 테이블: {table_name}")
            print("-" * 80)
            
            # 컬럼 정보
            columns = inspector.get_columns(table_name, schema='public')
            print("컬럼:")
            for col in columns:
                col_type = str(col['type'])
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                default = f" DEFAULT {col['default']}" if col.get('default') is not None else ""
                print(f"  - {col['name']}: {col_type} {nullable}{default}")
            
            # 인덱스 정보
            indexes = inspector.get_indexes(table_name, schema='public')
            if indexes:
                print("\n인덱스:")
                for idx in indexes:
                    unique = "UNIQUE " if idx['unique'] else ""
                    column_names = [c for c in idx.get('column_names', []) if c is not None]
                    if column_names:
                        columns_str = ", ".join(column_names)
                        print(f"  - {unique}{idx['name']}: ({columns_str})")
                    else:
                        print(f"  - {unique}{idx['name']}")
            
            # 외래 키 정보
            foreign_keys = inspector.get_foreign_keys(table_name, schema='public')
            if foreign_keys:
                print("\n외래 키:")
                for fk in foreign_keys:
                    ref_table = fk['referred_table']
                    ref_columns = ", ".join(fk['referred_columns'])
                    columns = ", ".join(fk['constrained_columns'])
                    print(f"  - {columns} → {ref_table}({ref_columns})")
            
            # 제약조건 정보
            pk_constraint = inspector.get_pk_constraint(table_name, schema='public')
            if pk_constraint:
                print(f"\nPRIMARY KEY: {', '.join(pk_constraint['constrained_columns'])}")
            
            # CHECK 제약조건 조회
            with engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT conname, pg_get_constraintdef(oid) as definition
                    FROM pg_constraint
                    WHERE conrelid = '{table_name}'::regclass
                    AND contype = 'c'
                """))
                check_constraints = result.fetchall()
                if check_constraints:
                    print("\nCHECK 제약조건:")
                    for conname, definition in check_constraints:
                        print(f"  - {conname}: {definition}")
        
        print("\n" + "=" * 80)
        print("완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

