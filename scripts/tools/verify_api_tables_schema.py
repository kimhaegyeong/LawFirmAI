#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 서버용 테이블 (users, sessions, messages) 스키마 검증 스크립트
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
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
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

def verify_table_schema(engine, table_name, expected_columns):
    """테이블 스키마 검증"""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name, schema='public')
    
    actual_columns = {col['name']: str(col['type']) for col in columns}
    
    print(f"\n📊 테이블: {table_name}")
    print("-" * 80)
    
    all_match = True
    for col_name, expected_type in expected_columns.items():
        if col_name not in actual_columns:
            print(f"  ❌ 누락된 컬럼: {col_name}")
            all_match = False
        else:
            actual_type = actual_columns[col_name]
            # 타입 비교 (대소문자 무시, 일부 타입은 유사성만 확인)
            if expected_type.lower() in actual_type.lower() or actual_type.lower() in expected_type.lower():
                print(f"  ✅ {col_name}: {actual_type}")
            else:
                print(f"  ⚠️  {col_name}: {actual_type} (예상: {expected_type})")
                # JSONB와 JSON은 호환 가능하므로 경고만
                if 'json' in expected_type.lower() and 'json' in actual_type.lower():
                    print(f"      (JSON/JSONB는 호환 가능)")
                else:
                    all_match = False
    
    return all_match

def main():
    """메인 함수"""
    db_url = build_database_url()
    if not db_url:
        print("❌ DATABASE_URL을 구성할 수 없습니다.")
        return 1
    
    if not db_url.startswith('postgresql'):
        print(f"⚠️  이 스크립트는 PostgreSQL에서만 작동합니다.")
        return 1
    
    print("=" * 80)
    print("API 서버용 테이블 스키마 검증")
    print("=" * 80)
    print(f"데이터베이스: {db_url.split('@')[1] if '@' in db_url else db_url}")
    
    engine = get_database_connection(database_url=db_url)
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        
        # 예상 스키마 정의
        expected_schemas = {
            'users': {
                'user_id': 'VARCHAR',
                'email': 'VARCHAR',
                'name': 'TEXT',
                'picture': 'TEXT',
                'provider': 'VARCHAR',
                'google_access_token': 'TEXT',
                'google_refresh_token': 'TEXT',
                'created_at': 'TIMESTAMP',
                'updated_at': 'TIMESTAMP',
            },
            'sessions': {
                'session_id': 'VARCHAR',
                'title': 'TEXT',
                'created_at': 'TIMESTAMP',
                'updated_at': 'TIMESTAMP',
                'message_count': 'INTEGER',
                'user_id': 'VARCHAR',
                'ip_address': 'VARCHAR',
            },
            'messages': {
                'message_id': 'VARCHAR',
                'session_id': 'VARCHAR',
                'role': 'VARCHAR',
                'content': 'TEXT',
                'timestamp': 'TIMESTAMP',
                'metadata': 'JSONB',  # 또는 JSON
            },
        }
        
        all_valid = True
        for table_name in ['users', 'sessions', 'messages']:
            if table_name not in tables:
                print(f"\n❌ 테이블이 존재하지 않습니다: {table_name}")
                all_valid = False
                continue
            
            is_valid = verify_table_schema(engine, table_name, expected_schemas[table_name])
            if not is_valid:
                all_valid = False
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ 모든 테이블 스키마가 정상입니다!")
        else:
            print("⚠️  일부 테이블 스키마에 문제가 있습니다.")
            print("   마이그레이션 스크립트를 실행하거나 SQLAlchemy init_database()를 사용하세요.")
        print("=" * 80)
        
        return 0 if all_valid else 1
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

