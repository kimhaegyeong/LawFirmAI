#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 서버용 테이블 (users, sessions, messages) 존재 여부 확인 및 생성 스크립트
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

def check_tables_exist(engine):
    """users, sessions, messages 테이블 존재 여부 확인"""
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema='public')
    
    required_tables = ['users', 'sessions', 'messages']
    existing_tables = []
    missing_tables = []
    
    for table in required_tables:
        if table in tables:
            existing_tables.append(table)
        else:
            missing_tables.append(table)
    
    return existing_tables, missing_tables

def create_tables(engine):
    """SQLAlchemy 모델을 사용하여 테이블 생성"""
    print("📦 SQLAlchemy 모델을 사용하여 테이블 생성 중...")
    
    # 프로젝트 루트를 sys.path에 추가
    sys.path.insert(0, str(_PROJECT_ROOT))
    
    # .env 파일 로드
    try:
        from utils.env_loader import ensure_env_loaded
        ensure_env_loaded(_PROJECT_ROOT)
    except ImportError:
        try:
            from dotenv import load_dotenv
            root_env = _PROJECT_ROOT / ".env"
            if root_env.exists():
                load_dotenv(dotenv_path=str(root_env), override=False)
            langgraph_env = _PROJECT_ROOT / "lawfirm_langgraph" / ".env"
            if langgraph_env.exists():
                load_dotenv(dotenv_path=str(langgraph_env), override=True)
        except ImportError:
            pass
    
    # SQLAlchemy 모델 임포트
    from api.database.connection import init_database
    
    try:
        init_database()
        print("✅ 테이블 생성 완료!")
        return True
    except Exception as e:
        print(f"❌ 테이블 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    db_url = build_database_url()
    if not db_url:
        print("❌ DATABASE_URL을 구성할 수 없습니다.")
        print("환경 변수 DATABASE_URL 또는 POSTGRES_* 변수를 설정하세요.")
        return 1
    
    if not db_url.startswith('postgresql'):
        print(f"⚠️  이 스크립트는 PostgreSQL에서만 작동합니다.")
        print(f"현재 데이터베이스: {db_url.split('://')[0]}")
        return 1
    
    print("=" * 80)
    print("API 서버용 테이블 확인 및 생성")
    print("=" * 80)
    print(f"데이터베이스: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()
    
    engine = get_database_connection(database_url=db_url)
    
    try:
        # 테이블 존재 여부 확인
        existing_tables, missing_tables = check_tables_exist(engine)
        
        print("📋 테이블 상태:")
        if existing_tables:
            print(f"  ✅ 존재하는 테이블: {', '.join(existing_tables)}")
        if missing_tables:
            print(f"  ❌ 누락된 테이블: {', '.join(missing_tables)}")
        
        print()
        
        if not missing_tables:
            print("✅ 모든 테이블이 존재합니다!")
            return 0
        
        # 누락된 테이블이 있으면 생성
        print(f"⚠️  누락된 테이블 {len(missing_tables)}개를 생성합니다...")
        print()
        
        success = create_tables(engine)
        
        if success:
            # 다시 확인
            existing_tables, missing_tables = check_tables_exist(engine)
            if not missing_tables:
                print()
                print("✅ 모든 테이블이 정상적으로 생성되었습니다!")
                return 0
            else:
                print()
                print(f"⚠️  여전히 누락된 테이블: {', '.join(missing_tables)}")
                return 1
        else:
            return 1
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

