#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복 데이터 제거 SQL 실행 스크립트
cleanup_duplicate_articles.sql 파일을 실행하여 중복 데이터를 제거합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/execute_cleanup_sql.py -> 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils/env_loader.py 사용)
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 유틸리티 임포트
from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def execute_sql_file(conn, sql_file_path: Path):
    """SQL 파일을 읽어서 실행"""
    
    if not sql_file_path.exists():
        raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
    
    print(f"📄 SQL 파일 읽기: {sql_file_path}")
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # SQL 문을 세미콜론으로 분리
    # 주석 라인은 제외하고 실제 SQL만 추출
    sql_statements = []
    current_statement = []
    
    for line in sql_content.split('\n'):
        line = line.strip()
        # 주석 라인 건너뛰기
        if line.startswith('--') or not line:
            continue
        
        current_statement.append(line)
        
        # 세미콜론으로 끝나면 하나의 SQL 문 완성
        if line.endswith(';'):
            statement = ' '.join(current_statement)
            if statement.strip():
                sql_statements.append(statement)
            current_statement = []
    
    # 마지막 문장 처리
    if current_statement:
        statement = ' '.join(current_statement)
        if statement.strip():
            sql_statements.append(statement)
    
    print(f"📊 실행할 SQL 문 개수: {len(sql_statements)}개")
    print()
    
    # 트랜잭션 시작
    deleted_count = 0
    error_count = 0
    
    try:
        for i, sql_stmt in enumerate(sql_statements, 1):
            try:
                result = conn.execute(text(sql_stmt))
                deleted = result.rowcount
                deleted_count += deleted
                
                if i % 50 == 0 or i == len(sql_statements):
                    print(f"진행 중... ({i}/{len(sql_statements)}) - 삭제된 레코드: {deleted_count}개")
                
            except Exception as e:
                error_count += 1
                logger.error(f"SQL 실행 오류 ({i}/{len(sql_statements)}): {e}")
                logger.error(f"실패한 SQL: {sql_stmt[:100]}...")
        
        # 트랜잭션은 context manager가 자동으로 커밋/롤백 처리
        print()
        print("✅ 모든 SQL 문 실행 완료")
        print(f"   삭제된 레코드: {deleted_count}개")
        if error_count > 0:
            print(f"   오류 발생: {error_count}개")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"트랜잭션 오류: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='중복 데이터 제거 SQL 실행')
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--sql-file',
        type=str,
        default=None,
        help='실행할 SQL 파일 경로 (기본값: cleanup_duplicate_articles.sql)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제로 실행하지 않고 SQL 문만 확인'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # SQL 파일 경로 결정
    if args.sql_file:
        sql_file_path = Path(args.sql_file)
    else:
        sql_file_path = Path(_PROJECT_ROOT) / "scripts" / "ingest" / "open_law" / "scripts" / "cleanup_duplicate_articles.sql"
    
    if not sql_file_path.exists():
        logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
        return
    
    print("=" * 80)
    print("중복 데이터 제거 SQL 실행")
    print("=" * 80)
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN 모드: 실제로 실행하지 않습니다")
        print()
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            statements = [s for s in content.split(';') if s.strip() and not s.strip().startswith('--')]
            print(f"실행될 SQL 문 개수: {len(statements)}개")
            print()
            print("처음 5개 SQL 문 미리보기:")
            for i, stmt in enumerate(statements[:5], 1):
                print(f"{i}. {stmt.strip()[:100]}...")
        return
    
    # 데이터베이스 연결
    engine = create_engine(
        args.db,
        pool_pre_ping=True,
        echo=False
    )
    
    with engine.begin() as conn:
        try:
            execute_sql_file(conn, sql_file_path)
            print()
            print("=" * 80)
            print("✅ 중복 데이터 제거 완료")
            print("=" * 80)
        except Exception as e:
            print()
            print("=" * 80)
            print(f"❌ 오류 발생: {e}")
            print("   변경사항이 롤백되었습니다.")
            print("=" * 80)
            raise


if __name__ == '__main__':
    main()

