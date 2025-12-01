#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL 스키마 초기화 스크립트
001_create_lawfirm_v2_postgresql.sql 파일을 실행하여 스키마를 생성합니다.
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# run_postgresql_migration.py는 scripts/migrations/scripts/init/에 있으므로
# parents[4]가 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
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

# 공통 유틸리티 임포트
# sys.path에 프로젝트 루트가 이미 추가되어 있으므로 직접 임포트 시도
try:
    from scripts.migrations.utils.database import build_database_url, execute_sql_file
except ImportError:
    # 실패 시 직접 경로로 임포트
    import importlib.util
    utils_path = _PROJECT_ROOT / "scripts" / "migrations" / "utils" / "database.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Database utils file not found: {utils_path}")
    spec = importlib.util.spec_from_file_location("database_utils", str(utils_path))
    database_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_utils)
    build_database_url = database_utils.build_database_url
    execute_sql_file = database_utils.execute_sql_file

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='PostgreSQL 스키마 초기화')
    db_url = build_database_url()
    logger.info(f"build_database_url() 결과: {db_url}")
    parser.add_argument(
        '--db',
        default=db_url,
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--sql-file',
        default='scripts/migrations/schema/001_create_lawfirm_v2_postgresql.sql',
        help='SQL 스키마 파일 경로'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return 1
    
    # SQL 파일 경로 확인
    sql_file = Path(args.sql_file)
    if not sql_file.is_absolute():
        sql_file = _PROJECT_ROOT / sql_file
    
    # SQL 파일 실행
    success = execute_sql_file(sql_file, database_url=args.db, use_psql=True)
    
    if success:
        logger.info("✅ 스키마 초기화 완료!")
        return 0
    else:
        logger.error("❌ 스키마 초기화 실패")
        return 1


if __name__ == '__main__':
    sys.exit(main())

