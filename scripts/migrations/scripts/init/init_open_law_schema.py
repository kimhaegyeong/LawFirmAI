#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Law 스키마 초기화 스크립트
PostgreSQL 데이터베이스에 Open Law API 수집을 위한 스키마를 생성합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils/env_loader.py 사용)
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        # scripts/.env 파일 우선 로드
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        # 프로젝트 루트 .env 파일 로드
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 유틸리티 임포트
from scripts.migrations.utils.database import build_database_url, get_database_connection

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Open Law 스키마 초기화')
    parser.add_argument(
        '--db',
        default=build_database_url() or os.getenv('DATABASE_URL'),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--sql-file',
        default='scripts/migrations/schema/create_open_law_schema.sql',
        help='SQL 스키마 파일 경로'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # SQL 파일 읽기
    sql_file = Path(args.sql_file)
    if not sql_file.is_absolute():
        sql_file = _PROJECT_ROOT / sql_file
    
    if not sql_file.exists():
        logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file}")
        return
    
    # 데이터베이스 연결
    logger.info(f"데이터베이스 연결: {args.db}")
    engine = get_database_connection(database_url=args.db)
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # SQL 실행
    with engine.connect() as conn:
        # 트랜잭션 시작
        trans = conn.begin()
        try:
            # SQL 문장들을 세미콜론으로 분리하여 실행
            # 주석만 있는 줄은 제외
            statements = []
            for s in sql_content.split(';'):
                s = s.strip()
                # 주석만 있는 줄이나 빈 줄은 제외
                if s and not all(line.strip().startswith('--') or not line.strip() for line in s.split('\n')):
                    statements.append(s)
            
            for i, statement in enumerate(statements, 1):
                if statement:
                    logger.info(f"SQL 문장 {i}/{len(statements)} 실행 중...")
                    conn.execute(text(statement))
            
            trans.commit()
            logger.info("스키마 생성 완료")
        
        except Exception as e:
            trans.rollback()
            logger.error(f"스키마 생성 실패: {e}")
            raise


if __name__ == '__main__':
    main()

