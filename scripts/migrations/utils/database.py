"""
데이터베이스 연결 및 SQL 실행 유틸리티
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


def build_database_url() -> Optional[str]:
    """
    데이터베이스 URL 구성
    
    우선순위:
    1. DATABASE_URL 환경변수
    2. POSTGRES_* 환경변수 조합
    
    Returns:
        데이터베이스 URL 또는 None
    """
    # 1. DATABASE_URL 환경변수 확인
    db_url = os.getenv('DATABASE_URL')
    if db_url and db_url.strip():
        logger.debug("DATABASE_URL 환경변수에서 가져옴")
        return db_url
    
    # 2. POSTGRES_* 환경변수로 구성
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'lawfirmai_local')
    user = os.getenv('POSTGRES_USER', 'lawfirmai')
    password = os.getenv('POSTGRES_PASSWORD', 'local_password')
    
    logger.debug(f"POSTGRES_* 환경변수로 구성: host={host}, port={port}, db={db}, user={user}")
    
    # password는 URL 인코딩 필요
    encoded_password = quote_plus(password)
    url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
    logger.debug(f"구성된 URL: postgresql://{user}:***@{host}:{port}/{db}")
    return url


def get_database_connection(database_url: Optional[str] = None, pool_pre_ping: bool = True):
    """
    데이터베이스 연결 생성
    
    Args:
        database_url: 데이터베이스 URL (None이면 build_database_url() 사용)
        pool_pre_ping: 연결 풀 사전 핑 여부
    
    Returns:
        SQLAlchemy Engine
    """
    if database_url is None:
        database_url = build_database_url()
    
    if not database_url:
        raise ValueError("데이터베이스 URL을 구성할 수 없습니다. DATABASE_URL 또는 POSTGRES_* 환경변수를 설정하세요.")
    
    return create_engine(
        database_url,
        pool_pre_ping=pool_pre_ping,
        echo=False
    )


def execute_sql_file(
    sql_file: Path,
    database_url: Optional[str] = None,
    use_psql: bool = True,
    on_error_stop: bool = False
) -> bool:
    """
    SQL 파일 실행
    
    Args:
        sql_file: SQL 파일 경로
        database_url: 데이터베이스 URL (None이면 build_database_url() 사용)
        use_psql: psql 사용 여부 (True면 psql, False면 psycopg2)
        on_error_stop: 오류 발생 시 중단 여부
    
    Returns:
        성공 여부
    """
    if not sql_file.exists():
        logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file}")
        return False
    
    if database_url is None:
        database_url = build_database_url()
    
    if not database_url:
        logger.error("데이터베이스 URL을 구성할 수 없습니다.")
        return False
    
    logger.info(f"SQL 파일 읽기: {sql_file}")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # psql 사용 시도
    if use_psql:
        try:
            parsed = urlparse(database_url)
            dbname = parsed.path[1:] if parsed.path.startswith('/') else parsed.path
            user = parsed.username
            password = parsed.password
            host = parsed.hostname
            port = parsed.port or 5432
            
            # PGPASSWORD 환경변수 설정
            env = os.environ.copy()
            if password:
                env['PGPASSWORD'] = password
            
            # psql 명령어 구성
            psql_cmd = [
                'psql',
                '-h', host,
                '-p', str(port),
                '-U', user,
                '-d', dbname,
                '-f', str(sql_file.absolute()),
                '-v', f'ON_ERROR_STOP={"1" if on_error_stop else "0"}'
            ]
            
            logger.info(f"psql 명령어 실행: {' '.join(psql_cmd[:6])}...")
            
            # psql 실행
            result = subprocess.run(
                psql_cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                logger.info("✅ SQL 파일 실행 완료!")
                if result.stdout:
                    logger.debug(f"psql 출력:\n{result.stdout}")
                return True
            else:
                # 일부 오류는 무시 (이미 존재하는 경우 등)
                error_output = result.stderr or result.stdout or ""
                if any(keyword in error_output.lower() for keyword in ['already exists', 'duplicate', 'if not exists']):
                    logger.warning(f"일부 항목이 이미 존재합니다 (무시)")
                    logger.info("✅ SQL 파일 실행 완료 (일부 항목 건너뜀)")
                    return True
                else:
                    logger.error(f"❌ SQL 파일 실행 실패 (exit code: {result.returncode})")
                    if result.stderr:
                        logger.error(f"오류 출력:\n{result.stderr}")
                    if result.stdout:
                        logger.error(f"표준 출력:\n{result.stdout}")
                    return False
        
        except FileNotFoundError:
            logger.warning("psql 명령어를 찾을 수 없습니다. psycopg2를 사용합니다...")
            use_psql = False
    
    # Fallback: psycopg2 사용
    if not use_psql:
        try:
            import psycopg2
            from .sql_parser import parse_sql_statements
            
            conn = psycopg2.connect(database_url)
            conn.autocommit = True  # 각 문장을 자동 커밋
            
            try:
                cursor = conn.cursor()
                
                # SQL 문장 파싱
                statements = parse_sql_statements(sql_content)
                
                logger.info(f"총 {len(statements)}개의 SQL 문장 실행 중...")
                for i, statement in enumerate(statements, 1):
                    if not statement or statement.strip() == ';':
                        continue
                    try:
                        cursor.execute(statement)
                        logger.debug(f"문장 {i}/{len(statements)} 실행 완료")
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['already exists', 'duplicate', 'if not exists']):
                            logger.debug(f"문장 {i} 건너뜀 (이미 존재)")
                        else:
                            logger.warning(f"문장 {i} 실행 중 오류: {str(e)[:200]}")
                
                logger.info("✅ SQL 파일 실행 완료!")
                return True
            finally:
                conn.close()
        
        except Exception as e:
            logger.error(f"❌ SQL 파일 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return False

