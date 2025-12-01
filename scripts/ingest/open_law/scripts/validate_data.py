#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수집된 데이터 검증 스크립트
PostgreSQL 데이터베이스의 데이터 품질을 검증합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/validate_data.py -> 프로젝트 루트
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


def validate_statutes(conn, domain: str = None):
    """법령 데이터 검증"""
    issues = []
    
    # 필수 필드 검증
    if domain:
        where_clause = f"WHERE domain = '{domain}' AND (law_id IS NULL OR law_name_kr IS NULL OR law_name_kr = '')"
    else:
        where_clause = "WHERE law_id IS NULL OR law_name_kr IS NULL OR law_name_kr = ''"
    
    result = conn.execute(
        text(f"""
            SELECT COUNT(*) 
            FROM statutes 
            {where_clause}
        """)
    )
    null_count = result.fetchone()[0]
    if null_count > 0:
        issues.append(f"법령: 필수 필드가 비어있는 레코드 {null_count}개")
    
    # 중복 검증
    if domain:
        where_clause = f"WHERE domain = '{domain}'"
    else:
        where_clause = ""
    
    result = conn.execute(
        text(f"""
            SELECT law_id, COUNT(*) as cnt
            FROM statutes
            {where_clause}
            GROUP BY law_id
            HAVING COUNT(*) > 1
        """)
    )
    duplicates = result.fetchall()
    if duplicates:
        issues.append(f"법령: 중복된 law_id {len(duplicates)}개")
    
    # 조문 연결 검증
    result = conn.execute(
        text(f"""
            SELECT COUNT(*) 
            FROM statutes_articles sa
            LEFT JOIN statutes s ON sa.statute_id = s.id
            WHERE s.id IS NULL
        """)
    )
    orphan_count = result.fetchone()[0]
    if orphan_count > 0:
        issues.append(f"조문: 부모 법령이 없는 조문 {orphan_count}개")
    
    return issues


def validate_precedents(conn, domain: str = None):
    """판례 데이터 검증"""
    issues = []
    
    # 필수 필드 검증
    if domain:
        where_clause = f"WHERE domain = '{domain}' AND (precedent_id IS NULL OR case_name IS NULL OR case_name = '')"
    else:
        where_clause = "WHERE precedent_id IS NULL OR case_name IS NULL OR case_name = ''"
    
    result = conn.execute(
        text(f"""
            SELECT COUNT(*) 
            FROM precedents 
            {where_clause}
        """)
    )
    null_count = result.fetchone()[0]
    if null_count > 0:
        issues.append(f"판례: 필수 필드가 비어있는 레코드 {null_count}개")
    
    # 중복 검증
    if domain:
        where_clause = f"WHERE domain = '{domain}'"
    else:
        where_clause = ""
    
    result = conn.execute(
        text(f"""
            SELECT precedent_id, COUNT(*) as cnt
            FROM precedents
            {where_clause}
            GROUP BY precedent_id
            HAVING COUNT(*) > 1
        """)
    )
    duplicates = result.fetchall()
    if duplicates:
        issues.append(f"판례: 중복된 precedent_id {len(duplicates)}개")
    
    # 본문 연결 검증
    result = conn.execute(
        text(f"""
            SELECT COUNT(*) 
            FROM precedent_contents pc
            LEFT JOIN precedents p ON pc.precedent_id = p.id
            WHERE p.id IS NULL
        """)
    )
    orphan_count = result.fetchone()[0]
    if orphan_count > 0:
        issues.append(f"판례 본문: 부모 판례가 없는 본문 {orphan_count}개")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='수집된 데이터 검증')
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law', 'administrative_law'],
        help='검증할 분야 (지정하지 않으면 전체)'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # 데이터베이스 연결
    engine = create_engine(
        args.db,
        pool_pre_ping=True,
        echo=False
    )
    
    print("=" * 80)
    print("Open Law API 데이터 검증")
    print("=" * 80)
    print()
    
    all_issues = []
    
    with engine.connect() as conn:
        # 법령 검증
        print("📋 법령 데이터 검증 중...")
        statute_issues = validate_statutes(conn, args.domain)
        all_issues.extend(statute_issues)
        
        # 판례 검증
        print("⚖️  판례 데이터 검증 중...")
        precedent_issues = validate_precedents(conn, args.domain)
        all_issues.extend(precedent_issues)
        
        print()
        
        # 결과 출력
        if all_issues:
            print("⚠️  발견된 문제:")
            print("-" * 80)
            for issue in all_issues:
                print(f"  - {issue}")
            print()
            print(f"총 {len(all_issues)}개의 문제가 발견되었습니다.")
        else:
            print("✅ 데이터 검증 완료: 문제 없음")
        
        print()
        print("=" * 80)


if __name__ == '__main__':
    main()

