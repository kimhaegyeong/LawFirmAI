#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수집 진행 상황 확인 스크립트
PostgreSQL 데이터베이스에서 수집된 데이터의 통계를 확인합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/check_collection_status.py -> 프로젝트 루트
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
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
    # 직접 구현 (fallback)
    from urllib.parse import quote_plus
    def build_database_url():
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        if db and user and password:
            encoded_password = quote_plus(password)
            return f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        return None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='수집 진행 상황 확인')
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law', 'administrative_law', 'all'],
        default='all',
        help='확인할 분야'
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
    print("Open Law API 데이터 수집 진행 상황")
    print("=" * 80)
    print()
    
    with engine.connect() as conn:
        # 법령 통계
        print("📋 법령 통계")
        print("-" * 80)
        
        if args.domain in ['civil_law', 'all']:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT law_id) as unique_laws
                    FROM statutes
                    WHERE domain = 'civil_law'
                """)
            )
            row = result.fetchone()
            print(f"  민사법: {row[0]}개 레코드, {row[1]}개 법령")
            
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE s.domain = 'civil_law'
                """)
            )
            article_count = result.fetchone()[0]
            print(f"  민사법 조문: {article_count}개")
        
        if args.domain in ['criminal_law', 'all']:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT law_id) as unique_laws
                    FROM statutes
                    WHERE domain = 'criminal_law'
                """)
            )
            row = result.fetchone()
            print(f"  형법: {row[0]}개 레코드, {row[1]}개 법령")
            
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE s.domain = 'criminal_law'
                """)
            )
            article_count = result.fetchone()[0]
            print(f"  형법 조문: {article_count}개")
        
        if args.domain in ['administrative_law', 'all']:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT law_id) as unique_laws
                    FROM statutes
                    WHERE domain = 'administrative_law'
                """)
            )
            row = result.fetchone()
            print(f"  행정법: {row[0]}개 레코드, {row[1]}개 법령")
            
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE s.domain = 'administrative_law'
                """)
            )
            article_count = result.fetchone()[0]
            print(f"  행정법 조문: {article_count}개")
        
        print()
        
        # 판례 통계
        print("⚖️  판례 통계")
        print("-" * 80)
        
        if args.domain in ['civil_law', 'all']:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT precedent_id) as unique_precedents
                    FROM precedents
                    WHERE domain = 'civil_law'
                """)
            )
            row = result.fetchone()
            print(f"  민사법: {row[0]}개 레코드, {row[1]}개 판례")
            
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM precedent_contents pc
                    JOIN precedents p ON pc.precedent_id = p.id
                    WHERE p.domain = 'civil_law'
                """)
            )
            content_count = result.fetchone()[0]
            print(f"  민사법 본문 섹션: {content_count}개")
            
            # 판례 수집 상태 상세 통계
            result = conn.execute(
                text("""
                    WITH section_counts AS (
                        SELECT 
                            p.precedent_id,
                            COUNT(DISTINCT pc.section_type) as cnt
                        FROM precedents p
                        LEFT JOIN precedent_contents pc ON p.id = pc.precedent_id
                        WHERE p.domain = 'civil_law'
                        GROUP BY p.precedent_id
                    )
                    SELECT 
                        COUNT(*) FILTER (WHERE cnt >= 3) as complete,
                        COUNT(*) FILTER (WHERE cnt > 0 AND cnt < 3) as partial,
                        COUNT(*) FILTER (WHERE cnt = 0 OR cnt IS NULL) as none,
                        COUNT(*) as total
                    FROM section_counts
                """)
            )
            status_row = result.fetchone()
            if status_row[3] > 0:
                complete_pct = (status_row[0] * 100) // status_row[3]
                partial_pct = (status_row[1] * 100) // status_row[3]
                none_pct = (status_row[2] * 100) // status_row[3]
                print(f"  민사법 판례 수집 상태:")
                print(f"    완전 수집 (3개 섹션): {status_row[0]:,}개 ({complete_pct}%)")
                print(f"    부분 수집 (1-2개 섹션): {status_row[1]:,}개 ({partial_pct}%)")
                print(f"    미수집 (0개 섹션): {status_row[2]:,}개 ({none_pct}%)")
            
            # 섹션 타입별 통계
            result = conn.execute(
                text("""
                    SELECT 
                        pc.section_type,
                        COUNT(*) as total
                    FROM precedent_contents pc
                    JOIN precedents p ON pc.precedent_id = p.id
                    WHERE p.domain = 'civil_law'
                    GROUP BY pc.section_type
                    ORDER BY pc.section_type
                """)
            )
            print(f"  민사법 섹션 타입별:")
            for row in result:
                print(f"    {row[0]}: {row[1]:,}개")
        
        if args.domain in ['criminal_law', 'all']:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT precedent_id) as unique_precedents
                    FROM precedents
                    WHERE domain = 'criminal_law'
                """)
            )
            row = result.fetchone()
            print(f"  형법: {row[0]}개 레코드, {row[1]}개 판례")
            
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM precedent_contents pc
                    JOIN precedents p ON pc.precedent_id = p.id
                    WHERE p.domain = 'criminal_law'
                """)
            )
            content_count = result.fetchone()[0]
            print(f"  형법 본문 섹션: {content_count}개")
            
            # 판례 수집 상태 상세 통계
            result = conn.execute(
                text("""
                    WITH section_counts AS (
                        SELECT 
                            p.precedent_id,
                            COUNT(DISTINCT pc.section_type) as cnt
                        FROM precedents p
                        LEFT JOIN precedent_contents pc ON p.id = pc.precedent_id
                        WHERE p.domain = 'criminal_law'
                        GROUP BY p.precedent_id
                    )
                    SELECT 
                        COUNT(*) FILTER (WHERE cnt >= 3) as complete,
                        COUNT(*) FILTER (WHERE cnt > 0 AND cnt < 3) as partial,
                        COUNT(*) FILTER (WHERE cnt = 0 OR cnt IS NULL) as none,
                        COUNT(*) as total
                    FROM section_counts
                """)
            )
            status_row = result.fetchone()
            if status_row[3] > 0:
                complete_pct = (status_row[0] * 100) // status_row[3]
                partial_pct = (status_row[1] * 100) // status_row[3]
                none_pct = (status_row[2] * 100) // status_row[3]
                print(f"  형법 판례 수집 상태:")
                print(f"    완전 수집 (3개 섹션): {status_row[0]:,}개 ({complete_pct}%)")
                print(f"    부분 수집 (1-2개 섹션): {status_row[1]:,}개 ({partial_pct}%)")
                print(f"    미수집 (0개 섹션): {status_row[2]:,}개 ({none_pct}%)")
            
            # 섹션 타입별 통계
            result = conn.execute(
                text("""
                    SELECT 
                        pc.section_type,
                        COUNT(*) as total
                    FROM precedent_contents pc
                    JOIN precedents p ON pc.precedent_id = p.id
                    WHERE p.domain = 'criminal_law'
                    GROUP BY pc.section_type
                    ORDER BY pc.section_type
                """)
            )
            print(f"  형법 섹션 타입별:")
            for row in result:
                print(f"    {row[0]}: {row[1]:,}개")
        
        print()
        
        # 수집 일자별 통계
        print("📅 수집 일자별 통계")
        print("-" * 80)
        
        result = conn.execute(
            text("""
                SELECT 
                    DATE(collected_at) as collection_date,
                    COUNT(*) as count
                FROM statutes
                GROUP BY DATE(collected_at)
                ORDER BY collection_date DESC
                LIMIT 10
            """)
        )
        print("  법령 수집 일자:")
        for row in result:
            print(f"    {row[0]}: {row[1]}개")
        
        result = conn.execute(
            text("""
                SELECT 
                    DATE(collected_at) as collection_date,
                    COUNT(*) as count
                FROM precedents
                GROUP BY DATE(collected_at)
                ORDER BY collection_date DESC
                LIMIT 10
            """)
        )
        print("  판례 수집 일자:")
        for row in result:
            print(f"    {row[0]}: {row[1]}개")
        
        print()
        
        # 최근 수집된 법령/판례
        print("🆕 최근 수집된 데이터")
        print("-" * 80)
        
        result = conn.execute(
            text("""
                SELECT 
                    law_name_kr,
                    domain,
                    collected_at
                FROM statutes
                ORDER BY collected_at DESC
                LIMIT 5
            """)
        )
        print("  최근 법령:")
        for row in result:
            print(f"    [{row[1]}] {row[0]} ({row[2]})")
        
        result = conn.execute(
            text("""
                SELECT 
                    case_name,
                    domain,
                    collected_at
                FROM precedents
                ORDER BY collected_at DESC
                LIMIT 5
            """)
        )
        print("  최근 판례:")
        for row in result:
            print(f"    [{row[1]}] {row[0]} ({row[2]})")
        
        print()
        print("=" * 80)


if __name__ == '__main__':
    main()

