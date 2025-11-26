#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precedent_contents 테이블의 중복 데이터 정리 스크립트
동일한 precedent_id와 section_type 조합의 중복 레코드를 제거합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
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
            load_dotenv(dotenv_path=str(root_env), override=True)
    except ImportError:
        pass

from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='precedent_contents 중복 데이터 정리')
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 삭제하지 않고 통계만 출력'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 PostgreSQL 환경변수가 필요합니다.")
        return
    
    # 데이터베이스 연결
    engine = create_engine(args.db, pool_pre_ping=True)
    
    with engine.connect() as conn:
        # 중복 통계 확인
        logger.info("중복 데이터 통계 확인 중...")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT (precedent_id, section_type)) as unique_combos,
                COUNT(*) - COUNT(DISTINCT (precedent_id, section_type)) as duplicates
            FROM precedent_contents
        """))
        stats = result.fetchone()
        total = stats[0]
        unique = stats[1]
        duplicates = stats[2]
        
        logger.info(f"전체 레코드: {total:,}개")
        logger.info(f"고유 조합: {unique:,}개")
        logger.info(f"중복 레코드: {duplicates:,}개 ({duplicates*100/total:.1f}%)")
        
        if duplicates == 0:
            logger.info("중복 데이터가 없습니다.")
            return
        
        if args.dry_run:
            logger.info("--dry-run 모드: 실제 삭제는 수행하지 않습니다.")
            logger.info("중복 레코드 예시:")
            result = conn.execute(text("""
                SELECT precedent_id, section_type, COUNT(*) as cnt
                FROM precedent_contents
                GROUP BY precedent_id, section_type
                HAVING COUNT(*) > 1
                ORDER BY cnt DESC
                LIMIT 10
            """))
            for row in result:
                logger.info(f"  precedent_id: {row[0]}, section_type: {row[1]}, 개수: {row[2]}")
            return
        
        # 중복 제거 (각 조합에서 가장 오래된 레코드만 유지)
        logger.info("중복 데이터 제거 중...")
        result = conn.execute(text("""
            DELETE FROM precedent_contents
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY precedent_id, section_type 
                               ORDER BY id ASC
                           ) as rn
                    FROM precedent_contents
                ) t
                WHERE rn > 1
            )
        """))
        deleted_count = result.rowcount
        
        conn.commit()
        
        logger.info(f"중복 데이터 제거 완료: {deleted_count:,}개 레코드 삭제")
        
        # 최종 통계 확인
        result = conn.execute(text("""
            SELECT COUNT(*) as total
            FROM precedent_contents
        """))
        final_total = result.fetchone()[0]
        logger.info(f"최종 레코드 수: {final_total:,}개")


if __name__ == '__main__':
    main()

