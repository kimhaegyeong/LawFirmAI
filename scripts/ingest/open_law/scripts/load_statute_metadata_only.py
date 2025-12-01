#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령 목록만 빠르게 DB에 적재 (메타데이터만, 본문 수집 없음)
기존 StatuteCollector의 _save_statute_metadata() 메서드 활용
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from sqlalchemy import text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/load_statute_metadata_only.py -> 프로젝트 루트
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

from scripts.ingest.open_law.client import OpenLawClient
from scripts.ingest.open_law.collectors.statute_collector import StatuteCollector
from scripts.ingest.open_law.utils import build_database_url

# 로그 디렉토리 생성
Path('logs/open_law').mkdir(parents=True, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/open_law/load_statute_metadata.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_statute_metadata_only(
    input_file: str,
    database_url: str,
    domain: Optional[str] = None,
    batch_size: int = 100
):
    """
    법령 목록 JSON 파일에서 메타데이터만 추출하여 statutes 테이블에 저장
    (본문 수집 없이 빠르게 목록만 채움)
    
    Args:
        input_file: 법령 목록 JSON 파일 경로
        database_url: PostgreSQL 데이터베이스 URL
        domain: 법령 분야 (civil_law, criminal_law 등, None이면 각 법령의 domain 필드 사용)
        batch_size: 배치 커밋 크기
    """
    # API 클라이언트 생성 (OC는 필요 없지만 생성)
    oc = os.getenv('LAW_OPEN_API_OC', 'default')
    client = OpenLawClient(oc)
    collector = StatuteCollector(client, database_url)
    
    # 법령 목록 로드
    logger.info(f"법령 목록 파일 로드: {input_file}")
    statutes = collector.load_statute_list(input_file)
    logger.info(f"로드된 법령 수: {len(statutes)}개")
    
    # 세션 생성
    session = collector.Session()
    saved_count = 0
    updated_count = 0
    error_count = 0
    
    try:
        for i, statute in enumerate(statutes, 1):
            statute_name = statute.get('법령명한글') or statute.get('법령명', '알 수 없음')
            
            # 진행 상황 로깅
            if i % 100 == 0 or i == 1:
                logger.info(f"진행: {i}/{len(statutes)} - {statute_name}")
            
            try:
                # 기존 레코드 확인
                law_id = statute.get('법령ID') or statute.get('법령일련번호')
                if not law_id:
                    logger.warning(f"법령ID가 없습니다: {statute_name}")
                    error_count += 1
                    continue
                
                result = session.execute(
                    text("SELECT id FROM statutes WHERE law_id = :law_id"),
                    {"law_id": law_id}
                )
                existing = result.fetchone()
                
                # 도메인 결정: domain이 None이면 각 법령의 domain 필드 사용
                statute_domain = domain if domain else statute.get('domain', 'other')
                
                # 메타데이터만 저장 (기존 메서드 재사용)
                collector._save_statute_metadata(session, statute, domain=statute_domain)
                
                if existing:
                    updated_count += 1
                else:
                    saved_count += 1
                
                # 배치 커밋 (성능 최적화)
                if i % batch_size == 0:
                    session.commit()
                    logger.info(f"  → 배치 커밋 완료: 신규 {saved_count}개, 업데이트 {updated_count}개, 오류 {error_count}개")
            
            except Exception as e:
                logger.warning(f"법령 저장 실패 ({statute_name}): {e}")
                session.rollback()
                error_count += 1
                continue
        
        # 마지막 커밋
        session.commit()
        logger.info("법령 메타데이터 저장 완료:")
        logger.info(f"  - 신규 저장: {saved_count}개")
        logger.info(f"  - 업데이트: {updated_count}개")
        logger.info(f"  - 오류: {error_count}개")
        logger.info(f"  - 전체: {len(statutes)}개")
    
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='법령 목록만 빠르게 DB에 적재 (메타데이터만)')
    parser.add_argument(
        '--input',
        required=True,
        help='법령 목록 JSON 파일 경로'
    )
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--domain',
        default=None,
        choices=['civil_law', 'criminal_law', 'all'],
        help='법령 분야 (all이면 각 법령의 domain 필드 사용, 기본값: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 커밋 크기 (기본값: 100)'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # domain이 'all'이면 None으로 변환
    domain = None if args.domain == 'all' else args.domain
    
    # 법령 메타데이터만 저장
    load_statute_metadata_only(
        input_file=args.input,
        database_url=args.db,
        domain=domain,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

