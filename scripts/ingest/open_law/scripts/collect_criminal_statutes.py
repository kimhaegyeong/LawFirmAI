#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
형법 현행법령 수집 스크립트
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/collect_criminal_statutes.py -> 프로젝트 루트
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/open_law/criminal_statutes_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='형법 현행법령 수집')
    parser.add_argument(
        '--oc',
        default=os.getenv('LAW_OPEN_API_OC'),
        help='사용자 이메일 ID (환경변수: LAW_OPEN_API_OC)'
    )
    parser.add_argument(
        '--phase',
        choices=['list', 'content'],
        required=True,
        help='수집 단계: list (목록 수집), content (본문 수집)'
    )
    parser.add_argument(
        '--input',
        help='법령 목록 JSON 파일 경로 (content 단계에서 사용)'
    )
    parser.add_argument(
        '--output',
        default='data/raw/open_law/criminal_statutes_list.json',
        help='법령 목록 저장 경로 (list 단계에서 사용)'
    )
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        help='최대 페이지 수 (list 단계에서 사용)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='요청 간 지연 (초)'
    )
    
    args = parser.parse_args()
    
    # OC 필수 체크
    if not args.oc:
        logger.error("--oc 인자 또는 LAW_OPEN_API_OC 환경변수가 필요합니다.")
        return
    
    # 로그 디렉토리 생성
    Path('logs/open_law').mkdir(parents=True, exist_ok=True)
    
    # API 클라이언트 생성
    client = OpenLawClient(args.oc)
    client.rate_limit_delay = args.rate_limit
    
    if args.phase == 'list':
        # 목록 수집
        logger.info("형법 법령 목록 수집 시작")
        
        collector = StatuteCollector(client, args.db or "postgresql://localhost/lawfirmai")
        
        # 형법 관련 검색어
        queries = ['형법', '형사소송법']
        all_statutes = []
        
        for query in queries:
            logger.info(f"검색어 '{query}' 수집 시작")
            statutes = collector.collect_statute_list(
                query=query,
                domain='criminal_law',
                max_pages=args.max_pages
            )
            all_statutes.extend(statutes)
            logger.info(f"검색어 '{query}' 수집 완료: {len(statutes)}개")
        
        # 중복 제거 (법령ID 기준)
        seen_ids = set()
        unique_statutes = []
        for statute in all_statutes:
            law_id = statute.get('법령ID') or statute.get('법령일련번호')
            if law_id and law_id not in seen_ids:
                seen_ids.add(law_id)
                unique_statutes.append(statute)
        
        # 목록 저장
        collector.save_statute_list(unique_statutes, args.output)
        logger.info(f"형법 법령 목록 수집 완료: {len(unique_statutes)}개")
    
    elif args.phase == 'content':
        # 본문 수집
        if not args.input:
            logger.error("--input 인자가 필요합니다 (법령 목록 JSON 파일 경로)")
            return
        
        if not args.db:
            logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
            return
        
        logger.info("형법 법령 본문 수집 시작")
        
        collector = StatuteCollector(client, args.db)
        statutes = collector.load_statute_list(args.input)
        
        total_articles = 0
        for i, statute in enumerate(statutes, 1):
            logger.info(f"법령 수집 진행: {i}/{len(statutes)} - {statute.get('법령명한글')}")
            article_count = collector.collect_and_save_statute_content(
                statute,
                domain='criminal_law'
            )
            total_articles += article_count
        
        logger.info(f"형법 법령 본문 수집 완료: {len(statutes)}개 법령, {total_articles}개 조문")


if __name__ == '__main__':
    main()

