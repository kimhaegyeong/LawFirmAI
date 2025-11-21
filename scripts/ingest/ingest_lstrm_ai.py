# -*- coding: utf-8 -*-
"""
lstrmAI API 데이터 수집 스크립트
국가법령정보 공동활용 LAW OPEN DATA - 법령용어 검색 데이터 수집
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 환경 변수 로드 (import 전에 실행)
try:
    from utils.env_loader import ensure_env_loaded  # noqa: E402
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv  # noqa: E402
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

import os  # noqa: E402

from lawfirm_langgraph.core.data.connection_pool import get_connection_pool  # noqa: E402

from scripts.ingest.lstrm_ai_client import LstrmAIClient  # noqa: E402
from scripts.ingest.lstrm_ai_collector import LstrmAICollector  # noqa: E402

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _create_table(conn):
    """테이블 생성"""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS open_law_lstrm_ai_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- 검색 메타데이터
            search_keyword TEXT,
            search_page INTEGER,
            search_display INTEGER,
            homonym_yn TEXT,
            
            -- API 응답 원본 데이터 (JSON)
            raw_response_json TEXT NOT NULL,
            
            -- 개별 결과 항목
            term_id TEXT,
            term_name TEXT,
            homonym_exists TEXT,
            homonym_note TEXT,
            term_relation_link TEXT,
            article_relation_link TEXT,
            
            -- 수집 메타데이터
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            collection_method TEXT,
            api_request_url TEXT,
            
            -- 통계 정보
            total_count INTEGER,
            page_number INTEGER,
            num_of_rows INTEGER,
            
            UNIQUE(term_id, search_keyword, search_page)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_term_id 
        ON open_law_lstrm_ai_data(term_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_keyword 
        ON open_law_lstrm_ai_data(search_keyword)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_collected_at 
        ON open_law_lstrm_ai_data(collected_at)
    """)
    
    conn.commit()
    logger.info("테이블 생성 완료")


def _load_keywords(keywords_str: str = None, keyword_file: str = None) -> List[str]:
    """키워드 로드"""
    keywords = []
    
    if keywords_str:
        keywords.extend([k.strip() for k in keywords_str.split(',') if k.strip()])
    
    if keyword_file:
        file_path = Path(keyword_file)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords.extend([line.strip() for line in f if line.strip()])
        else:
            logger.warning(f"키워드 파일을 찾을 수 없습니다: {keyword_file}")
    
    return list(set(keywords))  # 중복 제거


def main():
    # 기본 DB 경로 (절대 경로)
    default_db_path = _PROJECT_ROOT / "data" / "lawfirm_v2.db"
    
    parser = argparse.ArgumentParser(description='lstrmAI API 데이터 수집')
    parser.add_argument(
        '--oc',
        default=os.getenv('LAW_OPEN_API_OC'),
        help='사용자 이메일 ID (환경변수: LAW_OPEN_API_OC)'
    )
    parser.add_argument('--keywords', help='검색 키워드 (쉼표 구분)')
    parser.add_argument('--keyword-file', help='키워드 파일 경로')
    parser.add_argument('--query', default='', help='검색 질의')
    parser.add_argument('--start-page', type=int, default=1, help='시작 페이지 번호')
    parser.add_argument('--max-pages', type=int, help='최대 페이지 수')
    parser.add_argument('--display', type=int, default=100, help='페이지당 결과 수')
    parser.add_argument(
        '--db-path',
        default=str(default_db_path),
        help=f'DB 경로 (기본값: {default_db_path})'
    )
    parser.add_argument('--rate-limit', type=float, default=0.5, help='요청 간 지연 (초)')
    
    args = parser.parse_args()
    
    # OC 필수 체크
    if not args.oc:
        logger.error("--oc 인자 또는 LAW_OPEN_API_OC 환경변수가 필요합니다.")
        return
    
    # 연결 풀 사용
    connection_pool = get_connection_pool(args.db_path)
    
    # 테이블 생성
    with connection_pool.get_connection_context() as conn:
        _create_table(conn)
    
    # API 클라이언트 및 수집기 생성
    client = LstrmAIClient(args.oc)
    client.rate_limit_delay = args.rate_limit
    
    collector = LstrmAICollector(client, args.db_path)
    
    # 수집 실행
    if args.keywords or args.keyword_file:
        keywords = _load_keywords(args.keywords, args.keyword_file)
        if not keywords:
            logger.error("키워드가 없습니다.")
            return
        
        logger.info(f"키워드 기반 수집 시작: {len(keywords)}개 키워드 (시작 페이지: {args.start_page})")
        total = collector.collect_by_keywords(keywords, args.max_pages, start_page=args.start_page)
    else:
        logger.info(f"전체 수집 시작: query='{args.query}' (시작 페이지: {args.start_page})")
        total = collector.collect_all_pages(args.query, args.max_pages, start_page=args.start_page)
    
    logger.info(f"수집 완료: 총 {total}건의 데이터를 수집했습니다.")


if __name__ == '__main__':
    main()

