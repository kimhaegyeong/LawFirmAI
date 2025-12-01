#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
형법 판례 수집 스크립트
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/collect_criminal_precedents.py -> 프로젝트 루트
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
        # 프로젝트 루트 .env 파일 우선 로드
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=True)
        # scripts/.env는 선택적으로 로드 (프로젝트 루트 .env가 없을 때만)
        if not root_env.exists():
            scripts_env = _PROJECT_ROOT / "scripts" / ".env"
            if scripts_env.exists():
                load_dotenv(dotenv_path=str(scripts_env), override=True)
    except ImportError:
        pass

from scripts.ingest.open_law.client import OpenLawClient
from scripts.ingest.open_law.collectors.precedent_collector import PrecedentCollector
from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/open_law/criminal_precedents_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='형법 판례 수집')
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
        help='판례 목록 JSON 파일 경로 (content 단계에서 사용)'
    )
    parser.add_argument(
        '--output',
        default='data/raw/open_law/criminal_precedents_list.json',
        help='판례 목록 저장 경로 (list 단계에서 사용)'
    )
    parser.add_argument(
        '--db',
        default=None,
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=100,
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
    
    # 데이터베이스 URL 설정 (환경변수에서 가져오기)
    if not args.db:
        args.db = build_database_url()
        if not args.db:
            logger.error("--db 인자 또는 PostgreSQL 환경변수(POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)가 필요합니다.")
            return
    
    # 로그 디렉토리 생성
    Path('logs/open_law').mkdir(parents=True, exist_ok=True)
    
    # API 클라이언트 생성
    client = OpenLawClient(args.oc)
    client.rate_limit_delay = args.rate_limit
    
    if args.phase == 'list':
        # 목록 수집
        logger.info("형법 판례 목록 수집 시작")
        
        collector = PrecedentCollector(client, args.db or "postgresql://localhost/lawfirmai")
        
        # 형법 관련 검색어
        search_queries = [
            {'query': '형사', 'jo': '형법'}
        ]
        all_precedents = []
        
        for search in search_queries:
            logger.info(f"검색어 '{search['query']}' 수집 시작")
            precedents = collector.collect_precedent_list(
                query=search['query'],
                jo=search.get('jo'),
                domain='criminal_law',
                max_pages=args.max_pages
            )
            all_precedents.extend(precedents)
            logger.info(f"검색어 '{search['query']}' 수집 완료: {len(precedents)}개")
        
        # 중복 제거 (판례일련번호 기준)
        seen_ids = set()
        unique_precedents = []
        for precedent in all_precedents:
            prec_id = precedent.get('판례일련번호') or precedent.get('prec id')
            if prec_id and prec_id not in seen_ids:
                seen_ids.add(prec_id)
                unique_precedents.append(precedent)
        
        # 목록 저장
        collector.save_precedent_list(unique_precedents, args.output)
        logger.info(f"형법 판례 목록 수집 완료: {len(unique_precedents)}개")
    
    elif args.phase == 'content':
        # 본문 수집
        if not args.input:
            logger.error("--input 인자가 필요합니다 (판례 목록 JSON 파일 경로)")
            return
        
        if not args.db:
            logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
            return
        
        logger.info("형법 판례 본문 수집 시작")
        
        collector = PrecedentCollector(client, args.db)
        
        # 처리 완료 파일 경로 생성
        input_path = Path(args.input)
        processed_path = input_path.parent / f"{input_path.stem}_processed.json"
        
        # 원본 파일에서 처리 완료된 판례 제외
        all_precedents = collector.load_precedent_list(args.input)
        
        # 처리 완료 파일이 있지만 실제로 저장되지 않았을 수 있으므로
        # 데이터베이스에서 실제 저장 여부를 확인
        from sqlalchemy import create_engine, text
        engine = create_engine(args.db, pool_pre_ping=True)
        with engine.connect() as conn:
            # 실제로 저장된 판례 ID 조회
            result = conn.execute(
                text("SELECT precedent_id FROM precedents WHERE domain = 'criminal_law'")
            )
            saved_precedent_ids = {row[0] for row in result}
        
        # 처리 완료 파일에 있지만 실제로 저장되지 않은 판례는 다시 처리
        processed_precedents = collector.filter_unprocessed_precedents(all_precedents, str(processed_path))
        processed_ids = {int(p.get('판례일련번호') or p.get('prec id') or 0) for p in processed_precedents if p.get('판례일련번호') or p.get('prec id')}
        
        # 실제로 저장되지 않은 판례 필터링
        precedents = []
        for p in all_precedents:
            prec_id = p.get('판례일련번호') or p.get('prec id')
            try:
                prec_id_int = int(prec_id) if prec_id else 0
            except (ValueError, TypeError):
                prec_id_int = 0
            
            # 처리 완료 파일에 있지만 실제로 저장되지 않은 경우 다시 처리
            if prec_id_int not in saved_precedent_ids:
                precedents.append(p)
        
        total_count = len(precedents)
        processed_count = len(all_precedents) - total_count
        
        if processed_count > 0:
            logger.info(f"이미 저장된 판례: {processed_count}개 (건너뛰기)")
        logger.info(f"처리할 판례: {total_count}개")
        
        # 이미 수집된 판례 및 섹션 정보 조회
        logger.info("이미 수집된 판례 및 섹션 확인 중...")
        from sqlalchemy import create_engine, text
        engine = create_engine(args.db, pool_pre_ping=True)
        with engine.connect() as conn:
            # 완전히 수집된 판례 (3개 섹션 모두)
            result = conn.execute(
                text("""
                    SELECT p.precedent_id, COUNT(DISTINCT pc.section_type) as section_count
                    FROM precedents p
                    INNER JOIN precedent_contents pc ON p.id = pc.precedent_id
                    WHERE p.domain = 'criminal_law'
                    GROUP BY p.precedent_id
                    HAVING COUNT(DISTINCT pc.section_type) >= 3
                """)
            )
            fully_collected_ids = {row[0] for row in result}
            
            # 부분 수집 판례의 이미 수집된 섹션 정보 (precedent_id -> set of section_types)
            result = conn.execute(
                text("""
                    SELECT p.precedent_id, pc.section_type
                    FROM precedents p
                    INNER JOIN precedent_contents pc ON p.id = pc.precedent_id
                    WHERE p.domain = 'criminal_law'
                """)
            )
            collected_sections_by_precedent = {}
            for row in result:
                precedent_id = row[0]
                section_type = row[1]
                if precedent_id not in collected_sections_by_precedent:
                    collected_sections_by_precedent[precedent_id] = set()
                collected_sections_by_precedent[precedent_id].add(section_type)
        
        if fully_collected_ids:
            logger.info(f"완전히 수집된 판례: {len(fully_collected_ids)}개 (API 호출 건너뛰기)")
        
        logger.info(f"총 {total_count}개 판례 중 {total_count - len(fully_collected_ids)}개 수집 시작")
        
        total_contents = 0
        skipped_count = 0
        processed_count_in_session = 0
        start_time = time.time()
        last_progress_time = start_time
        last_progress_count = 0
        progress_interval = 30  # 30초마다 진행 상황 출력
        progress_count_interval = 500  # 500개마다 진행 상황 출력
        
        # 처리 완료 대기 목록 (배치 저장용)
        pending_processed = []
        
        for i, precedent in enumerate(precedents, 1):
            precedent_id = precedent.get('판례일련번호') or precedent.get('prec id')
            try:
                precedent_id_int = int(precedent_id) if precedent_id else None
            except (ValueError, TypeError):
                precedent_id_int = None
            
            # 이미 완전히 수집된 판례는 API 호출 없이 건너뛰기
            if precedent_id_int and precedent_id_int in fully_collected_ids:
                skipped_count += 1
                # 처리 완료 파일에 추가 (배치 저장)
                pending_processed.append(precedent)
                processed_count_in_session += 1
                
                # 100개마다 또는 마지막에 일괄 저장
                if len(pending_processed) >= 100 or i == total_count:
                    for p in pending_processed:
                        collector.save_processed_precedent(str(processed_path), p)
                    pending_processed = []
                
                # 진행 상황 업데이트 (건너뛰는 경우에도 카운트)
                if i % 1000 == 0:
                    current_time = time.time()
                    total_elapsed = current_time - start_time
                    skip_rate = skipped_count / total_elapsed if total_elapsed > 0 else 0
                    logger.info(f"건너뛰기 진행: {i}/{total_count} ({i*100//total_count}%) | 건너뜀: {skipped_count}개 | 건너뛰기 속도: {skip_rate:.0f}개/초")
                continue
            
            # 부분 수집 판례의 경우, 이미 수집된 섹션 정보 전달
            existing_sections = collected_sections_by_precedent.get(precedent_id_int, set()) if precedent_id_int else set()
            
            # API 호출 여부와 관계없이 처리 완료로 표시 (0개 섹션이어도)
            api_called = False
            try:
                content_count = collector.collect_and_save_precedent_content(
                    precedent,
                    domain='criminal_law',
                    verbose=False,
                    existing_sections=existing_sections
                )
                total_contents += content_count
                api_called = True
            except Exception as e:
                logger.error(f"판례 수집 실패 (precedent_id={precedent_id}): {e}")
                # 에러가 발생해도 API 호출은 시도했으므로 처리 완료로 표시
                api_called = True
            
            # API 호출이 있었으면 처리 완료로 표시 (섹션 개수와 무관)
            if api_called:
                pending_processed.append(precedent)
                processed_count_in_session += 1
                
                # 100개마다 또는 마지막에 일괄 저장
                if len(pending_processed) >= 100 or i == total_count:
                    for p in pending_processed:
                        collector.save_processed_precedent(str(processed_path), p)
                    pending_processed = []
            
            # 진행 상황 주기적 출력 (30초마다 또는 500개마다 또는 마지막)
            current_time = time.time()
            if (current_time - last_progress_time >= progress_interval) or (i % progress_count_interval == 0) or (i == total_count):
                progress_pct = (i * 100) // total_count if total_count > 0 else 0
                
                # 전체 경과 시간과 속도 계산
                total_elapsed = current_time - start_time
                total_rate = i / total_elapsed if total_elapsed > 0 else 0
                
                # 마지막 간격 동안의 속도 계산
                interval_elapsed = current_time - last_progress_time
                interval_count = i - last_progress_count
                interval_rate = interval_count / interval_elapsed if interval_elapsed > 0 else 0
                
                # 예상 남은 시간 계산 (전체 평균 속도 기준)
                remaining = (total_count - i) / total_rate if total_rate > 0 else 0
                
                logger.info(
                    f"진행: {i}/{total_count} ({progress_pct}%) | "
                    f"섹션: {total_contents}개 | "
                    f"건너뜀: {skipped_count}개 | "
                    f"속도: {total_rate:.2f}개/초 (최근: {interval_rate:.2f}개/초) | "
                    f"예상 남은 시간: {remaining/60:.1f}분"
                )
                
                last_progress_time = current_time
                last_progress_count = i
        
        logger.info(f"형법 판례 본문 수집 완료: {total_count}개 판례, {total_contents}개 섹션")


if __name__ == '__main__':
    main()

