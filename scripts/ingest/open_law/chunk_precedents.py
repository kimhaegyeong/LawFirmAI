#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 데이터 청킹 스크립트
PostgreSQL precedent_contents 테이블의 데이터를 청킹하여 precedent_chunks 테이블에 저장
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/chunk_precedents.py -> parents[3] = 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils.env_loader 사용)
try:
    from utils.env_loader import ensure_env_loaded
    # ensure_env_loaded는 프로젝트 루트를 기대함
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    # 폴백: 직접 dotenv 사용
    try:
        from dotenv import load_dotenv
        # scripts/.env 파일 우선 로드
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        # 프로젝트 루트 .env 파일 로드
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=True)
    except ImportError:
        pass

# 판례 청킹 클래스 임포트
# scripts 디렉토리를 sys.path에 추가
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from scripts.utils.chunking.precedent_chunker import PrecedentChunker, PrecedentChunk
except ImportError:
    try:
        from utils.chunking.precedent_chunker import PrecedentChunker, PrecedentChunk
    except ImportError:
        # 직접 경로로 임포트
        import importlib.util
        chunker_path = _SCRIPTS_DIR / "utils" / "chunking" / "precedent_chunker.py"
        if chunker_path.exists():
            spec = importlib.util.spec_from_file_location("precedent_chunker", chunker_path)
            precedent_chunker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(precedent_chunker)
            PrecedentChunker = precedent_chunker.PrecedentChunker
            PrecedentChunk = precedent_chunker.PrecedentChunk
        else:
            raise ImportError(f"precedent_chunker.py를 찾을 수 없습니다: {chunker_path}")

# 데이터베이스 URL 빌드
try:
    from scripts.ingest.open_law.utils import build_database_url
except ImportError:
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


def get_precedent_contents(conn, limit: Optional[int] = None, offset: int = 0, domain: Optional[str] = None, last_id: Optional[int] = None):
    """
    precedent_contents 테이블에서 데이터 조회 (증분 처리: 이미 청킹된 데이터 제외)
    
    Args:
        conn: 데이터베이스 연결
        limit: 조회할 최대 개수
        offset: 오프셋 (사용하지 않음, last_id 사용)
        domain: 도메인 필터
        last_id: 마지막으로 처리한 id (커서 기반 페이지네이션)
    """
    query = """
        SELECT 
            pc.id,
            pc.precedent_id,
            pc.section_type,
            pc.section_content,
            pc.referenced_articles,
            pc.referenced_precedents,
            p.precedent_id as precedent_original_id,
            p.case_name,
            p.case_number,
            p.decision_date,
            p.court_name,
            p.domain
        FROM precedent_contents pc
        JOIN precedents p ON pc.precedent_id = p.id
        WHERE NOT EXISTS (
            SELECT 1 FROM precedent_chunks pch
            WHERE pch.precedent_content_id = pc.id
        )
    """
    
    params = {"limit": limit}
    
    if domain:
        query += " AND p.domain = :domain"
        params["domain"] = domain
    
    # 커서 기반 페이지네이션 사용 (offset 대신)
    if last_id:
        query += " AND pc.id > :last_id"
        params["last_id"] = last_id
    
    query += " ORDER BY pc.id LIMIT :limit"
    
    result = conn.execute(
        text(query),
        params
    )
    
    return result.fetchall()


def get_precedent_contents_count(conn, domain: Optional[str] = None):
    """청킹되지 않은 precedent_contents 개수 조회 (증분 처리)"""
    query = """
        SELECT COUNT(*)
        FROM precedent_contents pc
        JOIN precedents p ON pc.precedent_id = p.id
        WHERE NOT EXISTS (
            SELECT 1 FROM precedent_chunks pch
            WHERE pch.precedent_content_id = pc.id
        )
    """
    
    params = {}
    if domain:
        query += " AND p.domain = :domain"
        params["domain"] = domain
    
    result = conn.execute(text(query), params)
    return result.scalar()


def check_existing_chunks(conn, chunk_keys: List[tuple]) -> set:
    """
    이미 존재하는 청크 조합 확인 (배치 처리)
    
    Args:
        conn: 데이터베이스 연결
        chunk_keys: (precedent_content_id, chunk_index, embedding_version) 튜플 리스트
    
    Returns:
        이미 존재하는 청크 조합의 집합
    """
    if not chunk_keys:
        return set()
    
    # 배치 크기 제한 (PostgreSQL의 파라미터 제한 고려)
    batch_size = 1000
    existing = set()
    
    for i in range(0, len(chunk_keys), batch_size):
        batch_keys = chunk_keys[i:i + batch_size]
        
        # VALUES 절을 사용한 쿼리 (효율적)
        placeholders = []
        params = {}
        for j, (pc_id, chunk_idx, emb_ver) in enumerate(batch_keys):
            idx = i + j
            placeholders.append(f"(:pc_id_{idx}, :chunk_idx_{idx}, :emb_ver_{idx})")
            params[f"pc_id_{idx}"] = pc_id
            params[f"chunk_idx_{idx}"] = chunk_idx
            params[f"emb_ver_{idx}"] = emb_ver
        
        if not placeholders:
            continue
        
        query = text(f"""
            SELECT precedent_content_id, chunk_index, embedding_version
            FROM precedent_chunks
            WHERE (precedent_content_id, chunk_index, embedding_version) IN (
                VALUES {', '.join(placeholders)}
            )
        """)
        
        try:
            result = conn.execute(query, params)
            batch_existing = {(row[0], row[1], row[2]) for row in result}
            existing.update(batch_existing)
        except Exception as e:
            logger.warning(f"기존 청크 확인 실패 (배치 {i//batch_size + 1}): {e}")
            # 실패 시 해당 배치는 모두 삽입 시도
            continue
    
    return existing


def insert_chunks_batch(engine, chunks: List[PrecedentChunk]):
    """
    precedent_chunks 테이블에 청크 배치 삽입 (성능 최적화, 중복 확인 포함)
    각 청크를 별도 트랜잭션으로 처리하여 하나가 실패해도 나머지에 영향 없도록
    
    Args:
        engine: SQLAlchemy 엔진
        chunks: 삽입할 청크 리스트
    
    Returns:
        실제 삽입된 청크 개수
    """
    if not chunks:
        return 0
    
    # 배치 INSERT를 위한 VALUES 리스트 구성
    values_list = []
    chunk_keys = []  # 중복 확인용
    for chunk in chunks:
        try:
            metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
            precedent_content_id = chunk.metadata["precedent_content_id"]
            chunk_index = chunk.chunk_index
            embedding_version = 1  # 기본값
            
            chunk_keys.append((precedent_content_id, chunk_index, embedding_version))
            
            # SQL 인젝션 방지를 위해 파라미터화된 쿼리 사용
            values_list.append({
                "precedent_content_id": precedent_content_id,
                "chunk_index": chunk_index,
                "chunk_content": chunk.chunk_content,
                "chunk_length": chunk.chunk_length,
                "metadata": metadata_json
            })
        except Exception as e:
            logger.warning(f"청크 데이터 구성 실패: {e}")
            continue
    
    if not values_list:
        return 0
    
    # 중복 확인: 이미 존재하는 청크 필터링
    with engine.connect() as conn:
        existing_chunks = check_existing_chunks(conn, chunk_keys)
    
    # 중복이 아닌 청크만 필터링
    filtered_values = []
    for i, chunk_data in enumerate(values_list):
        chunk_key = chunk_keys[i]
        if chunk_key not in existing_chunks:
            filtered_values.append(chunk_data)
        else:
            logger.debug(
                f"중복 청크 건너뛰기: precedent_content_id={chunk_key[0]}, "
                f"chunk_index={chunk_key[1]}, embedding_version={chunk_key[2]}"
            )
    
    if not filtered_values:
        logger.debug(f"모든 청크가 이미 존재함: {len(values_list)}개")
        return 0
    
    # 각 청크를 별도 트랜잭션으로 처리
    # CAST 함수 사용하여 JSONB로 변환
    # embedding_version은 기본값 1 사용
    insert_query = text("""
        INSERT INTO precedent_chunks (
            precedent_content_id,
            chunk_index,
            chunk_content,
            chunk_length,
            metadata,
            embedding_version
        ) VALUES (
            :precedent_content_id,
            :chunk_index,
            :chunk_content,
            :chunk_length,
            CAST(:metadata AS JSONB),
            1
        )
    """)
    
    inserted_count = 0
    for chunk_data in filtered_values:
        try:
            # 각 청크를 별도 트랜잭션으로 삽입
            with engine.begin() as conn:
                result = conn.execute(
                    insert_query,
                    chunk_data
                )
                if result.rowcount > 0:
                    inserted_count += 1
        except Exception as e:
            # 개별 청크 삽입 실패 시 로깅하고 계속 진행
            logger.warning(
                f"개별 청크 삽입 실패 (precedent_content_id={chunk_data.get('precedent_content_id')}, "
                f"chunk_index={chunk_data.get('chunk_index')}): {e}"
            )
            continue
    
    if inserted_count > 0:
        logger.debug(f"배치 삽입 완료: {inserted_count}/{len(filtered_values)}개 삽입됨 (전체 {len(values_list)}개 중)")
    return inserted_count
    


def chunk_precedents(
    db_url: str,
    batch_size: int = 100,
    limit: Optional[int] = None,
    domain: Optional[str] = None
):
    """
    판례 데이터 청킹 실행 (증분 처리 지원, 성능 최적화)
    
    Args:
        db_url: 데이터베이스 URL
        batch_size: 배치 크기
        limit: 최대 처리 개수 (None이면 전체)
        domain: 도메인 필터 (civil_law, criminal_law 등)
    """
    logger.info("판례 데이터 청킹 시작 (증분 처리 모드, 성능 최적화)")
    if domain:
        logger.info(f"도메인 필터: {domain}")
    
    # 데이터베이스 연결 (연결 풀 최적화)
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        echo=False
    )
    
    chunker = PrecedentChunker()
    
    total_processed = 0
    total_chunks = 0
    last_id = None  # 커서 기반 페이지네이션
    commit_interval = 50  # 50개마다 커밋 (트랜잭션 크기 최적화)
    
    # 전체 개수 조회를 위한 연결
    with engine.connect() as conn:
        total_count = get_precedent_contents_count(conn, domain=domain)
        logger.info(f"청킹 대상: {total_count}개 (이미 청킹된 데이터 제외)")
        
        if total_count == 0:
            logger.info("청킹할 데이터가 없습니다. (모든 데이터가 이미 청킹됨)")
            return
    
    # 배치 처리 (각 배치마다 별도 트랜잭션 사용, 커서 기반 페이지네이션)
    while True:
        # 데이터 조회를 위한 연결
        with engine.connect() as conn:
            rows = get_precedent_contents(conn, limit=batch_size, domain=domain, last_id=last_id)
            
            if not rows:
                break
            
            # 배치 단위로 청크 수집
            batch_chunks = []
            batch_processed = 0
            
            # 각 행 처리
            for row in rows:
                try:
                    # 메타데이터 구성
                    precedent_metadata = {
                        "precedent_id": row.precedent_original_id,
                        "case_name": row.case_name,
                        "case_number": row.case_number,
                        "decision_date": row.decision_date,
                        "court_name": row.court_name,
                        "domain": row.domain
                    }
                    
                    # 청킹 수행
                    chunks = chunker.chunk_precedent_content(
                        precedent_content_id=row.id,
                        section_type=row.section_type,
                        section_content=row.section_content,
                        precedent_metadata=precedent_metadata,
                        referenced_articles=row.referenced_articles,
                        referenced_precedents=row.referenced_precedents
                    )
                    
                    # 청크를 배치에 추가
                    batch_chunks.extend(chunks)
                    batch_processed += 1
                    total_processed += 1
                    
                    # 메모리 최적화: 큰 변수 명시적 삭제
                    del chunks
                    del precedent_metadata
                
                except Exception as e:
                    logger.error(f"행 처리 실패 (id={row.id}): {e}")
                    # 예외 발생 시에도 total_processed는 증가하지 않음 (continue로 건너뜀)
                    continue
                
                # 주기적으로 배치 INSERT 및 커밋
                if batch_processed >= commit_interval or total_processed == total_count:
                    if batch_chunks:
                        try:
                            # 별도 연결로 배치 삽입 (각 청크를 개별 트랜잭션으로 처리)
                            inserted = insert_chunks_batch(engine, batch_chunks)
                            total_chunks += inserted
                            logger.debug(f"배치 삽입 완료: {inserted}개 삽입됨, 총 {total_chunks}개")
                            
                            # 메모리 최적화
                            del batch_chunks
                            batch_chunks = []
                            batch_processed = 0
                            gc.collect()
                        except Exception as e:
                            logger.error(f"배치 삽입 실패: {e}", exc_info=True)
                            # 개별 삽입으로 폴백
                            fallback_count = 0
                            for chunk in batch_chunks:
                                try:
                                    inserted = insert_chunks_batch(engine, [chunk])
                                    if inserted > 0:
                                        fallback_count += inserted
                                    total_chunks += inserted
                                except Exception as e2:
                                    logger.warning(f"개별 청크 삽입 실패: {e2}")
                            logger.info(f"폴백 삽입 완료: {fallback_count}개 삽입됨")
                            del batch_chunks
                            batch_chunks = []
                            batch_processed = 0
                            gc.collect()
                
                # 진행 상황 로깅
                if total_processed % 100 == 0 or total_processed == total_count:
                    logger.info(
                        f"진행 상황: {total_processed}/{total_count} "
                        f"처리됨 ({total_processed*100//total_count if total_count > 0 else 0}%), "
                        f"{total_chunks}개 청크 생성"
                    )
            
            # 남은 청크 처리
            if batch_chunks:
                try:
                    logger.debug(f"최종 배치 삽입 시작: {len(batch_chunks)}개 청크")
                    inserted = insert_chunks_batch(engine, batch_chunks)
                    total_chunks += inserted
                    logger.debug(f"최종 배치 삽입 완료: {inserted}개 삽입됨")
                    del batch_chunks
                    gc.collect()
                except Exception as e:
                    logger.error(f"최종 배치 삽입 실패: {e}", exc_info=True)
                    del batch_chunks
                    gc.collect()
            
            # 마지막 처리한 id 업데이트 (커서 기반 페이지네이션)
            if rows:
                last_id = rows[-1].id
            else:
                break
            
            # 메모리 최적화: rows 명시적 삭제
            del rows
            gc.collect()
        
        # Limit 체크
        if limit and total_processed >= limit:
            break
    
    logger.info(
        f"청킹 완료: {total_processed}개 처리, {total_chunks}개 청크 생성"
    )
    
    # 최종 메모리 정리
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='판례 데이터 청킹')
    parser.add_argument(
        '--db',
        default=None,
        help='PostgreSQL 데이터베이스 URL (기본값: 환경변수에서 자동 로드)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기 (기본값: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='처리할 최대 개수 (기본값: 전체)'
    )
    parser.add_argument(
        '--domain',
        choices=['civil_law', 'criminal_law'],
        default=None,
        help='도메인 필터 (기본값: 전체)'
    )
    
    args = parser.parse_args()
    
    # 데이터베이스 URL 확인 (우선순위: --db 인자 > build_database_url())
    # build_database_url()은 PostgreSQL URL만 반환해야 함 (SQLite URL 무시)
    db_url = args.db
    if not db_url:
        db_url = build_database_url()
        
        # build_database_url()이 SQLite URL을 반환한 경우, None으로 처리
        if db_url and not db_url.startswith('postgresql'):
            logger.warning("build_database_url()이 SQLite URL을 반환했습니다. PostgreSQL 환경 변수를 확인합니다.")
            # PostgreSQL 환경 변수 직접 확인
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB')
            user = os.getenv('POSTGRES_USER')
            password = os.getenv('POSTGRES_PASSWORD')
            if db and user and password:
                from urllib.parse import quote_plus
                encoded_password = quote_plus(password)
                db_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
                logger.info(f"PostgreSQL URL 구성: postgresql://{user}:***@{host}:{port}/{db}")
            else:
                db_url = None
    
    if not db_url:
        logger.error("--db 인자 또는 PostgreSQL 환경변수가 필요합니다.")
        logger.error("PostgreSQL 환경 변수: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        return
    
    try:
        chunk_precedents(
            db_url=db_url,
            batch_size=args.batch_size,
            limit=args.limit,
            domain=args.domain
        )
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

