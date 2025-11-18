#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기존 데이터 재임베딩 스크립트

기존 text_chunks 데이터를 새로운 청킹 전략으로 재청킹하고 재임베딩합니다.
"""
import argparse
import sqlite3
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embeddings import SentenceEmbedder
from scripts.utils.chunking.factory import ChunkingFactory
from scripts.utils.embedding_version_manager import EmbeddingVersionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def restore_original_document(
    conn: sqlite3.Connection,
    source_type: str,
    source_id: int
) -> Optional[str]:
    """원본 문서 텍스트 복원"""
    try:
        conn.row_factory = sqlite3.Row
        
        if source_type == "statute_article":
            cursor = conn.execute(
                "SELECT text FROM statute_articles WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row:
                return row['text']
        
        elif source_type == "case_paragraph":
            # case_paragraph는 case_paragraphs 테이블에서 복원
            # source_id는 case_id
            cursor = conn.execute(
                "SELECT text FROM case_paragraphs WHERE case_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
            # case_paragraphs가 없으면 cases 테이블에서 조회 시도
            cursor = conn.execute(
                "SELECT full_text FROM cases WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row and row['full_text']:
                return row['full_text']
        
        elif source_type == "decision_paragraph":
            # decision_paragraph는 decision_paragraphs 테이블에서 복원
            cursor = conn.execute(
                "SELECT text FROM decision_paragraphs WHERE decision_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
            # decision_paragraphs가 없으면 decisions 테이블에서 조회 시도
            cursor = conn.execute(
                "SELECT text FROM decisions WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row and row['text']:
                return row['text']
        
        elif source_type == "interpretation_paragraph":
            # interpretation_paragraph는 interpretation_paragraphs 테이블에서 복원
            cursor = conn.execute(
                "SELECT text FROM interpretation_paragraphs WHERE interpretation_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
            # interpretation_paragraphs가 없으면 interpretations 테이블에서 조회 시도
            cursor = conn.execute(
                "SELECT text FROM interpretations WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row and row['text']:
                return row['text']
        
        # 복원 실패 시 text_chunks에서 조합
        cursor = conn.execute(
            "SELECT text FROM text_chunks WHERE source_type = ? AND source_id = ? AND text IS NOT NULL AND text != '' ORDER BY chunk_index",
            (source_type, source_id)
        )
        rows = cursor.fetchall()
        if rows:
            # 모든 청크를 조합
            texts = [row['text'] for row in rows if row['text']]
            if texts:
                return "\n".join(texts)
        
        return None
        
    except Exception as e:
        logger.error(f"Error restoring document ({source_type}, {source_id}): {e}")
        return None


def get_unique_documents(conn: sqlite3.Connection, source_type: Optional[str] = None) -> List[Tuple[str, int]]:
    """고유한 문서 목록 조회"""
    try:
        conn.row_factory = sqlite3.Row
        
        if source_type:
            query = """
                SELECT DISTINCT source_type, source_id 
                FROM text_chunks 
                WHERE source_type = ?
                ORDER BY source_type, source_id
            """
            cursor = conn.execute(query, (source_type,))
        else:
            query = """
                SELECT DISTINCT source_type, source_id 
                FROM text_chunks 
                ORDER BY source_type, source_id
            """
            cursor = conn.execute(query)
        
        return [(row['source_type'], row['source_id']) for row in cursor.fetchall()]
        
    except Exception as e:
        logger.error(f"Error getting unique documents: {e}")
        return []


def re_embed_document(
    conn: sqlite3.Connection,
    source_type: str,
    source_id: int,
    embedder: SentenceEmbedder,
    version_manager: EmbeddingVersionManager,
    version_id: int,
    chunking_strategy: str = "standard",
    query_type: Optional[str] = None,
    batch_size: int = 128,
    skip_if_exists: bool = True
) -> Tuple[int, int]:
    """
    단일 문서 재임베딩
    
    Args:
        version_manager: 재사용할 EmbeddingVersionManager 인스턴스
        version_id: 재사용할 버전 ID
        skip_if_exists: 이미 재임베딩된 문서를 건너뛸지 여부
    
    Returns:
        (deleted_chunks, inserted_chunks) 튜플
    """
    try:
        # 이미 재임베딩된 문서인지 확인 (성능 최적화)
        if skip_if_exists:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM text_chunks WHERE source_type = ? AND source_id = ? AND chunking_strategy = ? AND embedding_version_id = ?",
                (source_type, source_id, chunking_strategy, version_id)
            )
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logger.debug(f"Skipping already re-embedded document: {source_type}, {source_id}")
                return (0, existing_count)
        
        # 원본 문서 복원
        original_text = restore_original_document(conn, source_type, source_id)
        if not original_text:
            logger.warning(f"Could not restore original document: {source_type}, {source_id}")
            return (0, 0)
        
        # 기존 청크 및 임베딩 삭제
        deleted_chunks, deleted_embeddings = version_manager.delete_chunks_by_version(
            source_type=source_type,
            source_id=source_id
        )
        
        # 청킹 전략 사용
        strategy = ChunkingFactory.create_strategy(
            strategy_name=chunking_strategy,
            query_type=query_type
        )
        
        # 원본 문서를 청킹 가능한 형식으로 변환
        if source_type == "statute_article":
            # statute_article은 sentences 리스트로 변환
            sentences = [s.strip() for s in original_text.split('\n') if s.strip()]
            content = sentences
        else:
            # case_paragraph, decision_paragraph, interpretation_paragraph는 paragraphs 리스트
            paragraphs = [p.strip() for p in original_text.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in original_text.split('\n') if p.strip()]
            content = paragraphs
        
        # 청킹 수행
        chunk_results = strategy.chunk(
            content=content,
            source_type=source_type,
            source_id=source_id
        )
        
        if not chunk_results:
            logger.warning(f"No chunks generated for {source_type}, {source_id}")
            return (deleted_chunks, 0)
        
        # 기존 chunk_index 확인
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM text_chunks WHERE source_type=? AND source_id=?",
            (source_type, source_id)
        ).fetchone()[0]
        next_chunk_index = int(max_idx) + 1
        
        # 청크 및 임베딩 삽입
        chunk_ids = []
        texts_to_embed = []
        
        # 소스 타입별 메타데이터 조회 (모든 청크에 공통으로 사용)
        source_metadata = None
        try:
            import json
            if source_type == "case_paragraph":
                cursor_meta = conn.execute("""
                    SELECT c.doc_id, c.casenames, c.court
                    FROM cases c
                    WHERE c.id = ?
                """, (source_id,))
                row = cursor_meta.fetchone()
                if row:
                    source_metadata = {
                        'doc_id': row['doc_id'],
                        'casenames': row['casenames'],
                        'court': row['court']
                    }
            elif source_type == "decision_paragraph":
                cursor_meta = conn.execute("""
                    SELECT d.org, d.doc_id
                    FROM decisions d
                    WHERE d.id = ?
                """, (source_id,))
                row = cursor_meta.fetchone()
                if row:
                    source_metadata = {
                        'org': row['org'],
                        'doc_id': row['doc_id']
                    }
            elif source_type == "statute_article":
                cursor_meta = conn.execute("""
                    SELECT s.name as statute_name, sa.article_no
                    FROM statute_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.id = ?
                """, (source_id,))
                row = cursor_meta.fetchone()
                if row:
                    source_metadata = {
                        'statute_name': row['statute_name'],
                        'law_name': row['statute_name'],
                        'article_no': row['article_no'],
                        'article_number': row['article_no']
                    }
            elif source_type == "interpretation_paragraph":
                cursor_meta = conn.execute("""
                    SELECT i.org, i.doc_id, i.title
                    FROM interpretations i
                    WHERE i.id = ?
                """, (source_id,))
                row = cursor_meta.fetchone()
                if row:
                    source_metadata = {
                        'org': row['org'],
                        'doc_id': row['doc_id'],
                        'title': row['title']
                    }
        except Exception as e:
            logger.debug(f"Failed to get source metadata for {source_type} {source_id}: {e}")
        
        # 메타데이터 JSON 생성
        meta_json = None
        if source_metadata:
            try:
                meta_json = json.dumps(source_metadata, ensure_ascii=False)
            except Exception as e:
                logger.debug(f"Failed to serialize metadata for {source_type} {source_id}: {e}")
        
        for i, chunk_result in enumerate(chunk_results):
            chunk_idx = next_chunk_index + i
            metadata = chunk_result.metadata
            
            cursor = conn.execute(
                """INSERT INTO text_chunks(
                    source_type, source_id, level, chunk_index,
                    start_char, end_char, overlap_chars, text, token_count, meta,
                    chunking_strategy, chunk_size_category, chunk_group_id,
                    query_type, original_document_id, embedding_version_id
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    source_type,
                    source_id,
                    metadata.get("level", "paragraph"),
                    chunk_idx,
                    None, None, None,
                    chunk_result.text,
                    None,
                    meta_json,  # 메타데이터 JSON 저장
                    metadata.get("chunking_strategy"),
                    metadata.get("chunk_size_category"),
                    metadata.get("chunk_group_id"),
                    metadata.get("query_type"),
                    metadata.get("original_document_id"),
                    version_id
                )
            )
            
            chunk_id = cursor.lastrowid
            chunk_ids.append(chunk_id)
            texts_to_embed.append(chunk_result.text)
        
        # 임베딩 생성 (배치 처리)
        if texts_to_embed:
            vecs = embedder.encode(texts_to_embed, batch_size=batch_size)
            dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
            model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            
            # 임베딩 삽입
            embedding_data = [
                (chunk_id, model_name, dim, vec.tobytes(), version_id)
                for chunk_id, vec in zip(chunk_ids, vecs)
            ]
            
            conn.executemany(
                "INSERT INTO embeddings(chunk_id, model, dim, vector, version_id) VALUES(?,?,?,?,?)",
                embedding_data
            )
            
            conn.commit()
            logger.info(f"Re-embedded {source_type} {source_id}: {len(chunk_ids)} chunks")
            return (deleted_chunks, len(chunk_ids))
        
        return (deleted_chunks, 0)
        
    except Exception as e:
        logger.error(f"Error re-embedding document ({source_type}, {source_id}): {e}", exc_info=True)
        conn.rollback()
        return (0, 0)


def main():
    parser = argparse.ArgumentParser(description='기존 데이터 재임베딩')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--source-type', help='특정 source_type만 처리 (예: statute_article, case_paragraph)')
    parser.add_argument('--chunking-strategy', default='standard', choices=['standard', 'dynamic', 'hybrid'],
                       help='사용할 청킹 전략')
    parser.add_argument('--query-type', help='동적 청킹을 위한 쿼리 타입')
    parser.add_argument('--batch-size', type=int, default=128, help='임베딩 배치 크기')
    parser.add_argument('--limit', type=int, help='처리할 문서 수 제한 (테스트용)')
    parser.add_argument('--dry-run', action='store_true', help='실제 변경 없이 테스트만 수행')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN 모드: 실제 변경 없이 테스트만 수행합니다.")
    
    # 데이터베이스 연결
    conn = sqlite3.connect(args.db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    
    # 임베더 초기화
    logger.info("임베더 초기화 중...")
    import os
    model_name = os.getenv("EMBEDDING_MODEL")
    if model_name:
        logger.info(f"Using embedding model from environment: {model_name}")
        embedder = SentenceEmbedder(model_name)
    else:
        logger.info("Using default embedding model")
        embedder = SentenceEmbedder()
    
    # 고유한 문서 목록 조회
    logger.info("고유한 문서 목록 조회 중...")
    documents = get_unique_documents(conn, args.source_type)
    
    if args.limit:
        documents = documents[:args.limit]
        logger.info(f"제한 적용: {args.limit}개 문서만 처리")
    
    logger.info(f"총 {len(documents)}개 문서를 재임베딩합니다.")
    logger.info(f"청킹 전략: {args.chunking_strategy}")
    
    if not args.dry_run:
        # 버전 관리자 초기화 (한 번만 수행)
        db_path = args.db
        version_manager = EmbeddingVersionManager(db_path)
        
        # 활성 버전 조회 또는 생성 (한 번만 수행)
        model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
        # 모델명을 포함한 고유한 버전명 생성
        model_short = model_name.split('/')[-1].replace('-', '_')[:20]  # 모델명에서 짧은 식별자 추출
        version_name = f"v2.0.0-{args.chunking_strategy}-{model_short}"
        
        # 기존 버전 확인
        existing_version = version_manager.get_version_by_name(version_name)
        if existing_version:
            version_id = existing_version['id']
            logger.info(f"기존 버전 사용: {version_name} (ID: {version_id})")
        else:
            # 새 버전 생성
            try:
                version_id = version_manager.register_version(
                    version_name=version_name,
                    chunking_strategy=args.chunking_strategy,
                    model_name=model_name,
                    description=f"{args.chunking_strategy} 청킹 전략, 모델: {model_name}",
                    set_active=True
                )
                logger.info(f"새 버전 생성: {version_name} (ID: {version_id})")
            except Exception as e:
                # 버전 생성 실패 시 기존 활성 버전 사용
                logger.warning(f"버전 생성 실패: {e}, 기존 활성 버전 사용 시도")
                active_version = version_manager.get_active_version(args.chunking_strategy)
                if active_version:
                    version_id = active_version['id']
                    logger.info(f"기존 활성 버전 사용: {active_version['version_name']} (ID: {version_id})")
                else:
                    raise
        
        logger.info(f"사용할 버전 ID: {version_id}")
        
        # 진행 상황 표시
        total_deleted = 0
        total_inserted = 0
        skipped = 0
        
        for source_type, source_id in tqdm(documents, desc="재임베딩 진행"):
            deleted, inserted = re_embed_document(
                conn=conn,
                source_type=source_type,
                source_id=source_id,
                embedder=embedder,
                version_manager=version_manager,
                version_id=version_id,
                chunking_strategy=args.chunking_strategy,
                query_type=args.query_type,
                batch_size=args.batch_size,
                skip_if_exists=True
            )
            if deleted == 0 and inserted > 0:
                skipped += 1
            total_deleted += deleted
            total_inserted += inserted
        
        logger.info(f"재임베딩 완료: {total_deleted}개 청크 삭제, {total_inserted}개 청크 삽입, {skipped}개 문서 건너뜀")
    else:
        logger.info(f"DRY RUN: {len(documents)}개 문서를 처리할 예정입니다.")
        # 샘플 문서 하나만 테스트
        if documents:
            sample = documents[0]
            logger.info(f"샘플 문서 테스트: {sample[0]}, {sample[1]}")
            original_text = restore_original_document(conn, sample[0], sample[1])
            if original_text:
                logger.info(f"원본 문서 복원 성공: {len(original_text)}자")
            else:
                logger.warning("원본 문서 복원 실패")
    
    conn.close()
    logger.info("작업 완료!")


if __name__ == '__main__':
    main()

