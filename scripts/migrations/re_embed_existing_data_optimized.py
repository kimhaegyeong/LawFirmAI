#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기존 데이터 재임베딩 스크립트 (최적화 버전)

최적화 사항:
1. 여러 문서의 청크를 모아서 배치 임베딩 생성 (3-4배 성능 향상)
2. 배치 커밋 (여러 문서를 한 트랜잭션으로 처리)
3. GPU 사용 확인 및 최적화
4. 인덱스 최적화
"""
import argparse
import sqlite3
import sys
import logging
import gc
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

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


@dataclass
class ChunkData:
    """청크 데이터 클래스"""
    source_type: str
    source_id: int
    chunk_result: Any
    chunk_index_offset: int


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
            cursor = conn.execute(
                "SELECT text FROM case_paragraphs WHERE case_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
            cursor = conn.execute(
                "SELECT full_text FROM cases WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row and row['full_text']:
                return row['full_text']
        
        elif source_type == "decision_paragraph":
            cursor = conn.execute(
                "SELECT text FROM decision_paragraphs WHERE decision_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
            cursor = conn.execute(
                "SELECT text FROM decisions WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row and row['text']:
                return row['text']
        
        elif source_type == "interpretation_paragraph":
            cursor = conn.execute(
                "SELECT text FROM interpretation_paragraphs WHERE interpretation_id = ? ORDER BY para_index",
                (source_id,)
            )
            rows = cursor.fetchall()
            if rows:
                texts = [row['text'] for row in rows if row['text']]
                if texts:
                    return "\n".join(texts)
            
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
            texts = [row['text'] for row in rows if row['text']]
            if texts:
                return "\n".join(texts)
        
        return None
        
    except Exception as e:
        logger.error(f"Error restoring document ({source_type}, {source_id}): {e}")
        return None


def get_unique_documents(conn: sqlite3.Connection, source_type: Optional[str] = None) -> List[Tuple[str, int]]:
    """고유한 문서 목록 조회 (원본 테이블에서)"""
    try:
        conn.row_factory = sqlite3.Row
        documents = []
        
        # 원본 테이블에서 문서 가져오기
        if not source_type or source_type == "statute_article":
            cursor = conn.execute("SELECT id FROM statute_articles ORDER BY id")
            documents.extend([("statute_article", row['id']) for row in cursor.fetchall()])
        
        if not source_type or source_type == "case_paragraph":
            # case_paragraph는 case_id 기준으로 가져오기
            cursor = conn.execute("SELECT DISTINCT case_id FROM case_paragraphs ORDER BY case_id")
            case_ids = [row['case_id'] for row in cursor.fetchall()]
            # case_paragraphs에 없는 경우 cases 테이블에서도 확인
            if case_ids:
                placeholders = ','.join(['?'] * len(case_ids))
                cursor = conn.execute(
                    f"SELECT id FROM cases WHERE id NOT IN ({placeholders}) ORDER BY id",
                    case_ids
                )
                case_ids.extend([row['id'] for row in cursor.fetchall()])
            else:
                cursor = conn.execute("SELECT id FROM cases ORDER BY id")
                case_ids = [row['id'] for row in cursor.fetchall()]
            documents.extend([("case_paragraph", case_id) for case_id in case_ids])
        
        if not source_type or source_type == "decision_paragraph":
            # decision_paragraph는 decision_id 기준으로 가져오기
            cursor = conn.execute("SELECT DISTINCT decision_id FROM decision_paragraphs ORDER BY decision_id")
            decision_ids = [row['decision_id'] for row in cursor.fetchall()]
            # decision_paragraphs에 없는 경우 decisions 테이블에서도 확인
            if decision_ids:
                placeholders = ','.join(['?'] * len(decision_ids))
                cursor = conn.execute(
                    f"SELECT id FROM decisions WHERE id NOT IN ({placeholders}) ORDER BY id",
                    decision_ids
                )
                decision_ids.extend([row['id'] for row in cursor.fetchall()])
            else:
                cursor = conn.execute("SELECT id FROM decisions ORDER BY id")
                decision_ids = [row['id'] for row in cursor.fetchall()]
            documents.extend([("decision_paragraph", decision_id) for decision_id in decision_ids])
        
        if not source_type or source_type == "interpretation_paragraph":
            # interpretation_paragraph는 interpretation_id 기준으로 가져오기
            cursor = conn.execute("SELECT DISTINCT interpretation_id FROM interpretation_paragraphs ORDER BY interpretation_id")
            interpretation_ids = [row['interpretation_id'] for row in cursor.fetchall()]
            # interpretation_paragraphs에 없는 경우 interpretations 테이블에서도 확인
            if interpretation_ids:
                placeholders = ','.join(['?'] * len(interpretation_ids))
                cursor = conn.execute(
                    f"SELECT id FROM interpretations WHERE id NOT IN ({placeholders}) ORDER BY id",
                    interpretation_ids
                )
                interpretation_ids.extend([row['id'] for row in cursor.fetchall()])
            else:
                cursor = conn.execute("SELECT id FROM interpretations ORDER BY id")
                interpretation_ids = [row['id'] for row in cursor.fetchall()]
            documents.extend([("interpretation_paragraph", interpretation_id) for interpretation_id in interpretation_ids])
        
        return documents
        
    except Exception as e:
        logger.error(f"Error getting unique documents: {e}")
        return []


def filter_existing_documents_batch(
    conn: sqlite3.Connection,
    documents: List[Tuple[str, int]],
    chunking_strategy: str,
    version_id: int
) -> List[Tuple[str, int]]:
    """배치 단위로 이미 처리된 문서 필터링 (최적화: 배치 조회)"""
    if not documents:
        return []
    
    # source_type별로 그룹화하여 배치 조회
    by_type = {}
    for source_type, source_id in documents:
        if source_type not in by_type:
            by_type[source_type] = []
        by_type[source_type].append(source_id)
    
    existing_docs = set()
    for source_type, source_ids in by_type.items():
        if not source_ids:
            continue
        
        # SQLite의 SQLITE_MAX_VARIABLE_NUMBER 제한 고려 (기본값 999)
        # 안전하게 500개씩 배치 처리
        batch_size = 500
        for i in range(0, len(source_ids), batch_size):
            batch_ids = source_ids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch_ids))
            cursor = conn.execute(
                f"""SELECT DISTINCT source_type, source_id 
                    FROM text_chunks 
                    WHERE source_type = ?
                    AND source_id IN ({placeholders})
                    AND chunking_strategy = ? 
                    AND embedding_version_id = ?""",
                [source_type] + batch_ids + [chunking_strategy, version_id]
            )
            for row in cursor.fetchall():
                existing_docs.add((row['source_type'], row['source_id']))
    
    return [doc for doc in documents if doc not in existing_docs]


def restore_documents_batch(
    conn: sqlite3.Connection,
    documents: List[Tuple[str, int]]
) -> Dict[Tuple[str, int], str]:
    """여러 문서를 배치로 복원"""
    restored = {}
    
    # source_type별로 그룹화
    by_type = {}
    for source_type, source_id in documents:
        if source_type not in by_type:
            by_type[source_type] = []
        by_type[source_type].append(source_id)
    
    # 각 타입별로 배치 조회
    for source_type, source_ids in by_type.items():
        if not source_ids:
            continue
        
        placeholders = ','.join(['?'] * len(source_ids))
        
        if source_type == "statute_article":
            cursor = conn.execute(
                f"SELECT id, text FROM statute_articles WHERE id IN ({placeholders})",
                source_ids
            )
            for row in cursor.fetchall():
                restored[(source_type, row['id'])] = row['text']
        
        elif source_type == "case_paragraph":
            cursor = conn.execute(
                f"""SELECT case_id, GROUP_CONCAT(text, '\n') as full_text
                    FROM case_paragraphs 
                    WHERE case_id IN ({placeholders})
                    GROUP BY case_id""",
                source_ids
            )
            for row in cursor.fetchall():
                if row['full_text']:
                    restored[(source_type, row['case_id'])] = row['full_text']
                else:
                    cursor2 = conn.execute(
                        f"SELECT id, full_text FROM cases WHERE id IN ({placeholders})",
                        [row['case_id']]
                    )
                    row2 = cursor2.fetchone()
                    if row2 and row2['full_text']:
                        restored[(source_type, row['case_id'])] = row2['full_text']
        
        elif source_type == "decision_paragraph":
            cursor = conn.execute(
                f"""SELECT decision_id, GROUP_CONCAT(text, '\n') as full_text
                    FROM decision_paragraphs 
                    WHERE decision_id IN ({placeholders})
                    GROUP BY decision_id""",
                source_ids
            )
            for row in cursor.fetchall():
                if row['full_text']:
                    restored[(source_type, row['decision_id'])] = row['full_text']
                else:
                    cursor2 = conn.execute(
                        f"SELECT id, text FROM decisions WHERE id IN ({placeholders})",
                        [row['decision_id']]
                    )
                    row2 = cursor2.fetchone()
                    if row2 and row2['text']:
                        restored[(source_type, row['decision_id'])] = row2['text']
        
        elif source_type == "interpretation_paragraph":
            cursor = conn.execute(
                f"""SELECT interpretation_id, GROUP_CONCAT(text, '\n') as full_text
                    FROM interpretation_paragraphs 
                    WHERE interpretation_id IN ({placeholders})
                    GROUP BY interpretation_id""",
                source_ids
            )
            for row in cursor.fetchall():
                if row['full_text']:
                    restored[(source_type, row['interpretation_id'])] = row['full_text']
                else:
                    cursor2 = conn.execute(
                        f"SELECT id, text FROM interpretations WHERE id IN ({placeholders})",
                        [row['interpretation_id']]
                    )
                    row2 = cursor2.fetchone()
                    if row2 and row2['text']:
                        restored[(source_type, row['interpretation_id'])] = row2['text']
    
    return restored


def delete_chunks_batch(
    conn: sqlite3.Connection,
    documents: List[Tuple[str, int]],
    version_id: int
) -> Dict[Tuple[str, int], int]:
    """여러 문서의 청크를 배치로 삭제"""
    if not documents:
        return {}
    
    deleted_counts = {}
    
    # source_type과 source_id를 분리하여 IN 절 사용
    source_types = list(set(doc[0] for doc in documents))
    source_ids = list(set(doc[1] for doc in documents))
    
    type_placeholders = ','.join(['?'] * len(source_types))
    id_placeholders = ','.join(['?'] * len(source_ids))
    
    # 삭제할 청크 ID 조회
    cursor = conn.execute(
        f"""SELECT source_type, source_id, COUNT(*) as count
            FROM text_chunks 
            WHERE source_type IN ({type_placeholders})
            AND source_id IN ({id_placeholders})
            AND embedding_version_id = ?
            GROUP BY source_type, source_id""",
        source_types + source_ids + [version_id]
    )
    for row in cursor.fetchall():
        deleted_counts[(row['source_type'], row['source_id'])] = row['count']
    
    # embeddings 먼저 삭제
    conn.execute(
        f"""DELETE FROM embeddings 
            WHERE chunk_id IN (
                SELECT id FROM text_chunks 
                WHERE source_type IN ({type_placeholders})
                AND source_id IN ({id_placeholders})
                AND embedding_version_id = ?
            )""",
        source_types + source_ids + [version_id]
    )
    
    # text_chunks 삭제
    conn.execute(
        f"""DELETE FROM text_chunks 
            WHERE source_type IN ({type_placeholders})
            AND source_id IN ({id_placeholders})
            AND embedding_version_id = ?""",
        source_types + source_ids + [version_id]
    )
    
    return deleted_counts


def collect_chunks_for_batch(
    conn: sqlite3.Connection,
    documents: List[Tuple[str, int]],
    version_manager: EmbeddingVersionManager,
    version_id: int,
    chunking_strategy: str,
    query_type: Optional[str],
    skip_if_exists: bool = True,
    strategy=None
) -> Tuple[List[ChunkData], List[Tuple[str, int]], Dict[Tuple[str, int], int]]:
    """
    여러 문서의 청크를 수집하여 배치 임베딩 준비
    
    Returns:
        (chunks_data, processed_docs, deleted_counts)
    """
    chunks_data = []
    processed_docs = []
    deleted_counts = {}
    
    # 청킹 전략 재사용 (함수 외부에서 생성된 경우)
    if strategy is None:
        strategy = ChunkingFactory.create_strategy(
            strategy_name=chunking_strategy,
            query_type=query_type
        )
    
    # 배치 단위로 이미 처리된 문서 필터링
    if skip_if_exists:
        documents = filter_existing_documents_batch(
            conn, documents, chunking_strategy, version_id
        )
        if not documents:
            return [], [], {}
    
    # 배치 단위로 문서 복원
    restored_docs = restore_documents_batch(conn, documents)
    
    # 메모리 정리: 문서 복원 후 불필요한 데이터 정리
    gc.collect()
    
    # 배치 단위로 청크 삭제 (각 문서별로 정확하게 삭제)
    # UNIQUE 제약이 (source_type, source_id, chunk_index)에 걸려 있으므로
    # 모든 버전의 청크를 삭제해야 함 (재임베딩은 완전 교체 방식)
    if restored_docs:
        deleted_counts = {}
        for source_type, source_id in restored_docs.keys():
            # 해당 문서의 모든 버전 청크 개수 조회
            cursor = conn.execute(
                "SELECT COUNT(*) FROM text_chunks WHERE source_type = ? AND source_id = ?",
                (source_type, source_id)
            )
            count = cursor.fetchone()[0]
            if count > 0:
                deleted_counts[(source_type, source_id)] = count
                # embeddings 먼저 삭제 (모든 버전)
                conn.execute(
                    """DELETE FROM embeddings 
                        WHERE chunk_id IN (
                            SELECT id FROM text_chunks 
                            WHERE source_type = ? AND source_id = ?
                        )""",
                    (source_type, source_id)
                )
                # text_chunks 삭제 (모든 버전 - UNIQUE 제약 충돌 방지)
                conn.execute(
                    "DELETE FROM text_chunks WHERE source_type = ? AND source_id = ?",
                    (source_type, source_id)
                )
        # 삭제 후 즉시 커밋하여 UNIQUE 제약 오류 방지
        if deleted_counts:
            conn.commit()
    
    # 각 문서 처리
    for source_type, source_id in tqdm(documents, desc="청킹 수집", leave=False):
        # 복원된 문서 확인
        original_text = restored_docs.get((source_type, source_id))
        if not original_text:
            continue
        
        # 원본 문서를 청킹 가능한 형식으로 변환
        if source_type == "statute_article":
            sentences = [s.strip() for s in original_text.split('\n') if s.strip()]
            content = sentences
        else:
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
            continue
        
        # chunk_index는 0부터 시작 (삭제 후 재생성하므로)
        next_chunk_index = 0
        
        # 청크 데이터 수집
        for i, chunk_result in enumerate(chunk_results):
            chunks_data.append(ChunkData(
                source_type=source_type,
                source_id=source_id,
                chunk_result=chunk_result,
                chunk_index_offset=next_chunk_index + i
            ))
        
        processed_docs.append((source_type, source_id))
    
    return chunks_data, processed_docs, deleted_counts


def insert_chunks_and_embeddings_batch(
    conn: sqlite3.Connection,
    chunks_data: List[ChunkData],
    embeddings: List[Any],
    version_id: int,
    model_name: str,
    dim: int
) -> None:
    """청크 및 임베딩을 배치로 삽입"""
    if not chunks_data:
        return
    
    # 청크 삽입 데이터 준비
    chunk_inserts = []
    for chunk_data in chunks_data:
        metadata = chunk_data.chunk_result.metadata
        chunk_inserts.append((
            chunk_data.source_type,
            chunk_data.source_id,
            metadata.get("level", "paragraph"),
            chunk_data.chunk_index_offset,
            None, None, None,
            chunk_data.chunk_result.text,
            None,
            metadata.get("chunking_strategy"),
            metadata.get("chunk_size_category"),
            metadata.get("chunk_group_id"),
            metadata.get("query_type"),
            metadata.get("original_document_id"),
            version_id
        ))
    
    # executemany로 배치 삽입 (성능 향상)
    conn.executemany(
        """INSERT INTO text_chunks(
            source_type, source_id, level, chunk_index,
            start_char, end_char, overlap_chars, text, token_count, meta,
            chunking_strategy, chunk_size_category, chunk_group_id,
            query_type, original_document_id, embedding_version_id
        ) VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?)""",
        chunk_inserts
    )
    
    # 삽입된 청크 ID를 배치로 조회
    chunk_ids = []
    for chunk_data in chunks_data:
        cursor = conn.execute(
            """SELECT id FROM text_chunks 
               WHERE source_type = ? AND source_id = ? AND chunk_index = ? 
               AND embedding_version_id = ?
               ORDER BY id DESC LIMIT 1""",
            (chunk_data.source_type, chunk_data.source_id, 
             chunk_data.chunk_index_offset, version_id)
        )
        row = cursor.fetchone()
        if row:
            chunk_ids.append(row[0])
        else:
            logger.warning(f"청크 ID를 찾을 수 없음: {chunk_data.source_type}/{chunk_data.source_id}/{chunk_data.chunk_index_offset}")
    
    # 임베딩 삽입
    embedding_data = [
        (chunk_id, model_name, dim, vec.tobytes(), version_id)
        for chunk_id, vec in zip(chunk_ids, embeddings)
    ]
    
    conn.executemany(
        "INSERT INTO embeddings(chunk_id, model, dim, vector, version_id) VALUES(?,?,?,?,?)",
        embedding_data
    )
    
    # 메모리 정리: 임베딩 데이터 삭제
    del embedding_data
    del chunk_ids
    del chunk_inserts


def re_embed_documents_batch_optimized(
    conn: sqlite3.Connection,
    documents: List[Tuple[str, int]],
    embedder: SentenceEmbedder,
    version_manager: EmbeddingVersionManager,
    version_id: int,
    chunking_strategy: str = "standard",
    query_type: Optional[str] = None,
    doc_batch_size: int = 200,
    embedding_batch_size: int = 512,
    skip_if_exists: bool = True,
    commit_interval: int = 5
) -> Tuple[int, int, int]:
    """
    여러 문서를 배치로 처리하여 성능 향상
    
    Args:
        commit_interval: 몇 개 배치마다 커밋할지 (기본값: 5)
    
    Returns:
        (total_deleted, total_inserted, skipped_count)
    """
    total_deleted = 0
    total_inserted = 0
    skipped_count = 0
    
    # 청킹 전략 한 번만 생성하여 재사용
    strategy = ChunkingFactory.create_strategy(
        strategy_name=chunking_strategy,
        query_type=query_type
    )
    
    # 문서를 배치로 처리
    batch_num = 0
    for batch_start in range(0, len(documents), doc_batch_size):
        batch_num += 1
        batch_docs = documents[batch_start:batch_start + doc_batch_size]
        
        # 1단계: 배치 문서의 청크 수집
        chunks_data, processed_docs, deleted_counts = collect_chunks_for_batch(
            conn=conn,
            documents=batch_docs,
            version_manager=version_manager,
            version_id=version_id,
            chunking_strategy=chunking_strategy,
            query_type=query_type,
            skip_if_exists=skip_if_exists,
            strategy=strategy
        )
        
        if not chunks_data:
            skipped_count += len(batch_docs) - len(processed_docs)
            continue
        
        # 2단계: 모든 청크를 한 번에 임베딩 생성 (가장 큰 성능 향상)
        all_texts = [chunk_data.chunk_result.text for chunk_data in chunks_data]
        logger.info(f"배치 임베딩 생성: {len(all_texts)}개 청크")
        
        all_embeddings = embedder.encode(all_texts, batch_size=embedding_batch_size)
        dim = all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else all_embeddings.shape[0]
        model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
        # 3단계: 배치로 DB 삽입
        insert_chunks_and_embeddings_batch(
            conn=conn,
            chunks_data=chunks_data,
            embeddings=all_embeddings,
            version_id=version_id,
            model_name=model_name,
            dim=dim
        )
        
        # 메모리 정리: 대용량 객체 삭제
        del all_texts
        del all_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 통계 업데이트 (삭제 전에 값 저장)
        processed_count = len(processed_docs)
        chunks_count = len(chunks_data)
        
        # 커밋 최적화: 여러 배치마다 커밋
        if batch_num % commit_interval == 0:
            conn.commit()
            logger.info(f"커밋 완료 (배치 {batch_num}): {processed_count}개 문서 처리, {chunks_count}개 청크 생성")
            # 주기적인 가비지 컬렉션 (커밋 시마다)
            gc.collect()
        for doc in processed_docs:
            total_deleted += deleted_counts.get(doc, 0)
        total_inserted += chunks_count
        skipped_count += len(batch_docs) - processed_count
        
        # 배치 데이터 정리
        del chunks_data
        del processed_docs
        del deleted_counts
        
        logger.info(f"배치 {batch_num} 처리 완료: {processed_count}개 문서, {chunks_count}개 청크 (전체 진행률: {batch_num * doc_batch_size}/{len(documents)})")
    
    # 마지막 커밋
    conn.commit()
    
    return total_deleted, total_inserted, skipped_count


def optimize_database_indexes(conn: sqlite3.Connection):
    """데이터베이스 인덱스 최적화"""
    logger.info("데이터베이스 인덱스 최적화 중...")
    
    indexes = [
        ("idx_case_paragraphs_case_id_para_index", 
         "CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case_id_para_index ON case_paragraphs(case_id, para_index)"),
        ("idx_decision_paragraphs_decision_id_para_index",
         "CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision_id_para_index ON decision_paragraphs(decision_id, para_index)"),
        ("idx_interpretation_paragraphs_interpretation_id_para_index",
         "CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interpretation_id_para_index ON interpretation_paragraphs(interpretation_id, para_index)"),
        ("idx_text_chunks_source_lookup",
         "CREATE INDEX IF NOT EXISTS idx_text_chunks_source_lookup ON text_chunks(source_type, source_id, chunking_strategy, embedding_version_id)"),
    ]
    
    for index_name, create_sql in indexes:
        try:
            conn.execute(create_sql)
            logger.debug(f"인덱스 생성/확인: {index_name}")
        except Exception as e:
            logger.warning(f"인덱스 생성 실패 ({index_name}): {e}")
    
    conn.commit()
    logger.info("인덱스 최적화 완료")


def main():
    parser = argparse.ArgumentParser(description='기존 데이터 재임베딩 (최적화 버전)')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--source-type', help='특정 source_type만 처리 (예: statute_article, case_paragraph)')
    parser.add_argument('--chunking-strategy', default='standard', choices=['standard', 'dynamic', 'hybrid'],
                       help='사용할 청킹 전략')
    parser.add_argument('--query-type', help='동적 청킹을 위한 쿼리 타입')
    parser.add_argument('--embedding-batch-size', type=int, default=None, help='임베딩 배치 크기 (지정하지 않으면 자동 조정)')
    parser.add_argument('--doc-batch-size', type=int, default=500, help='문서 배치 크기 (기본값: 500)')
    parser.add_argument('--commit-interval', type=int, default=5, help='몇 개 배치마다 커밋할지 (기본값: 5)')
    parser.add_argument('--limit', type=int, help='처리할 문서 수 제한 (테스트용)')
    parser.add_argument('--version-id', type=int, help='사용할 임베딩 버전 ID (지정하지 않으면 활성 버전 사용)')
    parser.add_argument('--dry-run', action='store_true', help='실제 변경 없이 테스트만 수행')
    parser.add_argument('--skip-index-optimization', action='store_true', help='인덱스 최적화 건너뛰기')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN 모드: 실제 변경 없이 테스트만 수행합니다.")
    
    # GPU 사용 확인 및 배치 크기 자동 조정
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA 사용 가능: {cuda_available}")
    
    # 임베딩 배치 크기 자동 조정
    if args.embedding_batch_size is None:
        if cuda_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 8 * 1024**3:  # 8GB 이상
                args.embedding_batch_size = 4096
            elif gpu_memory > 4 * 1024**3:  # 4GB 이상
                args.embedding_batch_size = 2048
            else:
                args.embedding_batch_size = 1024
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"자동 조정된 임베딩 배치 크기: {args.embedding_batch_size}")
        else:
            # CPU 사용 시 메모리 기반 조정
            try:
                import psutil
                available_memory = psutil.virtual_memory().available
                if available_memory > 16 * 1024**3:  # 16GB 이상
                    args.embedding_batch_size = 2048
                elif available_memory > 10 * 1024**3:  # 10GB 이상
                    args.embedding_batch_size = 1536
                elif available_memory > 8 * 1024**3:  # 8GB 이상
                    args.embedding_batch_size = 1024
                else:
                    args.embedding_batch_size = 512
                logger.info(f"사용 가능한 메모리: {available_memory / 1024**3:.2f} GB")
                logger.info(f"자동 조정된 임베딩 배치 크기: {args.embedding_batch_size}")
            except ImportError:
                logger.warning("psutil이 설치되지 않아 기본값(512) 사용")
                args.embedding_batch_size = 512
    else:
        logger.info(f"지정된 임베딩 배치 크기: {args.embedding_batch_size}")
    
    # 데이터베이스 연결
    conn = sqlite3.connect(args.db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")  # 성능 최적화
    conn.execute("PRAGMA cache_size = -256000")  # 256MB 캐시 (128MB에서 증가)
    conn.execute("PRAGMA temp_store = MEMORY")  # 임시 테이블을 메모리에 저장
    conn.execute("PRAGMA mmap_size = 536870912")  # 512MB 메모리 맵 (256MB에서 증가)
    
    # 인덱스 최적화
    if not args.skip_index_optimization:
        optimize_database_indexes(conn)
    
    # 임베더 초기화
    logger.info("임베더 초기화 중...")
    embedder = SentenceEmbedder()
    logger.info(f"임베더 디바이스: {embedder.device}")
    
    # 고유한 문서 목록 조회
    logger.info("고유한 문서 목록 조회 중...")
    documents = get_unique_documents(conn, args.source_type)
    
    if args.limit:
        documents = documents[:args.limit]
        logger.info(f"제한 적용: {args.limit}개 문서만 처리")
    
    logger.info(f"총 {len(documents)}개 문서를 재임베딩합니다.")
    logger.info(f"청킹 전략: {args.chunking_strategy}")
    logger.info(f"문서 배치 크기: {args.doc_batch_size}")
    logger.info(f"임베딩 배치 크기: {args.embedding_batch_size}")
    
    if not args.dry_run:
        # 버전 관리자 초기화 (한 번만 수행)
        db_path = args.db
        version_manager = EmbeddingVersionManager(db_path)
        
        # 버전 ID 결정
        if args.version_id:
            # 지정된 버전 ID 사용
            version_id = args.version_id
            version_info = version_manager.get_version_statistics(version_id)
            if not version_info:
                raise ValueError(f"버전 ID {version_id}를 찾을 수 없습니다.")
            logger.info(f"지정된 버전 사용: {version_info.get('version_name', 'N/A')} (ID: {version_id})")
        else:
            # 활성 버전 조회 또는 생성 (한 번만 수행)
            active_version = version_manager.get_active_version(args.chunking_strategy)
            if not active_version:
                model_name = getattr(embedder.model, 'name_or_path', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS')
                version_name = f"v1.0.0-{args.chunking_strategy}"
                version_id = version_manager.register_version(
                    version_name=version_name,
                    chunking_strategy=args.chunking_strategy,
                    model_name=model_name,
                    description=f"{args.chunking_strategy} 청킹 전략 (재임베딩 최적화)",
                    set_active=True
                )
            else:
                version_id = active_version['id']
        
        logger.info(f"사용할 버전 ID: {version_id}")
        
        # 최적화된 배치 재임베딩
        total_deleted, total_inserted, skipped_count = re_embed_documents_batch_optimized(
            conn=conn,
            documents=documents,
            embedder=embedder,
            version_manager=version_manager,
            version_id=version_id,
            chunking_strategy=args.chunking_strategy,
            query_type=args.query_type,
            doc_batch_size=args.doc_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            skip_if_exists=True,
            commit_interval=getattr(args, 'commit_interval', 5)
        )
        
        logger.info(f"재임베딩 완료: {total_deleted}개 청크 삭제, {total_inserted}개 청크 삽입, {skipped_count}개 문서 건너뜀")
    else:
        logger.info(f"DRY RUN: {len(documents)}개 문서를 처리할 예정입니다.")
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

