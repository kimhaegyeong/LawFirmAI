"""
청킹 전략 변경 시 완전 교체 방식으로 마이그레이션하는 스크립트

기존 청크를 삭제하고 새 청킹 전략으로 재생성합니다.
"""
import sqlite3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from scripts.utils.chunking.factory import ChunkingFactory
from scripts.utils.embeddings import SentenceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_document_chunking(
    db_path: str,
    source_type: str,
    source_id: int,
    new_chunking_strategy: str,
    new_query_type: Optional[str] = None,
    model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    replace_existing: bool = True
) -> bool:
    """
    특정 문서의 청킹 전략을 완전 교체 방식으로 변경
    
    Args:
        db_path: 데이터베이스 경로
        source_type: 소스 타입
        source_id: 소스 ID
        new_chunking_strategy: 새 청킹 전략
        new_query_type: 새 쿼리 타입 (동적 청킹용)
        model_name: 임베딩 모델명
        replace_existing: 기존 청크 삭제 여부
    
    Returns:
        bool: 성공 여부
    """
    version_manager = EmbeddingVersionManager(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # 1. 활성 버전 조회 또는 생성
        active_version = version_manager.get_active_version(new_chunking_strategy)
        if not active_version:
            # 새 버전 등록
            version_name = f"v1.0.0-{new_chunking_strategy}"
            version_id = version_manager.register_version(
                version_name=version_name,
                chunking_strategy=new_chunking_strategy,
                model_name=model_name,
                description=f"{new_chunking_strategy} 청킹 전략",
                set_active=True
            )
        else:
            version_id = active_version['id']
        
        # 2. 기존 청크 삭제 (완전 교체 방식)
        if replace_existing:
            deleted_chunks, deleted_embeddings = version_manager.delete_chunks_by_version(
                source_type=source_type,
                source_id=source_id
            )
            logger.info(
                f"Deleted {deleted_chunks} chunks and {deleted_embeddings} embeddings "
                f"for {source_type}/{source_id}"
            )
        
        # 3. 원본 문서 내용 조회
        content = None
        if source_type == "statute_article":
            cursor = conn.execute(
                "SELECT text FROM statute_articles WHERE id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row:
                # sentences 리스트로 변환 (간단히 줄바꿈으로 분리)
                content = [s.strip() for s in row['text'].split('\n') if s.strip()]
        elif source_type in ["case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
            if source_type == "case_paragraph":
                cursor = conn.execute(
                    "SELECT text FROM case_paragraphs WHERE case_id = ? ORDER BY para_index",
                    (source_id,)
                )
            elif source_type == "decision_paragraph":
                cursor = conn.execute(
                    "SELECT text FROM decision_paragraphs WHERE decision_id = ? ORDER BY para_index",
                    (source_id,)
                )
            else:  # interpretation_paragraph
                cursor = conn.execute(
                    "SELECT text FROM interpretation_paragraphs WHERE interpretation_id = ? ORDER BY para_index",
                    (source_id,)
                )
            
            content = [row['text'] for row in cursor.fetchall() if row['text']]
        
        if not content:
            logger.warning(f"No content found for {source_type}/{source_id}")
            return False
        
        # 4. 새 청킹 전략으로 청크 생성
        strategy = ChunkingFactory.create_strategy(
            strategy_name=new_chunking_strategy,
            query_type=new_query_type
        )
        
        chunk_results = strategy.chunk(
            content=content,
            source_type=source_type,
            source_id=source_id
        )
        
        if not chunk_results:
            logger.warning(f"No chunks generated for {source_type}/{source_id}")
            return False
        
        # 5. 새 청크 및 임베딩 저장
        embedder = SentenceEmbedder(model_name=model_name)
        
        chunk_ids = []
        texts_to_embed = []
        
        for chunk_result in chunk_results:
            metadata = chunk_result.metadata
            cursor = conn.execute("""
                INSERT INTO text_chunks(
                    source_type, source_id, level, chunk_index,
                    start_char, end_char, overlap_chars, text, token_count, meta,
                    chunking_strategy, chunk_size_category, chunk_group_id, 
                    query_type, original_document_id, embedding_version_id
                ) VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?)
            """, (
                source_type,
                source_id,
                metadata.get("level", "paragraph"),
                chunk_result.chunk_index,
                None, None, None,
                chunk_result.text,
                None,
                metadata.get("chunking_strategy"),
                metadata.get("chunk_size_category"),
                metadata.get("chunk_group_id"),
                metadata.get("query_type"),
                metadata.get("original_document_id"),
                version_id
            ))
            
            chunk_id = cursor.lastrowid
            chunk_ids.append(chunk_id)
            texts_to_embed.append(chunk_result.text)
        
        # 6. 임베딩 생성 및 저장
        if texts_to_embed:
            vecs = embedder.encode(texts_to_embed, batch_size=64)
            dim = vecs.shape[1] if len(vecs.shape) > 1 else vecs.shape[0]
            
            for chunk_id, vec in zip(chunk_ids, vecs):
                conn.execute("""
                    INSERT INTO embeddings(chunk_id, model, dim, vector, version_id)
                    VALUES(?,?,?,?,?)
                """, (
                    chunk_id,
                    model_name,
                    dim,
                    vec.tobytes(),
                    version_id
                ))
        
        conn.commit()
        logger.info(
            f"Successfully migrated {source_type}/{source_id} to {new_chunking_strategy} "
            f"({len(chunk_ids)} chunks)"
        )
        
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to migrate {source_type}/{source_id}: {e}", exc_info=True)
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='청킹 전략 완전 교체 마이그레이션'
    )
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--source-type', required=True, 
                       choices=['statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph'],
                       help='소스 타입')
    parser.add_argument('--source-id', type=int, required=True, help='소스 ID')
    parser.add_argument('--strategy', required=True,
                       choices=['standard', 'dynamic', 'hybrid'],
                       help='새 청킹 전략')
    parser.add_argument('--query-type', help='쿼리 타입 (동적 청킹용)')
    parser.add_argument('--model', default='snunlp/KR-SBERT-V40K-klueNLI-augSTS',
                       help='임베딩 모델명')
    parser.add_argument('--no-replace', action='store_true',
                       help='기존 청크 유지 (공존 방식)')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
    
    success = migrate_document_chunking(
        db_path=str(db_path),
        source_type=args.source_type,
        source_id=args.source_id,
        new_chunking_strategy=args.strategy,
        new_query_type=args.query_type,
        model_name=args.model,
        replace_existing=not args.no_replace
    )
    
    if success:
        logger.info("Migration completed successfully")
    else:
        logger.error("Migration failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

