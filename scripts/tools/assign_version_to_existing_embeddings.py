"""
기존 임베딩에 버전 ID 할당 스크립트

데이터베이스에 이미 존재하는 임베딩 데이터에 embedding_version_id를 할당합니다.
"""
import argparse
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def assign_version_to_all_chunks(db_path: str, version_id: int, dry_run: bool = False):
    """
    모든 청크에 버전 ID 할당
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 할당할 버전 ID
        dry_run: 실제로 할당하지 않고 통계만 조회
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # 현재 상태 확인
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding_version_id IS NULL THEN 1 END) as null_version_chunks,
                COUNT(CASE WHEN embedding_version_id IS NOT NULL THEN 1 END) as assigned_chunks
            FROM text_chunks
        """)
        stats = dict(cursor.fetchone())
        
        logger.info("=" * 80)
        logger.info("Current Status")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Chunks with version ID: {stats['assigned_chunks']}")
        logger.info(f"Chunks without version ID: {stats['null_version_chunks']}")
        
        if stats['null_version_chunks'] == 0:
            logger.info("All chunks already have version IDs assigned")
            return True
        
        # 버전 정보 확인
        cursor.execute("""
            SELECT * FROM embedding_versions WHERE id = ?
        """, (version_id,))
        version_row = cursor.fetchone()
        
        if not version_row:
            logger.error(f"Version ID {version_id} not found")
            return False
        
        version_info = dict(version_row)
        logger.info(f"\nTarget version: {version_info['version_name']} (ID: {version_id})")
        logger.info(f"Chunking strategy: {version_info['chunking_strategy']}")
        
        if dry_run:
            logger.info("\n[DRY RUN] Would assign version ID to chunks")
            logger.info(f"Chunks to update: {stats['null_version_chunks']}")
            return True
        
        # 버전 ID 할당
        logger.info("\n" + "=" * 80)
        logger.info("Assigning Version ID")
        logger.info("=" * 80)
        
        cursor.execute("""
            UPDATE text_chunks
            SET embedding_version_id = ?
            WHERE embedding_version_id IS NULL
        """, (version_id,))
        
        updated_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"✓ Assigned version ID {version_id} to {updated_count} chunks")
        
        # 최종 상태 확인
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding_version_id = ? THEN 1 END) as version_chunks
            FROM text_chunks
        """, (version_id,))
        final_stats = dict(cursor.fetchone())
        
        logger.info(f"\nFinal status:")
        logger.info(f"Total chunks: {final_stats['total_chunks']}")
        logger.info(f"Chunks with version ID {version_id}: {final_stats['version_chunks']}")
        
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to assign version ID: {e}", exc_info=True)
        return False
    finally:
        conn.close()


def assign_version_by_strategy(db_path: str, version_id: int, chunking_strategy: str, dry_run: bool = False):
    """
    특정 청킹 전략의 청크에만 버전 ID 할당
    
    Args:
        db_path: 데이터베이스 경로
        version_id: 할당할 버전 ID
        chunking_strategy: 청킹 전략 필터
        dry_run: 실제로 할당하지 않고 통계만 조회
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # 현재 상태 확인
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding_version_id IS NULL THEN 1 END) as null_version_chunks
            FROM text_chunks
            WHERE chunking_strategy = ?
        """, (chunking_strategy,))
        stats = dict(cursor.fetchone())
        
        logger.info("=" * 80)
        logger.info(f"Chunks with strategy '{chunking_strategy}'")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Chunks without version ID: {stats['null_version_chunks']}")
        
        if stats['null_version_chunks'] == 0:
            logger.info("All chunks already have version IDs assigned")
            return True
        
        if dry_run:
            logger.info(f"\n[DRY RUN] Would assign version ID {version_id} to {stats['null_version_chunks']} chunks")
            return True
        
        # 버전 ID 할당
        logger.info("\n" + "=" * 80)
        logger.info("Assigning Version ID")
        logger.info("=" * 80)
        
        cursor.execute("""
            UPDATE text_chunks
            SET embedding_version_id = ?
            WHERE chunking_strategy = ? AND embedding_version_id IS NULL
        """, (version_id, chunking_strategy))
        
        updated_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"✓ Assigned version ID {version_id} to {updated_count} chunks")
        
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to assign version ID: {e}", exc_info=True)
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Assign version ID to existing embeddings")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--version-id", type=int, required=True, help="Version ID to assign")
    parser.add_argument("--chunking-strategy", help="Filter by chunking strategy (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual updates)")
    
    args = parser.parse_args()
    
    db_path = args.db
    version_id = args.version_id
    
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no changes will be made")
    
    if args.chunking_strategy:
        success = assign_version_by_strategy(
            db_path, 
            version_id, 
            args.chunking_strategy,
            dry_run=args.dry_run
        )
    else:
        success = assign_version_to_all_chunks(
            db_path, 
            version_id,
            dry_run=args.dry_run
        )
    
    if success:
        logger.info("\n✓ Operation completed successfully")
        return 0
    else:
        logger.error("\n✗ Operation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

