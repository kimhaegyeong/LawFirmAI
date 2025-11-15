"""
FAISS 버전 마이그레이션 스크립트

기존 FAISS 인덱스를 새 버전으로 마이그레이션합니다.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.faiss_version_manager import FAISSVersionManager
from utils.embedding_version_manager import EmbeddingVersionManager
from utils.faiss_migration_manager import FAISSMigrationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Migrate FAISS index to new version")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--source-version", required=True, help="Source FAISS version name")
    parser.add_argument("--target-version", required=True, help="Target FAISS version name")
    parser.add_argument("--embedding-version-id", type=int, required=True, help="Target embedding version ID")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--vector-store-base", default="data/vector_store", help="Vector store base path")
    
    args = parser.parse_args()
    
    db_path = args.db
    source_version = args.source_version
    target_version = args.target_version
    embedding_version_id = args.embedding_version_id
    batch_size = args.batch_size
    vector_store_base = args.vector_store_base
    
    logger.info(f"Starting FAISS migration: {source_version} -> {target_version}")
    
    faiss_version_manager = FAISSVersionManager(vector_store_base)
    embedding_version_manager = EmbeddingVersionManager(db_path)
    migration_manager = FAISSMigrationManager(
        faiss_version_manager,
        embedding_version_manager,
        db_path
    )
    
    version_info = embedding_version_manager.get_version_statistics(embedding_version_id)
    if not version_info:
        logger.error(f"Embedding version {embedding_version_id} not found")
        return 1
    
    logger.info(f"Target embedding version: {version_info.get('version_name')}")
    
    source_path = faiss_version_manager.get_version_path(source_version)
    if not source_path:
        logger.error(f"Source version {source_version} not found")
        return 1
    
    logger.info(f"Source version path: {source_path}")
    
    logger.info("Migration script prepared. Actual migration logic should be implemented based on specific requirements.")
    logger.info("This script provides the framework for document-by-document migration.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

