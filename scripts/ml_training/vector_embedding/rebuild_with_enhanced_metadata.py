#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터스토어 전체 재생성 스크립트 (Enhanced Metadata)

기존 벡터스토어를 백업하고, metadata가 강화된 새 버전(v2.0.0)으로 재생성합니다.
"""

import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml_training.vector_embedding.incremental_precedent_vector_builder import IncrementalPrecedentVectorBuilder
from scripts.ml_training.vector_embedding.incremental_vector_builder import IncrementalVectorBuilder
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

logger = logging.getLogger(__name__)


def backup_vector_store(base_path: Path, backup_suffix: str = None) -> Path:
    """
    벡터스토어 백업
    
    Args:
        base_path: 벡터스토어 기본 경로
        backup_suffix: 백업 디렉토리 접미사 (None이면 타임스탬프 사용)
    
    Returns:
        Path: 백업 경로
    """
    if backup_suffix is None:
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_path = base_path.parent / f"{base_path.name}_backup_{backup_suffix}"
    
    if base_path.exists():
        logger.info(f"Backing up vector store from {base_path} to {backup_path}")
        shutil.copytree(base_path, backup_path, dirs_exist_ok=True)
        logger.info(f"Backup completed: {backup_path}")
    else:
        logger.warning(f"Vector store path does not exist: {base_path}. Skipping backup.")
    
    return backup_path


def rebuild_precedent_vector_store(
    base_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents",
    version: str = "v2.0.0",
    backup: bool = True
) -> bool:
    """
    판례 벡터스토어 재생성
    
    Args:
        base_path: 벡터스토어 기본 경로
        version: 새 버전 번호
        backup: 백업 여부
    
    Returns:
        bool: 성공 여부
    """
    base_path = Path(base_path)
    
    if backup:
        backup_vector_store(base_path)
    
    version_manager = VectorStoreVersionManager(base_path)
    
    builder = IncrementalPrecedentVectorBuilder(
        embedding_output_path=str(base_path),
        version=version
    )
    
    version_metadata = {
        "model_name": builder.model_name,
        "dimension": builder.dimension,
        "index_type": builder.index_type,
        "document_count": 0,
        "metadata_schema_version": "2.0",
        "changes": [
            "Added case_number, court, decision_date, announce_date, doc_id, casenames to metadata",
            "Added type and source_type fields",
            "Enhanced metadata extraction from case data"
        ]
    }
    
    if not version_manager.create_version(version, version_metadata):
        logger.error(f"Failed to create version {version}")
        return False
    
    logger.info(f"Starting precedent vector store rebuild for version {version}")
    
    categories = ["civil", "criminal", "family", "tax", "administrative", "patent"]
    total_chunks = 0
    
    for category in categories:
        logger.info(f"Processing category: {category}")
        stats = builder.build_incremental_embeddings(category=category)
        total_chunks += stats.get('total_chunks_added', 0)
        logger.info(f"Category {category} completed. Chunks added: {stats.get('total_chunks_added', 0)}")
    
    version_metadata["document_count"] = total_chunks
    version_manager.update_version(version, {"document_count": total_chunks})
    
    logger.info(f"Precedent vector store rebuild completed. Total chunks: {total_chunks}")
    return True


def rebuild_statute_vector_store(
    base_path: str = "data/embeddings/ml_enhanced_ko_sroberta",
    version: str = "v2.0.0",
    backup: bool = True
) -> bool:
    """
    법령 벡터스토어 재생성
    
    Args:
        base_path: 벡터스토어 기본 경로
        version: 새 버전 번호
        backup: 백업 여부
    
    Returns:
        bool: 성공 여부
    """
    base_path = Path(base_path)
    
    if backup:
        backup_vector_store(base_path)
    
    version_manager = VectorStoreVersionManager(base_path)
    
    builder = IncrementalVectorBuilder(
        embedding_output_path=str(base_path),
        version=version
    )
    
    version_metadata = {
        "model_name": builder.model_name,
        "dimension": builder.dimension,
        "index_type": builder.index_type,
        "document_count": 0,
        "metadata_schema_version": "2.0",
        "changes": [
            "Added type and source_type fields",
            "Added raw_source_file field for source tracking"
        ]
    }
    
    if not version_manager.create_version(version, version_metadata):
        logger.error(f"Failed to create version {version}")
        return False
    
    logger.info(f"Starting statute vector store rebuild for version {version}")
    
    stats = builder.build_incremental_embeddings(data_type="law_only")
    total_chunks = stats.get('total_chunks_added', 0)
    
    version_metadata["document_count"] = total_chunks
    version_manager.update_version(version, {"document_count": total_chunks})
    
    logger.info(f"Statute vector store rebuild completed. Total chunks: {total_chunks}")
    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="벡터스토어 전체 재생성 (Enhanced Metadata)")
    parser.add_argument('--store-type', choices=['precedent', 'statute', 'all'], default='all',
                        help='재생성할 벡터스토어 타입')
    parser.add_argument('--version', default='v2.0.0',
                        help='새 버전 번호 (예: v2.0.0)')
    parser.add_argument('--precedent-path', default='data/embeddings/ml_enhanced_ko_sroberta_precedents',
                        help='판례 벡터스토어 경로')
    parser.add_argument('--statute-path', default='data/embeddings/ml_enhanced_ko_sroberta',
                        help='법령 벡터스토어 경로')
    parser.add_argument('--no-backup', action='store_true',
                        help='백업하지 않음 (주의: 기존 데이터가 삭제될 수 있음)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    backup = not args.no_backup
    
    try:
        success = True
        
        if args.store_type in ['precedent', 'all']:
            logger.info("=" * 60)
            logger.info("Rebuilding Precedent Vector Store")
            logger.info("=" * 60)
            if not rebuild_precedent_vector_store(
                base_path=args.precedent_path,
                version=args.version,
                backup=backup
            ):
                success = False
        
        if args.store_type in ['statute', 'all']:
            logger.info("=" * 60)
            logger.info("Rebuilding Statute Vector Store")
            logger.info("=" * 60)
            if not rebuild_statute_vector_store(
                base_path=args.statute_path,
                version=args.version,
                backup=backup
            ):
                success = False
        
        if success:
            logger.info("=" * 60)
            logger.info("Vector store rebuild completed successfully!")
            logger.info("=" * 60)
        else:
            logger.error("Vector store rebuild failed. Check logs for details.")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in vector store rebuild: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

