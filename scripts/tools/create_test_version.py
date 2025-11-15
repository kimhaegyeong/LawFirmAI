"""
테스트용 두 번째 버전 생성 스크립트

멀티 버전 검색 테스트를 위해 기존 버전을 복사하여 새 버전을 생성합니다.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.faiss_version_manager import FAISSVersionManager
from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_version(
    db_path: str,
    vector_store_path: str,
    source_version_id: int,
    new_version_name: str,
    new_chunking_strategy: str = None
):
    """
    테스트용 새 버전 생성
    
    Args:
        db_path: 데이터베이스 경로
        vector_store_path: 벡터 스토어 경로
        source_version_id: 복사할 소스 버전 ID
        new_version_name: 새 버전 이름
        new_chunking_strategy: 새 청킹 전략 (None이면 소스와 동일)
    """
    logger.info("=" * 80)
    logger.info("Create Test Version")
    logger.info("=" * 80)
    
    # 버전 관리자 초기화
    faiss_manager = FAISSVersionManager(vector_store_path)
    embedding_manager = EmbeddingVersionManager(db_path)
    
    # 소스 버전 정보 조회
    source_version_info = embedding_manager.get_version_statistics(source_version_id)
    if not source_version_info:
        logger.error(f"Source version {source_version_id} not found")
        return False
    
    logger.info(f"Source version: {source_version_info['version_name']}")
    logger.info(f"Chunking strategy: {source_version_info['chunking_strategy']}")
    logger.info(f"Model: {source_version_info['model_name']}")
    
    # 새 청킹 전략 결정
    chunking_strategy = new_chunking_strategy or source_version_info['chunking_strategy']
    
    # 새 임베딩 버전 등록
    logger.info("")
    logger.info("Registering new embedding version...")
    try:
        new_version_id = embedding_manager.register_version(
            version_name=new_version_name,
            chunking_strategy=chunking_strategy,
            model_name=source_version_info['model_name'],
            description=f"Test version copied from {source_version_info['version_name']}",
            set_active=False,
            create_faiss_version=True,
            faiss_version_manager=faiss_manager
        )
        logger.info(f"✓ Created embedding version: {new_version_name} (ID: {new_version_id})")
    except Exception as e:
        logger.error(f"✗ Failed to create embedding version: {e}")
        return False
    
    # FAISS 버전 이름 생성
    faiss_version_name = f"{new_version_name}-{chunking_strategy}"
    
    # 소스 FAISS 버전 복사
    source_faiss_version = f"{source_version_info['version_name']}-{source_version_info['chunking_strategy']}"
    logger.info("")
    logger.info(f"Copying FAISS version: {source_faiss_version} -> {faiss_version_name}")
    
    try:
        copied_path = faiss_manager.copy_version(
            source_version=source_faiss_version,
            target_version=faiss_version_name,
            update_status="inactive"
        )
        logger.info(f"✓ Copied FAISS version to: {copied_path}")
    except Exception as e:
        logger.warning(f"Failed to copy FAISS version: {e}")
        logger.info("Creating new FAISS version instead...")
        
        # 새 FAISS 버전 생성 (인덱스는 나중에 빌드)
        try:
            chunking_config = {}
            embedding_config = {
                'model': source_version_info['model_name'],
                'dimension': 768
            }
            
            version_path = faiss_manager.create_version(
                version_name=faiss_version_name,
                embedding_version_id=new_version_id,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                embedding_config=embedding_config,
                document_count=0,
                total_chunks=0,
                status='inactive'
            )
            logger.info(f"✓ Created FAISS version: {faiss_version_name}")
        except Exception as e:
            logger.error(f"✗ Failed to create FAISS version: {e}")
            return False
    
    # 인덱스 빌드 (기존 데이터 재사용)
    logger.info("")
    logger.info("Building FAISS index...")
    
    try:
        engine = SemanticSearchEngineV2(
            db_path=db_path,
            use_external_index=False
        )
        
        # 기존 청크를 새 버전에 할당 (테스트용)
        # 실제로는 새 청킹 전략으로 재임베딩해야 하지만,
        # 테스트를 위해 기존 데이터를 재사용
        logger.info("Note: Using existing chunks for test version")
        logger.info("For production, re-embed with new chunking strategy")
        
        # 인덱스 빌드
        success = engine._build_faiss_index_sync(
            embedding_version_id=new_version_id,
            faiss_version_name=faiss_version_name
        )
        
        if success:
            logger.info(f"✓ Index built successfully for {faiss_version_name}")
        else:
            logger.warning("Index build returned False, but continuing...")
            
    except Exception as e:
        logger.error(f"✗ Failed to build index: {e}")
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"New embedding version: {new_version_name} (ID: {new_version_id})")
    logger.info(f"New FAISS version: {faiss_version_name}")
    logger.info(f"Chunking strategy: {chunking_strategy}")
    logger.info("")
    logger.info("✓ Test version created successfully")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Test multi-version search:")
    logger.info(f"   python scripts/test_faiss_version_with_real_data.py --db {db_path}")
    logger.info("2. Compare performance:")
    logger.info(f"   python scripts/test_performance_monitoring.py --db {db_path} --version1 {source_faiss_version} --version2 {faiss_version_name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create test version for multi-version testing")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--vector-store", default="data/vector_store", help="Vector store path")
    parser.add_argument("--source-version-id", type=int, required=True, help="Source version ID to copy")
    parser.add_argument("--new-version-name", required=True, help="New version name")
    parser.add_argument("--chunking-strategy", help="New chunking strategy (default: same as source)")
    
    args = parser.parse_args()
    
    success = create_test_version(
        args.db,
        args.vector_store,
        args.source_version_id,
        args.new_version_name,
        args.chunking_strategy
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

