"""
실제 데이터로 FAISS 버전 관리 시스템 검증

실제 데이터베이스의 임베딩 데이터를 사용하여 FAISS 버전 생성, 전환, 검색을 테스트합니다.
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


def test_version_creation(db_path: str, vector_store_base: str):
    """버전 생성 테스트"""
    logger.info("=" * 80)
    logger.info("Test 1: FAISS Version Creation")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    embedding_manager = EmbeddingVersionManager(db_path)
    
    # 활성 버전 조회
    active_versions = embedding_manager.get_all_active_versions()
    if not active_versions:
        logger.warning("No active embedding versions found. Please create an embedding version first.")
        return False
    
    logger.info(f"Found {len(active_versions)} active embedding version(s)")
    
    for version_info in active_versions:
        version_id = version_info['id']
        version_name = version_info['version_name']
        chunking_strategy = version_info['chunking_strategy']
        model_name = version_info['model_name']
        
        logger.info(f"\nProcessing version: {version_name} (ID: {version_id})")
        logger.info(f"  Chunking strategy: {chunking_strategy}")
        logger.info(f"  Model: {model_name}")
        
        # 통계 정보 조회
        stats = embedding_manager.get_version_statistics(version_id)
        logger.info(f"  Chunks: {stats.get('chunk_count', 0)}")
        logger.info(f"  Documents: {stats.get('document_count', 0)}")
        
        # FAISS 버전 이름 생성
        faiss_version_name = f"{version_name}-{chunking_strategy}"
        
        # FAISS 버전이 이미 존재하는지 확인
        existing_path = faiss_manager.get_version_path(faiss_version_name)
        if existing_path:
            logger.info(f"  FAISS version already exists: {faiss_version_name}")
            continue
        
        # FAISS 버전 생성 (인덱스는 나중에 빌드)
        try:
            chunking_config = stats.get('metadata', {}).get('chunking_config', {}) if isinstance(stats.get('metadata'), dict) else {}
            embedding_config = {
                'model': model_name,
                'dimension': 768  # 기본값, 실제로는 데이터베이스에서 조회해야 함
            }
            
            version_path = faiss_manager.create_version(
                version_name=faiss_version_name,
                embedding_version_id=version_id,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                embedding_config=embedding_config,
                document_count=stats.get('document_count', 0),
                total_chunks=stats.get('chunk_count', 0),
                status='inactive'  # 인덱스 빌드 전까지는 inactive
            )
            
            logger.info(f"  ✓ Created FAISS version: {faiss_version_name}")
            logger.info(f"  Path: {version_path}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to create FAISS version: {e}")
            return False
    
    return True


def test_index_building(db_path: str, vector_store_base: str):
    """인덱스 빌드 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: FAISS Index Building")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    embedding_manager = EmbeddingVersionManager(db_path)
    
    # 활성 버전 조회
    active_versions = embedding_manager.get_all_active_versions()
    if not active_versions:
        logger.warning("No active embedding versions found.")
        return False
    
    for version_info in active_versions:
        version_id = version_info['id']
        version_name = version_info['version_name']
        chunking_strategy = version_info['chunking_strategy']
        faiss_version_name = f"{version_name}-{chunking_strategy}"
        
        logger.info(f"\nBuilding index for version: {faiss_version_name}")
        
        # 검색 엔진 초기화 (외부 인덱스 사용 안 함)
        try:
            engine = SemanticSearchEngineV2(
                db_path=db_path,
                use_external_index=False
            )
            
            # 인덱스 빌드
            logger.info("  Building FAISS index...")
            success = engine._build_faiss_index_sync(
                embedding_version_id=version_id,
                faiss_version_name=faiss_version_name
            )
            
            if success:
                logger.info(f"  ✓ Index built successfully for {faiss_version_name}")
                
                # 버전을 활성으로 설정
                faiss_manager.set_active_version(faiss_version_name)
                logger.info(f"  ✓ Set as active version")
            else:
                logger.error(f"  ✗ Failed to build index")
                return False
                
        except Exception as e:
            logger.error(f"  ✗ Error building index: {e}", exc_info=True)
            return False
    
    return True


def test_version_switching(db_path: str, vector_store_base: str):
    """버전 전환 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Version Switching")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    versions = faiss_manager.list_versions()
    
    if len(versions) < 2:
        logger.warning("Need at least 2 versions to test switching")
        return True
    
    logger.info(f"Found {len(versions)} versions")
    
    for version in versions[:2]:  # 처음 2개만 테스트
        version_name = version.get('version', 'unknown')
        logger.info(f"\nSwitching to version: {version_name}")
        
        if faiss_manager.set_active_version(version_name):
            active = faiss_manager.get_active_version()
            if active == version_name:
                logger.info(f"  ✓ Successfully switched to {version_name}")
            else:
                logger.error(f"  ✗ Active version mismatch: expected {version_name}, got {active}")
                return False
        else:
            logger.error(f"  ✗ Failed to switch to {version_name}")
            return False
    
    return True


def test_search_with_version(db_path: str, vector_store_base: str):
    """버전별 검색 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Search with Version")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    active_version = faiss_manager.get_active_version()
    
    if not active_version:
        logger.warning("No active version found")
        return False
    
    logger.info(f"Active version: {active_version}")
    
    # 검색 엔진 초기화
    try:
        engine = SemanticSearchEngineV2(db_path=db_path)
        
        # 테스트 쿼리
        test_queries = [
            "전세금 반환 보증",
            "계약 해지",
            "손해배상"
        ]
        
        for query in test_queries:
            logger.info(f"\nSearching: '{query}'")
            
            # 활성 버전으로 검색
            results = engine.search(
                query=query,
                k=5,
                faiss_version=active_version
            )
            
            logger.info(f"  Found {len(results)} results")
            if results:
                for i, result in enumerate(results[:3], 1):
                    score = result.get('score', 0.0)
                    text_preview = result.get('text', '')[:50]
                    logger.info(f"    {i}. Score: {score:.4f}, Text: {text_preview}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during search test: {e}", exc_info=True)
        return False


def test_multi_version_search(db_path: str, vector_store_base: str):
    """멀티 버전 검색 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 5: Multi-Version Search")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    versions = faiss_manager.list_versions()
    
    if len(versions) < 2:
        logger.warning("Need at least 2 versions to test multi-version search")
        return True
    
    version_names = [v.get('version') for v in versions[:2]]
    logger.info(f"Testing with versions: {version_names}")
    
    try:
        engine = SemanticSearchEngineV2(db_path=db_path)
        
        query = "전세금 반환 보증"
        logger.info(f"\nSearching: '{query}'")
        
        results = engine.search_multiple_versions(
            query=query,
            versions=version_names,
            k=5
        )
        
        for version_name, version_results in results.items():
            logger.info(f"\n  Version {version_name}: {len(version_results)} results")
            if version_results:
                for i, result in enumerate(version_results[:2], 1):
                    similarity = result.get('similarity', 0.0)
                    logger.info(f"    {i}. Similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during multi-version search test: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FAISS version management with real data")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--vector-store-base", default="data/vector_store", help="Vector store base path")
    parser.add_argument("--skip-creation", action="store_true", help="Skip version creation test")
    parser.add_argument("--skip-building", action="store_true", help="Skip index building test")
    parser.add_argument("--skip-switching", action="store_true", help="Skip version switching test")
    parser.add_argument("--skip-search", action="store_true", help="Skip search test")
    parser.add_argument("--skip-multi-search", action="store_true", help="Skip multi-version search test")
    
    args = parser.parse_args()
    
    db_path = args.db
    vector_store_base = args.vector_store_base
    
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    logger.info(f"Database: {db_path}")
    logger.info(f"Vector store base: {vector_store_base}")
    
    results = {}
    
    # Test 1: Version Creation
    if not args.skip_creation:
        results['creation'] = test_version_creation(db_path, vector_store_base)
    
    # Test 2: Index Building
    if not args.skip_building:
        results['building'] = test_index_building(db_path, vector_store_base)
    
    # Test 3: Version Switching
    if not args.skip_switching:
        results['switching'] = test_version_switching(db_path, vector_store_base)
    
    # Test 4: Search with Version
    if not args.skip_search:
        results['search'] = test_search_with_version(db_path, vector_store_base)
    
    # Test 5: Multi-Version Search
    if not args.skip_multi_search:
        results['multi_search'] = test_multi_version_search(db_path, vector_store_base)
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values()) if results else False
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

