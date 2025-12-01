"""
재임베딩된 데이터로 검색 시 활성 버전 자동 로드 테스트

SemanticSearchEngineV2가 초기화 시 및 검색 시 활성 FAISS 버전을 자동으로 로드하는지 검증합니다.
"""
import logging
import sys
import tempfile
import shutil
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "utils"))

from scripts.utils.faiss_version_manager import FAISSVersionManager
from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_active_version_auto_load_on_init(db_path: str, vector_store_base: str):
    """
    초기화 시 활성 버전 자동 로드 테스트
    
    재임베딩 후 활성 버전이 설정되어 있을 때,
    SemanticSearchEngineV2 초기화 시 자동으로 활성 버전을 로드하는지 확인합니다.
    """
    logger.info("=" * 80)
    logger.info("Test: Active Version Auto-Load on Initialization")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    embedding_manager = EmbeddingVersionManager(db_path)
    
    # 활성 버전 확인
    active_version = faiss_manager.get_active_version()
    if not active_version:
        logger.warning("No active FAISS version found. Skipping test.")
        return False
    
    logger.info(f"Active FAISS version: {active_version}")
    
    # 검색 엔진 초기화 (활성 버전 자동 로드 확인)
    # 외부 인덱스 사용을 명시적으로 비활성화
    try:
        # Config에서 외부 인덱스 설정을 무시하도록 강제
        import os
        original_use_external = os.environ.get('USE_EXTERNAL_VECTOR_STORE', None)
        os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'false'
        
        try:
            engine = SemanticSearchEngineV2(
                db_path=db_path,
                use_external_index=False
            )
        finally:
            # 환경 변수 복원
            if original_use_external is not None:
                os.environ['USE_EXTERNAL_VECTOR_STORE'] = original_use_external
            elif 'USE_EXTERNAL_VECTOR_STORE' in os.environ:
                del os.environ['USE_EXTERNAL_VECTOR_STORE']
        
        # 초기화 시 활성 버전이 로드되었는지 확인
        if engine.current_faiss_version == active_version:
            logger.info(f"✓ Active version '{active_version}' loaded automatically on initialization")
        else:
            logger.warning(f"⚠ Current version '{engine.current_faiss_version}' != Active version '{active_version}'")
            # 인덱스가 로드되었는지 확인
            if engine.index is not None:
                logger.info(f"  But index is loaded ({engine.index.ntotal} vectors)")
            else:
                logger.error("  ✗ Index not loaded")
                return False
        
        # 인덱스가 실제로 로드되었는지 확인
        if engine.index is None:
            logger.error("✗ Index not loaded during initialization")
            return False
        
        logger.info(f"✓ Index loaded successfully: {engine.index.ntotal} vectors")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during initialization: {e}", exc_info=True)
        return False


def test_active_version_auto_load_on_search(db_path: str, vector_store_base: str):
    """
    검색 시 활성 버전 자동 로드 테스트
    
    faiss_version 파라미터를 지정하지 않았을 때,
    활성 버전을 자동으로 로드하여 검색하는지 확인합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test: Active Version Auto-Load on Search")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    
    # 활성 버전 확인
    active_version = faiss_manager.get_active_version()
    if not active_version:
        logger.warning("No active FAISS version found. Skipping test.")
        return False
    
    logger.info(f"Active FAISS version: {active_version}")
    
    # 검색 엔진 초기화 (인덱스 없이)
    try:
        # Config에서 외부 인덱스 설정을 무시하도록 강제
        import os
        original_use_external = os.environ.get('USE_EXTERNAL_VECTOR_STORE', None)
        os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'false'
        
        try:
            # 임시로 인덱스를 None으로 설정하여 검색 시 로드되도록 함
            engine = SemanticSearchEngineV2(
                db_path=db_path,
                use_external_index=False
            )
        finally:
            # 환경 변수 복원
            if original_use_external is not None:
                os.environ['USE_EXTERNAL_VECTOR_STORE'] = original_use_external
            elif 'USE_EXTERNAL_VECTOR_STORE' in os.environ:
                del os.environ['USE_EXTERNAL_VECTOR_STORE']
        
        # 인덱스를 None으로 설정 (검색 시 재로드 테스트)
        original_index = engine.index
        engine.index = None
        engine.current_faiss_version = None
        
        logger.info("Index cleared, testing search with auto-load...")
        
        # 검색 실행 (faiss_version 파라미터 없이)
        test_query = "전세금 반환"
        results = engine.search(
            query=test_query,
            k=5,
            # faiss_version 파라미터를 지정하지 않음 -> 활성 버전 자동 로드
        )
        
        # 활성 버전이 로드되었는지 확인
        if engine.current_faiss_version == active_version:
            logger.info(f"✓ Active version '{active_version}' loaded automatically during search")
        else:
            logger.warning(f"⚠ Current version '{engine.current_faiss_version}' != Active version '{active_version}'")
        
        # 검색 결과 확인
        if results:
            logger.info(f"✓ Search successful: {len(results)} results found")
            logger.info(f"  Top result score: {results[0].get('score', 0.0):.4f}")
        else:
            logger.warning("⚠ No search results returned")
        
        # 인덱스가 로드되었는지 확인
        if engine.index is None:
            logger.error("✗ Index not loaded during search")
            return False
        
        logger.info(f"✓ Index loaded during search: {engine.index.ntotal} vectors")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during search test: {e}", exc_info=True)
        return False


def test_version_switch_and_auto_load(db_path: str, vector_store_base: str):
    """
    버전 전환 후 자동 로드 테스트
    
    활성 버전을 변경한 후, 새로운 검색 엔진 인스턴스가
    새로운 활성 버전을 자동으로 로드하는지 확인합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test: Version Switch and Auto-Load")
    logger.info("=" * 80)
    
    faiss_manager = FAISSVersionManager(vector_store_base)
    
    # 사용 가능한 버전 목록 조회
    versions = faiss_manager.list_versions()
    if len(versions) < 2:
        logger.warning("Need at least 2 versions for switch test. Skipping.")
        return False
    
    logger.info(f"Available versions: {[v['version'] for v in versions]}")
    
    # 첫 번째 버전을 활성으로 설정
    version1 = versions[0]['version']
    faiss_manager.set_active_version(version1)
    logger.info(f"Set active version to: {version1}")
    
    # 검색 엔진 초기화
    import os
    original_use_external = os.environ.get('USE_EXTERNAL_VECTOR_STORE', None)
    os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'false'
    
    try:
        engine1 = SemanticSearchEngineV2(db_path=db_path, use_external_index=False)
        if engine1.current_faiss_version != version1:
            logger.error(f"✗ Failed to load version {version1}")
            if original_use_external is not None:
                os.environ['USE_EXTERNAL_VECTOR_STORE'] = original_use_external
            elif 'USE_EXTERNAL_VECTOR_STORE' in os.environ:
                del os.environ['USE_EXTERNAL_VECTOR_STORE']
            return False
        logger.info(f"✓ Engine 1 loaded version: {engine1.current_faiss_version}")
        
        # 두 번째 버전으로 전환
        version2 = versions[1]['version']
        faiss_manager.set_active_version(version2)
        logger.info(f"Switched active version to: {version2}")
        
        # 새로운 검색 엔진 인스턴스 생성 (새 활성 버전 자동 로드)
        engine2 = SemanticSearchEngineV2(db_path=db_path, use_external_index=False)
        if engine2.current_faiss_version != version2:
            logger.error(f"✗ Failed to load new active version {version2}")
            return False
        logger.info(f"✓ Engine 2 loaded new active version: {engine2.current_faiss_version}")
        
        # 두 엔진이 다른 버전을 사용하는지 확인
        if engine1.current_faiss_version == engine2.current_faiss_version:
            logger.warning("⚠ Both engines using same version (expected after switch)")
    finally:
        # 환경 변수 복원
        if original_use_external is not None:
            os.environ['USE_EXTERNAL_VECTOR_STORE'] = original_use_external
        elif 'USE_EXTERNAL_VECTOR_STORE' in os.environ:
            del os.environ['USE_EXTERNAL_VECTOR_STORE']
    
    return True


def main():
    """메인 테스트 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test active version auto-load functionality")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/lawfirm_v2.db",
        help="Path to lawfirm_v2.db"
    )
    parser.add_argument(
        "--vector-store-base",
        type=str,
        default="data/vector_store",
        help="Base path for vector store"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    vector_store_base = Path(args.vector_store_base)
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False
    
    if not vector_store_base.exists():
        logger.error(f"Vector store base not found: {vector_store_base}")
        return False
    
    logger.info(f"Database: {db_path}")
    logger.info(f"Vector Store: {vector_store_base}")
    logger.info("")
    
    # 테스트 실행
    results = []
    
    results.append(("Active Version Auto-Load on Init", 
                   test_active_version_auto_load_on_init(str(db_path), str(vector_store_base))))
    
    results.append(("Active Version Auto-Load on Search", 
                   test_active_version_auto_load_on_search(str(db_path), str(vector_store_base))))
    
    results.append(("Version Switch and Auto-Load", 
                   test_version_switch_and_auto_load(str(db_path), str(vector_store_base))))
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("Test Results Summary")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

