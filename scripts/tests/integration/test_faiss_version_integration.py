"""
FAISS 버전 관리 통합 테스트

버전 생성, 전환, 멀티 버전 검색, 마이그레이션, 성능 모니터링을 통합적으로 테스트합니다.
"""
import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts" / "utils"))


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_db(temp_dir):
    """임시 데이터베이스 생성"""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_name TEXT UNIQUE,
            chunking_strategy TEXT,
            model_name TEXT,
            description TEXT,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            source_id INTEGER,
            text TEXT,
            chunk_index INTEGER,
            embedding_version_id INTEGER,
            chunking_strategy TEXT,
            chunk_size_category TEXT,
            chunk_group_id TEXT,
            FOREIGN KEY (embedding_version_id) REFERENCES embedding_versions(id)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            model TEXT,
            vector BLOB,
            dim INTEGER,
            FOREIGN KEY (chunk_id) REFERENCES text_chunks(id)
        )
    """)
    
    conn.commit()
    conn.close()
    return str(db_path)


def test_version_creation_and_switching(temp_dir, temp_db):
    """버전 생성 및 전환 통합 테스트"""
    from faiss_version_manager import FAISSVersionManager
    from embedding_version_manager import EmbeddingVersionManager
    
    faiss_manager = FAISSVersionManager(str(temp_dir))
    embedding_manager = EmbeddingVersionManager(temp_db)
    
    # Embedding 버전 생성
    version_id = embedding_manager.register_version(
        version_name="v1.0.0-test",
        chunking_strategy="standard",
        model_name="test-model",
        set_active=True,
        create_faiss_version=True,
        faiss_version_manager=faiss_manager
    )
    
    assert version_id > 0
    
    # FAISS 버전 확인
    faiss_version_name = f"v1.0.0-test-standard"
    version_path = faiss_manager.get_version_path(faiss_version_name)
    assert version_path is not None
    
    # 활성 버전 확인
    active_embedding = embedding_manager.get_active_version("standard")
    assert active_embedding is not None
    assert active_embedding["id"] == version_id
    
    active_faiss = faiss_manager.get_active_version()
    assert active_faiss == faiss_version_name or active_faiss is None


def test_multi_version_search_integration(temp_dir, temp_db):
    """멀티 버전 검색 통합 테스트"""
    from faiss_version_manager import FAISSVersionManager
    from multi_version_search import MultiVersionSearch
    import numpy as np
    
    faiss_manager = FAISSVersionManager(str(temp_dir))
    
    # 여러 버전 생성
    faiss_manager.create_version(
        version_name="v1.0.0-standard",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    faiss_manager.create_version(
        version_name="v1.0.0-dynamic",
        embedding_version_id=2,
        chunking_strategy="dynamic",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    multi_search = MultiVersionSearch(faiss_manager)
    
    query_vector = np.random.rand(768).astype('float32')
    results = multi_search.search_all_versions(
        query_vector=query_vector,
        versions=["v1.0.0-standard", "v1.0.0-dynamic"],
        k=5
    )
    
    assert isinstance(results, dict)
    assert "v1.0.0-standard" in results
    assert "v1.0.0-dynamic" in results


def test_migration_manager_integration(temp_dir, temp_db):
    """마이그레이션 관리자 통합 테스트"""
    from faiss_version_manager import FAISSVersionManager
    from embedding_version_manager import EmbeddingVersionManager
    from faiss_migration_manager import FAISSMigrationManager
    
    faiss_manager = FAISSVersionManager(str(temp_dir))
    embedding_manager = EmbeddingVersionManager(temp_db)
    migration_manager = FAISSMigrationManager(
        faiss_manager,
        embedding_manager,
        temp_db
    )
    
    # 마이그레이션 상태 조회
    status = migration_manager.get_migration_status()
    assert isinstance(status, dict)
    assert "total" in status
    
    # 원본 문서 조회 테스트 (테이블이 없는 경우)
    doc = migration_manager.get_original_document("statute_article", 1)
    assert doc is None or isinstance(doc, dict)


def test_performance_monitoring_integration(temp_dir):
    """성능 모니터링 통합 테스트"""
    from version_performance_monitor import VersionPerformanceMonitor
    
    monitor = VersionPerformanceMonitor(str(temp_dir / "performance_logs"))
    
    # 성능 로깅
    monitor.log_search(
        version="v1.0.0-test",
        query_id="test_query_1",
        latency_ms=45.2,
        relevance_score=0.85,
        user_feedback="positive"
    )
    
    monitor.log_search(
        version="v1.0.0-test",
        query_id="test_query_2",
        latency_ms=38.5,
        relevance_score=0.92,
        user_feedback="positive"
    )
    
    # 메트릭 조회
    metrics = monitor.get_version_metrics("v1.0.0-test")
    assert metrics is not None
    assert metrics["total_queries"] == 2
    assert metrics["feedback_positive"] == 2
    
    # 성능 비교 (같은 버전과 비교)
    comparison = monitor.compare_performance("v1.0.0-test", "v1.0.0-test")
    assert "error" not in comparison or comparison.get("error") is None


def test_integrated_version_manager(temp_dir, temp_db):
    """통합 버전 관리자 테스트"""
    from version_manager_integrated import IntegratedVersionManager
    
    integrated_manager = IntegratedVersionManager(
        db_path=temp_db,
        vector_store_base=str(temp_dir)
    )
    
    # 통합 버전 생성
    version_info = integrated_manager.create_version(
        version_name="v1.0.0-integrated",
        chunking_strategy="standard",
        model_name="test-model",
        chunking_config={"chunk_size": 1000},
        embedding_config={"model": "test-model", "dimension": 768},
        set_active=True
    )
    
    assert version_info["embedding_version_id"] > 0
    assert version_info["faiss_version_name"] == "v1.0.0-integrated-standard"
    
    # 버전 목록 조회
    versions = integrated_manager.list_versions()
    assert "embedding_versions" in versions
    assert "faiss_versions" in versions
    
    # 버전 정보 조회
    info = integrated_manager.get_version_info(
        embedding_version_id=version_info["embedding_version_id"],
        faiss_version_name=version_info["faiss_version_name"]
    )
    assert "embedding_version" in info or "faiss_version" in info

