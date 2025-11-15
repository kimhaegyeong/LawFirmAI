"""
FAISSVersionManager 단위 테스트
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from faiss_version_manager import FAISSVersionManager


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def version_manager(temp_dir):
    """FAISSVersionManager 인스턴스 생성"""
    return FAISSVersionManager(str(temp_dir))


def test_create_version(version_manager):
    """버전 생성 테스트"""
    version_path = version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={"chunk_size": 1000, "chunk_overlap": 200},
        embedding_config={"model": "test-model", "dimension": 768},
        document_count=100,
        total_chunks=1000,
        status="active"
    )
    
    assert version_path.exists()
    assert (version_path / "version_info.json").exists()
    
    with open(version_path / "version_info.json", 'r', encoding='utf-8') as f:
        info = json.load(f)
        assert info["version"] == "v1.0.0-test"
        assert info["embedding_version_id"] == 1
        assert info["chunking_strategy"] == "standard"


def test_set_active_version(version_manager):
    """활성 버전 설정 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    assert version_manager.set_active_version("v1.0.0-test")
    assert version_manager.get_active_version() == "v1.0.0-test"


def test_get_version_path(version_manager):
    """버전 경로 조회 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    path = version_manager.get_version_path("v1.0.0-test")
    assert path is not None
    assert path.exists()


def test_list_versions(version_manager):
    """버전 목록 조회 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test1",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    version_manager.create_version(
        version_name="v1.0.0-test2",
        embedding_version_id=2,
        chunking_strategy="dynamic",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    versions = version_manager.list_versions()
    assert len(versions) == 2
    assert any(v["version"] == "v1.0.0-test1" for v in versions)
    assert any(v["version"] == "v1.0.0-test2" for v in versions)


def test_copy_version(version_manager):
    """버전 복사 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    copied_path = version_manager.copy_version("v1.0.0-test", "v1.0.0-copy")
    assert copied_path is not None
    assert copied_path.exists()
    assert (copied_path / "version_info.json").exists()


def test_delete_version(version_manager):
    """버전 삭제 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    assert version_manager.delete_version("v1.0.0-test", force=True)
    assert version_manager.get_version_path("v1.0.0-test") is None


def test_get_version_info(version_manager):
    """버전 정보 조회 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={"chunk_size": 1000},
        embedding_config={"model": "test", "dimension": 768},
        document_count=100,
        total_chunks=1000
    )
    
    info = version_manager.get_version_info("v1.0.0-test")
    assert info is not None
    assert info["version"] == "v1.0.0-test"
    assert info["document_count"] == 100
    assert info["total_chunks"] == 1000

