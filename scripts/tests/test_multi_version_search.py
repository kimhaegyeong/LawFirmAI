"""
MultiVersionSearch 단위 테스트
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from faiss_version_manager import FAISSVersionManager
from multi_version_search import MultiVersionSearch


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


@pytest.fixture
def multi_search(version_manager):
    """MultiVersionSearch 인스턴스 생성"""
    return MultiVersionSearch(version_manager)


def test_load_version(multi_search, version_manager):
    """버전 로드 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    version_data = multi_search.load_version("v1.0.0-test")
    assert version_data is None or isinstance(version_data, dict)


def test_search_all_versions(multi_search, version_manager):
    """여러 버전 검색 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    query_vector = np.random.rand(768).astype('float32')
    results = multi_search.search_all_versions(
        query_vector=query_vector,
        versions=["v1.0.0-test"],
        k=5
    )
    
    assert isinstance(results, dict)
    assert "v1.0.0-test" in results


def test_ensemble_search(multi_search, version_manager):
    """앙상블 검색 테스트"""
    version_manager.create_version(
        version_name="v1.0.0-test",
        embedding_version_id=1,
        chunking_strategy="standard",
        chunking_config={},
        embedding_config={"model": "test", "dimension": 768}
    )
    
    query_vector = np.random.rand(768).astype('float32')
    results = multi_search.ensemble_search(
        query_vector=query_vector,
        versions=["v1.0.0-test"],
        weights=[1.0],
        k=5
    )
    
    assert isinstance(results, list)


def test_compare_results(multi_search, version_manager):
    """버전 비교 테스트"""
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
    
    query_vector = np.random.rand(768).astype('float32')
    comparison = multi_search.compare_results(
        query_vector=query_vector,
        version1="v1.0.0-test1",
        version2="v1.0.0-test2",
        k=10
    )
    
    assert isinstance(comparison, dict)
    assert "version1" in comparison
    assert "version2" in comparison


def test_clear_cache(multi_search):
    """캐시 정리 테스트"""
    multi_search.clear_cache()
    assert len(multi_search.loaded_indices) == 0

