# FAISS 벡터 임베딩 및 청킹 버전 관리 가이드

## 개요

FAISS 기반 벡터 임베딩과 청킹 전략을 버전별로 관리할 수 있는 시스템입니다. 파일 기반 FAISS 인덱스 버전 관리와 기존 SQLite 기반 EmbeddingVersionManager를 통합하여, 버전별 인덱스 저장, 멀티 버전 검색, A/B 테스트, 성능 모니터링 기능을 제공합니다.

## 시스템 구조

### 디렉토리 구조

```
data/
├── vector_store/              # FAISS 버전 저장소
│   ├── v1.0.0-standard/
│   │   ├── index.faiss       # FAISS 인덱스 파일
│   │   ├── metadata.pkl      # 청크 메타데이터
│   │   ├── id_mapping.json   # FAISS ID → chunk_id 매핑
│   │   └── version_info.json # 버전 정보
│   ├── v1.0.0-dynamic/
│   └── active_version.txt    # 현재 활성 버전
├── performance_logs/         # 성능 모니터링 로그
└── backups/
    └── faiss_versions/       # 백업 파일
```

### 버전 정보 스키마

`version_info.json` 구조:

```json
{
    "version": "v1.0.0-standard",
    "embedding_version_id": 1,
    "chunking_strategy": "standard",
    "created_at": "2024-11-15T10:00:00Z",
    "chunking_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "embedding_config": {
        "model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "dimension": 768
    },
    "document_count": 1500,
    "total_chunks": 15000,
    "status": "active"
}
```

## 주요 컴포넌트

### 1. FAISSVersionManager

파일 기반 FAISS 인덱스 버전 관리 클래스입니다.

**주요 메서드:**
- `create_version()`: 새 버전 생성
- `set_active_version()`: 활성 버전 설정
- `get_version_path()`: 버전 경로 조회
- `list_versions()`: 버전 목록 조회
- `copy_version()`: 버전 복사
- `delete_version()`: 버전 삭제
- `save_index()`: FAISS 인덱스 저장
- `load_index()`: FAISS 인덱스 로드

**사용 예시:**

```python
from scripts.utils.faiss_version_manager import FAISSVersionManager

manager = FAISSVersionManager("data/vector_store")

# 버전 생성
version_path = manager.create_version(
    version_name="v1.0.0-standard",
    embedding_version_id=1,
    chunking_strategy="standard",
    chunking_config={"chunk_size": 1000, "chunk_overlap": 200},
    embedding_config={"model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "dimension": 768},
    document_count=1500,
    total_chunks=15000,
    status="active"
)

# 활성 버전 설정
manager.set_active_version("v1.0.0-standard")

# 버전 목록 조회
versions = manager.list_versions()
```

### 2. SemanticSearchEngineV2 통합

검색 엔진에 FAISS 버전 관리가 통합되어 있습니다.

**주요 기능:**
- 버전별 인덱스 자동 저장
- 버전별 인덱스 로드
- 버전 지정 검색

**사용 예시:**

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

engine = SemanticSearchEngineV2(db_path="data/lawfirm_v2.db")

# 특정 버전으로 검색
results = engine.search(
    query="전세금 반환 보증",
    k=10,
    faiss_version="v1.0.0-standard"
)

# 여러 버전 동시 검색
multi_results = engine.search_multiple_versions(
    query="전세금 반환 보증",
    versions=["v1.0.0-standard", "v1.0.0-dynamic"],
    k=10
)

# 앙상블 검색
ensemble_results = engine.ensemble_search(
    query="전세금 반환 보증",
    versions=["v1.0.0-standard", "v1.0.0-dynamic"],
    weights=[0.6, 0.4],
    k=10
)
```

### 3. MultiVersionSearch

여러 FAISS 버전을 동시에 검색하여 결과를 비교하거나 앙상블합니다.

**주요 메서드:**
- `load_version()`: 버전을 메모리에 로드
- `search_all_versions()`: 모든 버전에서 동시 검색
- `ensemble_search()`: 여러 버전 결과 앙상블
- `compare_results()`: 두 버전의 검색 결과 비교

**사용 예시:**

```python
from scripts.utils.multi_version_search import MultiVersionSearch
from scripts.utils.faiss_version_manager import FAISSVersionManager
import numpy as np

manager = FAISSVersionManager("data/vector_store")
multi_search = MultiVersionSearch(manager)

# 쿼리 벡터 생성 (예시)
query_vector = np.array([...])  # 실제로는 임베딩 모델로 생성

# 모든 버전에서 검색
results = multi_search.search_all_versions(
    query_vector=query_vector,
    versions=["v1.0.0-standard", "v1.0.0-dynamic"],
    k=10
)

# 앙상블 검색
ensemble_results = multi_search.ensemble_search(
    query_vector=query_vector,
    versions=["v1.0.0-standard", "v1.0.0-dynamic"],
    weights=[0.6, 0.4],
    k=10
)

# 버전 비교
comparison = multi_search.compare_results(
    query_vector=query_vector,
    version1="v1.0.0-standard",
    version2="v1.0.0-dynamic",
    k=10
)
```

### 4. FAISSMigrationManager

버전 간 점진적 마이그레이션을 관리합니다.

**주요 메서드:**
- `migrate_documents()`: 문서별 점진적 마이그레이션
- `rollback_migration()`: 마이그레이션 롤백
- `get_migration_status()`: 마이그레이션 진행 상태 조회

**사용 예시:**

```python
from scripts.utils.faiss_migration_manager import FAISSMigrationManager

migration_manager = FAISSMigrationManager(
    faiss_version_manager,
    embedding_version_manager,
    db_path="data/lawfirm_v2.db"
)

# 문서 마이그레이션
result = await migration_manager.migrate_documents(
    source_version="v1.0.0-standard",
    target_version="v2.0.0-dynamic",
    document_ids=[("statute_article", 1), ("case_paragraph", 2)],
    rechunk_fn=rechunk_function,
    reembed_fn=reembed_function,
    batch_size=10
)

# 마이그레이션 상태 조회
status = migration_manager.get_migration_status("v2.0.0-dynamic")
```

### 5. VersionPerformanceMonitor

버전별 검색 성능을 추적하고 비교합니다.

**주요 메서드:**
- `log_search()`: 검색 성능 로깅
- `compare_performance()`: 버전별 성능 비교
- `get_version_metrics()`: 특정 버전의 메트릭 조회

**사용 예시:**

```python
from scripts.utils.version_performance_monitor import VersionPerformanceMonitor

monitor = VersionPerformanceMonitor("data/performance_logs")

# 검색 성능 로깅
monitor.log_search(
    version="v1.0.0-standard",
    query_id="query_123",
    latency_ms=45.2,
    relevance_score=0.85,
    user_feedback="positive"
)

# 성능 비교
comparison = monitor.compare_performance("v1.0.0-standard", "v1.0.0-dynamic")
print(f"Latency improvement: {comparison['latency_improvement_percent']:.2f}%")
print(f"Relevance improvement: {comparison['relevance_improvement_percent']:.2f}%")
```

### 6. FAISSBackupManager

FAISS 버전을 백업하고 복원합니다.

**주요 메서드:**
- `backup_version()`: 버전 전체를 압축 백업
- `restore_version()`: 백업에서 버전 복원
- `cleanup_old_backups()`: 오래된 백업 정리

**사용 예시:**

```python
from scripts.utils.faiss_backup_manager import FAISSBackupManager

backup_manager = FAISSBackupManager(
    backup_path="data/backups/faiss_versions",
    faiss_version_manager=manager
)

# 버전 백업
backup_file = backup_manager.backup_version("v1.0.0-standard")

# 버전 복원
backup_manager.restore_version(backup_file)

# 오래된 백업 정리
backup_manager.cleanup_old_backups(keep_recent=5)
```

### 7. IntegratedVersionManager

SQLite 버전 관리와 FAISS 버전 관리를 통합합니다.

**주요 메서드:**
- `create_version()`: 새 버전 생성 (SQLite + FAISS)
- `switch_version()`: 버전 전환
- `delete_version()`: 버전 삭제
- `list_versions()`: 모든 버전 목록 조회

**사용 예시:**

```python
from scripts.utils.version_manager_integrated import IntegratedVersionManager

integrated_manager = IntegratedVersionManager(
    db_path="data/lawfirm_v2.db",
    vector_store_base="data/vector_store"
)

# 통합 버전 생성
version_info = integrated_manager.create_version(
    version_name="v1.0.0-standard",
    chunking_strategy="standard",
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    chunking_config={"chunk_size": 1000, "chunk_overlap": 200},
    embedding_config={"model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "dimension": 768},
    set_active=True
)

# 버전 전환
integrated_manager.switch_version(
    embedding_version_id=1,
    faiss_version_name="v1.0.0-standard"
)
```

## CLI 도구

### faiss_version_switcher.py

FAISS 버전 관리를 위한 CLI 도구입니다.

**사용 예시:**

```bash
# 버전 목록 조회
python scripts/utils/faiss_version_switcher.py list

# 버전 통계 조회
python scripts/utils/faiss_version_switcher.py stats v1.0.0-standard

# 버전 전환
python scripts/utils/faiss_version_switcher.py switch v1.0.0-dynamic

# 버전 비교
python scripts/utils/faiss_version_switcher.py compare v1.0.0-standard v1.0.0-dynamic

# 버전 삭제
python scripts/utils/faiss_version_switcher.py delete v1.0.0-standard --force

# 버전 백업
python scripts/utils/faiss_version_switcher.py backup v1.0.0-standard

# 성능 통계 조회
python scripts/utils/faiss_version_switcher.py performance --version v1.0.0-standard
```

## 워크플로우

### 1. 초기 설정

```python
from scripts.utils.faiss_version_manager import FAISSVersionManager
from scripts.utils.embedding_version_manager import EmbeddingVersionManager

faiss_manager = FAISSVersionManager("data/vector_store")
embedding_manager = EmbeddingVersionManager("data/lawfirm_v2.db")
```

### 2. 프로덕션 버전 생성

```python
# EmbeddingVersionManager로 버전 등록
version_id = embedding_manager.register_version(
    version_name="prod_v1",
    chunking_strategy="standard",
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    set_active=True,
    create_faiss_version=True,
    faiss_version_manager=faiss_manager
)

# FAISS 인덱스 빌드 (SemanticSearchEngineV2에서 자동 처리)
```

### 3. 실험 버전 생성

```python
# 프로덕션 버전 복사
faiss_manager.copy_version("prod_v1", "exp_v2", update_status="experimental")

# 새 청킹 전략으로 재임베딩
# ... 재임베딩 로직 ...
```

### 4. A/B 테스트 실행

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

engine = SemanticSearchEngineV2(db_path="data/lawfirm_v2.db")

# 여러 버전에서 동시 검색
results = engine.search_multiple_versions(
    query="전세금 반환 보증",
    versions=["prod_v1", "exp_v2"],
    k=10
)
```

### 5. 성능 모니터링

```python
from scripts.utils.version_performance_monitor import VersionPerformanceMonitor
import time

monitor = VersionPerformanceMonitor()

# 검색 성능 로깅
start = time.time()
results = engine.search("전세금 반환 보증", faiss_version="exp_v2")
latency = (time.time() - start) * 1000

monitor.log_search(
    version="exp_v2",
    query_id="query_123",
    latency_ms=latency,
    relevance_score=0.85
)

# 성능 비교
comparison = monitor.compare_performance("prod_v1", "exp_v2")
if comparison["relevance_improvement_percent"] > 5:
    print("성능 개선 확인: 프로덕션 전환 고려")
```

### 6. 프로덕션 전환

```python
# 백업 후 전환
from scripts.utils.faiss_backup_manager import FAISSBackupManager

backup_manager = FAISSBackupManager(
    backup_path="data/backups/faiss_versions",
    faiss_version_manager=faiss_manager
)
backup_manager.backup_version("prod_v1")

# 버전 전환
faiss_manager.set_active_version("exp_v2")
embedding_manager.set_active_version(new_version_id)
```

## 주요 장점

1. **독립성**: 각 버전이 완전히 독립적으로 동작
2. **빠른 롤백**: 파일 기반이라 즉시 이전 버전으로 전환 가능
3. **A/B 테스트**: 여러 버전을 동시에 운영하며 비교
4. **오프라인 운영**: 클라우드 없이 로컬에서 완전한 제어
5. **비용 효율**: 별도의 서비스 비용 없음

## 주의사항

1. **디스크 공간**: 각 버전은 전체 인덱스를 저장하므로 디스크 공간이 필요합니다.
2. **메모리 사용**: 여러 버전을 동시에 메모리에 로드하면 메모리 사용량이 증가합니다.
3. **백업**: 중요한 버전은 정기적으로 백업하세요.
4. **성능 모니터링**: A/B 테스트 시 충분한 데이터를 수집한 후 판단하세요.

## 문제 해결

### 버전을 찾을 수 없음

```python
# 버전 목록 확인
versions = faiss_manager.list_versions()
print(versions)

# 활성 버전 확인
active = faiss_manager.get_active_version()
print(active)
```

### 인덱스 로드 실패

```python
# 버전 정보 확인
info = faiss_manager.get_version_info("v1.0.0-standard")
print(info)

# 인덱스 파일 확인
version_path = faiss_manager.get_version_path("v1.0.0-standard")
index_file = version_path / "index.faiss"
print(f"Index file exists: {index_file.exists()}")
```

### 성능 문제

```python
# 성능 메트릭 확인
monitor = VersionPerformanceMonitor()
metrics = monitor.get_version_metrics("v1.0.0-standard")
print(f"Avg latency: {metrics['avg_latency']:.2f} ms")
print(f"Avg relevance: {metrics['avg_relevance']:.2f}")
```

## 실제 데이터로 검증

실제 데이터베이스의 임베딩 데이터를 사용하여 FAISS 버전 관리 시스템을 검증할 수 있습니다.

### 검증 스크립트 실행

```bash
# 전체 테스트 실행
python scripts/test_faiss_version_with_real_data.py --db data/lawfirm_v2.db

# 특정 테스트만 실행
python scripts/test_faiss_version_with_real_data.py --db data/lawfirm_v2.db --skip-building --skip-switching

# 인덱스 빌드만 실행
python scripts/test_faiss_version_with_real_data.py --db data/lawfirm_v2.db --skip-creation --skip-switching --skip-search --skip-multi-search
```

### 검증 항목

1. **버전 생성**: 기존 EmbeddingVersionManager의 버전에 대응하는 FAISS 버전 생성
2. **인덱스 빌드**: 실제 임베딩 데이터로 FAISS 인덱스 빌드 및 저장
3. **버전 전환**: 여러 버전 간 전환 테스트
4. **버전별 검색**: 특정 버전으로 검색 수행
5. **멀티 버전 검색**: 여러 버전에서 동시 검색 및 결과 비교

### 예상 결과

```
================================================================================
Test 1: FAISS Version Creation
================================================================================
Found 1 active embedding version(s)

Processing version: v1.0.0-standard (ID: 1)
  Chunking strategy: standard
  Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
  Chunks: 15000
  Documents: 1500
  ✓ Created FAISS version: v1.0.0-standard-standard

================================================================================
Test 2: FAISS Index Building
================================================================================
Building index for version: v1.0.0-standard-standard
  Building FAISS index...
  ✓ Index built successfully for v1.0.0-standard-standard
  ✓ Set as active version

================================================================================
Test 3: Version Switching
================================================================================
Found 2 versions
  ✓ Successfully switched to v1.0.0-standard-standard

================================================================================
Test 4: Search with Version
================================================================================
Active version: v1.0.0-standard-standard
Searching: '전세금 반환 보증'
  Found 5 results
    1. Score: 0.8523, Text: 전세금 반환 보증에 관한 법률 조항...
    2. Score: 0.7845, Text: 계약 해지 시 전세금 반환...

================================================================================
Test Summary
================================================================================
creation        : ✓ PASSED
building        : ✓ PASSED
switching       : ✓ PASSED
search          : ✓ PASSED
multi_search    : ✓ PASSED

✓ All tests passed!
```

## 참고 자료

- [버전 관리 사용법](version_management_guide.md)
- [빠른 시작 가이드](faiss_version_quick_start.md)
- [SemanticSearchEngineV2 문서](../../../lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py)

