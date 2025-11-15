# FAISS 버전 관리 빠른 시작 가이드

## 개요

이 가이드는 FAISS 벡터 임베딩 및 청킹 버전 관리 시스템을 빠르게 시작하는 방법을 설명합니다.

## 사전 요구사항

- Python 3.8+
- FAISS 라이브러리 (`pip install faiss-cpu` 또는 `faiss-gpu`)
- 기존 임베딩 데이터가 있는 데이터베이스

## 빠른 시작

### 1. 기본 설정

```python
from scripts.utils.faiss_version_manager import FAISSVersionManager
from scripts.utils.embedding_version_manager import EmbeddingVersionManager

# 버전 관리자 초기화
faiss_manager = FAISSVersionManager("data/vector_store")
embedding_manager = EmbeddingVersionManager("data/lawfirm_v2.db")
```

### 2. 버전 생성

```python
# Embedding 버전 등록 (FAISS 버전도 자동 생성)
version_id = embedding_manager.register_version(
    version_name="v1.0.0-standard",
    chunking_strategy="standard",
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    set_active=True,
    create_faiss_version=True,
    faiss_version_manager=faiss_manager
)
```

### 3. FAISS 인덱스 빌드

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

# 검색 엔진 초기화
engine = SemanticSearchEngineV2(db_path="data/lawfirm_v2.db")

# 인덱스 빌드
engine._build_faiss_index_sync(
    embedding_version_id=version_id,
    faiss_version_name="v1.0.0-standard-standard"
)
```

### 4. 검색 사용

```python
# 특정 버전으로 검색
results = engine.search(
    query="전세금 반환 보증",
    k=10,
    faiss_version="v1.0.0-standard-standard"
)

# 여러 버전 동시 검색
multi_results = engine.search_multiple_versions(
    query="전세금 반환 보증",
    versions=["v1.0.0-standard-standard", "v1.0.0-dynamic-dynamic"],
    k=10
)
```

## 일반적인 워크플로우

### 시나리오 1: 새 청킹 전략 테스트

```python
# 1. 새 임베딩 버전 생성
new_version_id = embedding_manager.register_version(
    version_name="v2.0.0-dynamic",
    chunking_strategy="dynamic",
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    set_active=False,  # 먼저 비활성으로 생성
    create_faiss_version=True,
    faiss_version_manager=faiss_manager
)

# 2. 데이터 재임베딩 (기존 재임베딩 스크립트 사용)
# ... 재임베딩 로직 ...

# 3. FAISS 인덱스 빌드
engine._build_faiss_index_sync(
    embedding_version_id=new_version_id,
    faiss_version_name="v2.0.0-dynamic-dynamic"
)

# 4. A/B 테스트
results_v1 = engine.search("전세금 반환 보증", faiss_version="v1.0.0-standard-standard")
results_v2 = engine.search("전세금 반환 보증", faiss_version="v2.0.0-dynamic-dynamic")

# 5. 성능 비교 후 활성화
if results_v2_better:
    faiss_manager.set_active_version("v2.0.0-dynamic-dynamic")
    embedding_manager.set_active_version(new_version_id)
```

### 시나리오 2: 버전 백업 및 복원

```python
from scripts.utils.faiss_backup_manager import FAISSBackupManager

# 백업
backup_manager = FAISSBackupManager(
    backup_path="data/backups/faiss_versions",
    faiss_version_manager=faiss_manager
)
backup_file = backup_manager.backup_version("v1.0.0-standard-standard")

# 복원
backup_manager.restore_version(backup_file)
```

### 시나리오 3: 성능 모니터링

```python
from scripts.utils.version_performance_monitor import VersionPerformanceMonitor

monitor = VersionPerformanceMonitor()

# 검색 후 자동 로깅 (SemanticSearchEngineV2에 통합됨)
results = engine.search("전세금 반환 보증", faiss_version="v1.0.0-standard-standard")

# 성능 비교
comparison = monitor.compare_performance("v1.0.0-standard-standard", "v2.0.0-dynamic-dynamic")
print(f"Latency improvement: {comparison['latency_improvement_percent']:.2f}%")
```

## CLI 도구 사용

### 버전 목록 조회

```bash
python scripts/utils/faiss_version_switcher.py list
```

### 버전 전환

```bash
python scripts/utils/faiss_version_switcher.py switch v1.0.0-standard-standard
```

### 버전 비교

```bash
python scripts/utils/faiss_version_switcher.py compare v1.0.0-standard-standard v2.0.0-dynamic-dynamic
```

### 성능 대시보드 생성

```bash
python scripts/utils/version_performance_dashboard.py --format markdown --output performance_report.md
```

## 실제 데이터로 검증

```bash
# 전체 검증 테스트 실행
python scripts/test_faiss_version_with_real_data.py --db data/lawfirm_v2.db
```

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

### 인덱스 빌드 실패

```python
# 임베딩 데이터 확인
stats = embedding_manager.get_version_statistics(version_id)
print(f"Chunks: {stats['chunk_count']}")
print(f"Embeddings: {stats['embedding_count']}")

# 인덱스 빌드 재시도
engine._build_faiss_index_sync(embedding_version_id=version_id)
```

### 검색 결과가 없음

```python
# 인덱스 로드 확인
engine._load_faiss_index("v1.0.0-standard-standard")
print(f"Index loaded: {engine.index is not None}")
print(f"Total vectors: {engine.index.ntotal if engine.index else 0}")
```

## 다음 단계

- [상세 가이드](faiss_version_management_guide.md) 참조
- [버전 관리 사용법](version_management_guide.md) 참조
- 실제 데이터로 검증 스크립트 실행

