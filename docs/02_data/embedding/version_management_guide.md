# 벡터스토어 버전 관리 가이드

이 문서는 LawFirmAI 프로젝트에서 벡터 임베딩 인덱스의 버전을 관리하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [버전 관리 시스템 구조](#버전-관리-시스템-구조)
3. [버전 생성](#버전-생성)
4. [버전 조회 및 전환](#버전-조회-및-전환)
5. [버전 삭제](#버전-삭제)
6. [사용 예시](#사용-예시)
7. [모범 사례](#모범-사례)

## 개요

벡터스토어 버전 관리 시스템을 사용하면:
- ✅ 여러 버전의 인덱스를 유지하고 전환 가능
- ✅ 메타데이터 변경 시 이전 버전으로 롤백 가능
- ✅ 버전별 성능 비교 및 테스트 가능
- ✅ 프로덕션과 개발 환경에서 다른 버전 사용 가능

## 버전 관리 시스템 구조

### 디렉토리 구조

```
data/embeddings/ml_enhanced_ko_sroberta_precedents/
├── versions.json                    # 버전 메타데이터
├── ml_enhanced_faiss_index.faiss    # 기본 인덱스 (버전 없음)
├── ml_enhanced_faiss_index.json     # 기본 메타데이터
├── v2.0.0/                          # 버전별 디렉토리
│   ├── ml_enhanced_faiss_index.faiss
│   └── ml_enhanced_faiss_index.json
└── v1.5.0/                          # 다른 버전
    ├── ml_enhanced_faiss_index.faiss
    └── ml_enhanced_faiss_index.json
```

### versions.json 구조

```json
{
  "current_version": "v2.0.0",
  "versions": [
    {
      "version": "v2.0.0",
      "created_at": "2025-11-13T10:00:00",
      "metadata": {
        "model_name": "jhgan/ko-sroberta-multitask",
        "vector_count": 33598,
        "description": "Enhanced metadata version"
      }
    },
    {
      "version": "v1.5.0",
      "created_at": "2025-10-19T20:17:47",
      "metadata": {
        "model_name": "jhgan/ko-sroberta-multitask",
        "vector_count": 33598
      }
    }
  ]
}
```

## 버전 생성

### 방법 1: 벡터 빌더 스크립트 사용

벡터 임베딩을 생성할 때 버전을 지정합니다:

```python
from scripts.ml_training.vector_embedding.incremental_precedent_vector_builder import IncrementalPrecedentVectorBuilder

builder = IncrementalPrecedentVectorBuilder(
    embedding_output_path="data/embeddings/ml_enhanced_ko_sroberta_precedents",
    version="v2.0.0"  # 새 버전 지정
)

# 벡터 임베딩 생성
stats = builder.build_incremental_embeddings(category="civil")
```

### 방법 2: 버전 관리 스크립트 사용

기존 인덱스를 새 버전으로 복사:

```python
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager
from pathlib import Path

base_path = Path("data/embeddings/ml_enhanced_ko_sroberta_precedents")
version_manager = VectorStoreVersionManager(base_path)

# 새 버전 생성
version_manager.create_version(
    version="v2.0.0",
    metadata={
        "model_name": "jhgan/ko-sroberta-multitask",
        "vector_count": 33598,
        "description": "Enhanced metadata version"
    }
)
```

### 방법 3: 재빌드 스크립트 사용

전체 인덱스를 새 버전으로 재빌드:

```bash
python scripts/ml_training/vector_embedding/rebuild_with_enhanced_metadata.py \
    --base-path data/embeddings/ml_enhanced_ko_sroberta_precedents \
    --version v2.0.0
```

## 버전 조회 및 전환

### 현재 버전 확인

```python
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager
from pathlib import Path

version_manager = VectorStoreVersionManager(
    Path("data/embeddings/ml_enhanced_ko_sroberta_precedents")
)

# 현재 버전
current = version_manager.get_current_version()
print(f"Current version: {current}")

# 최신 버전
latest = version_manager.get_latest_version()
print(f"Latest version: {latest}")

# 모든 버전 목록
versions = version_manager.list_versions()
for v in versions:
    print(f"Version: {v['version']}, Created: {v['created_at']}")
```

### 버전 전환

#### 방법 1: 환경 변수 사용

`.env` 파일에서 버전 지정:

```env
VECTOR_STORE_VERSION=v2.0.0
```

#### 방법 2: 스크립트 사용

```bash
python scripts/ml_training/vector_embedding/switch_version.py \
    --base-path data/embeddings/ml_enhanced_ko_sroberta_precedents \
    --version v2.0.0
```

#### 방법 3: Python 코드에서

```python
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager
from pathlib import Path

version_manager = VectorStoreVersionManager(
    Path("data/embeddings/ml_enhanced_ko_sroberta_precedents")
)

# 버전 전환
success = version_manager.set_current_version("v2.0.0")
if success:
    print("Version switched successfully")
else:
    print("Failed to switch version")
```

### 버전 경로 조회

```python
version_manager = VectorStoreVersionManager(base_path)

# 특정 버전의 경로
version_path = version_manager.get_version_path("v2.0.0")
print(f"Version path: {version_path}")

# 현재 버전의 경로
current_path = version_manager.get_version_path()
print(f"Current version path: {current_path}")
```

## 버전 삭제

⚠️ **주의**: 현재 활성 버전은 삭제할 수 없습니다. 먼저 다른 버전으로 전환하세요.

```python
version_manager = VectorStoreVersionManager(base_path)

# 다른 버전으로 전환
version_manager.set_current_version("v1.5.0")

# 버전 삭제
success = version_manager.delete_version("v2.0.0")
if success:
    print("Version deleted successfully")
    # 실제 파일은 수동으로 삭제해야 할 수 있습니다
else:
    print("Failed to delete version (may be current version)")
```

## 사용 예시

### 예시 1: 새 메타데이터로 버전 업그레이드

```python
# 1. 기존 인덱스 백업 (v1.5.0으로 명명)
version_manager = VectorStoreVersionManager(base_path)
version_manager.create_version("v1.5.0", {"description": "Backup before upgrade"})

# 2. 새 버전으로 재빌드
builder = IncrementalPrecedentVectorBuilder(
    embedding_output_path=base_path,
    version="v2.0.0"
)
builder.build_incremental_embeddings()

# 3. 새 버전을 현재 버전으로 설정
version_manager.set_current_version("v2.0.0")
```

### 예시 2: 문제 발생 시 롤백

```python
# 문제 발견
print("Issue detected with v2.0.0")

# 이전 버전으로 롤백
version_manager.set_current_version("v1.5.0")
print("Rolled back to v1.5.0")

# 애플리케이션 재시작 필요
```

### 예시 3: 버전별 성능 비교

```python
versions = ["v1.5.0", "v2.0.0"]

for version in versions:
    version_manager.set_current_version(version)
    
    # 검색 엔진 재초기화
    engine = SemanticSearchEngineV2(
        use_external_index=True,
        vector_store_version=version
    )
    
    # 성능 테스트
    import time
    start = time.time()
    results = engine.search("테스트 쿼리", k=10)
    elapsed = time.time() - start
    
    print(f"{version}: {elapsed:.4f}s, {len(results)} results")
```

## 모범 사례

### 1. 버전 명명 규칙

Semantic Versioning을 따르세요:
- `v2.0.0`: 메이저 업데이트 (메타데이터 구조 변경)
- `v2.1.0`: 마이너 업데이트 (새 데이터 추가)
- `v2.1.1`: 패치 업데이트 (버그 수정)

### 2. 버전 생성 시점

- ✅ 메타데이터 구조 변경 시
- ✅ 대량의 새 데이터 추가 시
- ✅ 모델 변경 시
- ✅ 인덱스 최적화 후

### 3. 버전 관리 전략

```python
# 프로덕션 환경
VECTOR_STORE_VERSION=v2.0.0  # 안정적인 버전

# 개발 환경
VECTOR_STORE_VERSION=v2.1.0  # 최신 버전 테스트

# 스테이징 환경
VECTOR_STORE_VERSION=v2.0.0  # 프로덕션과 동일
```

### 4. 백업 전략

중요한 변경 전에는 항상 백업 버전을 생성:

```python
# 현재 버전을 백업
current = version_manager.get_current_version()
version_manager.create_version(
    f"{current}_backup_{datetime.now().strftime('%Y%m%d')}",
    {"description": "Backup before major update"}
)
```

### 5. 버전 정리

오래된 버전은 주기적으로 정리:

```python
# 6개월 이상 된 버전 삭제
from datetime import datetime, timedelta

versions = version_manager.list_versions()
cutoff_date = datetime.now() - timedelta(days=180)

for v in versions:
    created = datetime.fromisoformat(v['created_at'])
    if created < cutoff_date and v['version'] != version_manager.get_current_version():
        version_manager.delete_version(v['version'])
```

## 관련 스크립트

### 버전 관리 스크립트

- `scripts/ml_training/vector_embedding/version_manager.py`: 버전 관리 클래스
- `scripts/ml_training/vector_embedding/switch_version.py`: 버전 전환 스크립트
- `scripts/ml_training/vector_embedding/rebuild_with_enhanced_metadata.py`: 재빌드 스크립트

### 사용 예시

```bash
# 버전 목록 확인
python -c "from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager; from pathlib import Path; vm = VectorStoreVersionManager(Path('data/embeddings/ml_enhanced_ko_sroberta_precedents')); print([v['version'] for v in vm.list_versions()])"

# 버전 전환
python scripts/ml_training/vector_embedding/switch_version.py \
    --base-path data/embeddings/ml_enhanced_ko_sroberta_precedents \
    --version v2.0.0
```

## 문제 해결

### 문제 1: 버전을 찾을 수 없음

**증상**: `Version not found`

**해결**: 
- `versions.json` 파일이 존재하는지 확인
- 버전 번호가 정확한지 확인 (대소문자 구분)

### 문제 2: 현재 버전 삭제 불가

**증상**: `Cannot delete current version`

**해결**: 
- 먼저 다른 버전으로 전환
- 그 다음 삭제

### 문제 3: 버전 경로가 존재하지 않음

**증상**: `Version path does not exist`

**해결**: 
- 해당 버전의 디렉토리가 실제로 존재하는지 확인
- 인덱스 파일이 올바른 위치에 있는지 확인

## 관련 문서

- [외부 인덱스 설정 가이드](./external_index_config_guide.md)
- [벡터 임베딩 가이드](./embedding_guide.md)

