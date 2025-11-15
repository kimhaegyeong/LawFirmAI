# 외부 벡터 인덱스 설정 가이드

이 문서는 LawFirmAI 프로젝트에서 외부 FAISS 벡터 인덱스를 사용하도록 설정하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [환경 변수 설정](#환경-변수-설정)
3. [설정 옵션](#설정-옵션)
4. [사용 예시](#사용-예시)
5. [문제 해결](#문제-해결)

## 개요

LawFirmAI는 두 가지 방식으로 벡터 검색을 수행할 수 있습니다:

1. **DB 기반 인덱스**: `lawfirm_v2.db`에 저장된 임베딩을 사용하여 FAISS 인덱스를 자동 생성
2. **외부 인덱스**: 미리 생성된 FAISS 인덱스 파일을 직접 사용

외부 인덱스를 사용하면:
- ✅ 더 빠른 검색 성능 (인덱스가 이미 최적화됨)
- ✅ 버전 관리 가능 (여러 버전의 인덱스 유지)
- ✅ 메타데이터가 풍부한 검색 결과

## 환경 변수 설정

### 1. `.env` 파일 생성 또는 수정

프로젝트 루트 또는 `api/` 디렉토리에 `.env` 파일을 생성하거나 수정합니다.

```bash
# api/.env 또는 프로젝트 루트/.env
```

### 2. 외부 인덱스 활성화

```env
# 외부 벡터 인덱스 사용 활성화
USE_EXTERNAL_VECTOR_STORE=true

# 외부 벡터 인덱스 기본 경로
# 디렉토리 경로를 지정하면 자동으로 ml_enhanced_faiss_index.faiss 파일을 찾습니다
EXTERNAL_VECTOR_STORE_BASE_PATH=./data/embeddings/ml_enhanced_ko_sroberta_precedents

# 벡터스토어 버전 (선택사항)
# 지정하지 않으면 최신 버전을 자동으로 사용합니다
VECTOR_STORE_VERSION=v2.0.0
```

### 3. 전체 설정 예시

```env
# 데이터베이스 설정
DATABASE_PATH=./data/lawfirm_v2.db
DATABASE_URL=sqlite:///./data/lawfirm_v2.db

# 외부 벡터 인덱스 설정
USE_EXTERNAL_VECTOR_STORE=true
EXTERNAL_VECTOR_STORE_BASE_PATH=./data/embeddings/ml_enhanced_ko_sroberta_precedents
VECTOR_STORE_VERSION=v2.0.0

# 기타 설정
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 설정 옵션

### USE_EXTERNAL_VECTOR_STORE

- **타입**: `boolean`
- **기본값**: `false`
- **설명**: 외부 벡터 인덱스 사용 여부
- **예시**: `USE_EXTERNAL_VECTOR_STORE=true`

### EXTERNAL_VECTOR_STORE_BASE_PATH

- **타입**: `string` (경로)
- **기본값**: `None`
- **설명**: 외부 FAISS 인덱스가 저장된 디렉토리 경로
- **형식**: 
  - 상대 경로: `./data/embeddings/ml_enhanced_ko_sroberta_precedents`
  - 절대 경로: `D:/project/LawFirmAI/LawFirmAI/data/embeddings/ml_enhanced_ko_sroberta_precedents`
- **참고**: 디렉토리를 지정하면 자동으로 `ml_enhanced_faiss_index.faiss` 파일을 찾습니다

### VECTOR_STORE_VERSION

- **타입**: `string` (버전 번호)
- **기본값**: `None` (최신 버전 자동 사용)
- **설명**: 사용할 벡터스토어 버전 번호
- **형식**: `v2.0.0`, `v1.5.0` 등 (semantic versioning)
- **예시**: `VECTOR_STORE_VERSION=v2.0.0`

## 사용 예시

### 예시 1: 기본 외부 인덱스 사용

```env
USE_EXTERNAL_VECTOR_STORE=true
EXTERNAL_VECTOR_STORE_BASE_PATH=./data/embeddings/ml_enhanced_ko_sroberta_precedents
```

이 설정으로 최신 버전의 인덱스를 자동으로 사용합니다.

### 예시 2: 특정 버전 지정

```env
USE_EXTERNAL_VECTOR_STORE=true
EXTERNAL_VECTOR_STORE_BASE_PATH=./data/embeddings/ml_enhanced_ko_sroberta_precedents
VECTOR_STORE_VERSION=v2.0.0
```

특정 버전의 인덱스를 명시적으로 지정합니다.

### 예시 3: DB 기반 인덱스 사용 (기본값)

```env
# USE_EXTERNAL_VECTOR_STORE를 설정하지 않거나 false로 설정
USE_EXTERNAL_VECTOR_STORE=false
```

또는 환경 변수를 설정하지 않으면 DB 기반 인덱스를 사용합니다.

## 코드에서 사용

### Python 코드에서 직접 설정

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

# 외부 인덱스 사용
engine = SemanticSearchEngineV2(
    db_path='data/lawfirm_v2.db',
    use_external_index=True,
    external_index_path='./data/embeddings/ml_enhanced_ko_sroberta_precedents',
    vector_store_version='v2.0.0'  # 선택사항
)

# 검색 실행
results = engine.search('계약 해제', k=5)
```

### Config 클래스를 통한 설정

```python
from lawfirm_langgraph.core.utils.config import Config

config = Config()

# Config에서 자동으로 환경 변수를 읽어옵니다
print(f"Use external index: {config.use_external_vector_store}")
print(f"External path: {config.external_vector_store_base_path}")
print(f"Version: {config.vector_store_version}")
```

## 문제 해결

### 문제 1: 외부 인덱스를 찾을 수 없음

**증상**: 
```
External FAISS index not found: ...
```

**해결 방법**:
1. `EXTERNAL_VECTOR_STORE_BASE_PATH` 경로가 올바른지 확인
2. 해당 경로에 `ml_enhanced_faiss_index.faiss` 파일이 있는지 확인
3. 경로가 디렉토리인지 파일인지 확인 (디렉토리여야 함)

### 문제 2: 메타데이터가 로드되지 않음

**증상**:
```
External metadata length: 0
```

**해결 방법**:
1. `ml_enhanced_faiss_index.json` 파일이 같은 디렉토리에 있는지 확인
2. JSON 파일의 구조가 올바른지 확인 (`document_metadata`, `document_texts` 키 존재)

### 문제 3: 검색 결과가 0개

**증상**:
```
Found 0 results
```

**해결 방법**:
1. 인덱스가 제대로 로드되었는지 확인
2. `similarity_threshold` 값을 낮춰서 테스트
3. 로그에서 에러 메시지 확인

### 문제 4: 버전을 찾을 수 없음

**증상**:
```
No versions found in vector store
```

**해결 방법**:
1. 버전 관리 시스템이 설정되어 있는지 확인
2. `versions.json` 파일이 기본 경로에 있는지 확인
3. `VECTOR_STORE_VERSION`을 명시적으로 지정하거나 제거

## 검증 방법

설정이 제대로 적용되었는지 확인하려면:

```python
from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

engine = SemanticSearchEngineV2()

# 진단 정보 확인
if hasattr(engine, 'diagnose'):
    diagnosis = engine.diagnose()
    print(f"Available: {diagnosis.get('available')}")
    print(f"Index loaded: {diagnosis.get('faiss_index_exists')}")
    print(f"External metadata: {len(engine._external_metadata)} items")

# 검색 테스트
results = engine.search('테스트 쿼리', k=3)
print(f"Search results: {len(results)} items")
```

## 관련 문서

- [버전 관리 사용법](./version_management_guide.md)
- [벡터 임베딩 가이드](./embedding_guide.md)
- [API 통합 테스트](../../../tests/integration/test_api_external_index.py)

