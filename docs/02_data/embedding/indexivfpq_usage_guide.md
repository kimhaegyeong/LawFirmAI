# IndexIVFPQ 인덱스 사용 가이드

이 문서는 LangGraph에서 IndexIVFPQ 인덱스를 사용하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [IndexIVFPQ 인덱스 생성](#indexivfpq-인덱스-생성)
3. [환경 변수 설정](#환경-변수-설정)
4. [LangGraph에서 사용](#langgraph에서-사용)
5. [검증 방법](#검증-방법)

## 개요

IndexIVFPQ는 FAISS의 압축 인덱스 타입으로, Product Quantization을 사용하여 메모리 사용량을 크게 줄이면서도 검색 성능을 유지할 수 있습니다.

### 장점

- ✅ **메모리 효율성**: 원본 대비 약 48배 압축 (예: 95.46 MB → 1.99 MB)
- ✅ **빠른 검색**: IndexIVF 계열 인덱스의 빠른 검색 성능
- ✅ **대용량 데이터 지원**: 메모리 제약이 있는 환경에서 대용량 인덱스 사용 가능

## IndexIVFPQ 인덱스 생성

### 1. 인덱스 생성 스크립트 실행

```bash
python lawfirm_langgraph/tests/scripts/build_indexivfpq.py \
    --version-id 5 \
    --m 64 \
    --nbits 8
```

### 2. 파라미터 설명

- `--version-id`: 임베딩 버전 ID (필수)
- `--m`: Product Quantization 서브벡터 개수 (기본값: 64)
- `--nbits`: 각 서브벡터의 비트 수 (기본값: 8)
- `--nlist`: 클러스터 수 (선택사항, 자동 계산)
- `--output`: 출력 인덱스 파일 경로 (선택사항)

### 3. 생성된 파일

인덱스 생성 후 다음 파일들이 생성됩니다:

```
data/vector_store/v2.0.0-dynamic-dynamic-ivfpq/
├── index.faiss                    # IndexIVFPQ 인덱스 파일
├── index.chunk_ids.json          # chunk_id 매핑 파일
├── ml_enhanced_faiss_index.faiss # 호환성을 위한 복사본
└── ml_enhanced_faiss_index.chunk_ids.json # 호환성을 위한 복사본
```

## 환경 변수 설정

### 1. `.env` 파일 설정

프로젝트 루트 또는 `api/` 디렉토리의 `.env` 파일에 다음을 추가:

```env
# 외부 벡터 인덱스 사용 활성화
USE_EXTERNAL_VECTOR_STORE=true

# IndexIVFPQ 인덱스 경로 설정
EXTERNAL_VECTOR_STORE_BASE_PATH=./data/vector_store/v2.0.0-dynamic-dynamic-ivfpq
```

### 2. 환경 변수 설명

- `USE_EXTERNAL_VECTOR_STORE`: 외부 인덱스 사용 여부 (`true`/`false`)
- `EXTERNAL_VECTOR_STORE_BASE_PATH`: IndexIVFPQ 인덱스가 있는 디렉토리 경로
- `VECTOR_STORE_VERSION`: 벡터스토어 버전 번호 (선택사항)

## LangGraph에서 사용

### 자동 감지

LangGraph는 환경 변수를 읽어서 자동으로 IndexIVFPQ 인덱스를 로드합니다.

```python
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig

# 환경 변수에서 설정 읽기
config = LangGraphConfig.from_env()

# 워크플로우 초기화 (IndexIVFPQ 자동 로드)
workflow = EnhancedLegalQuestionWorkflow(config)

# 검색 실행
results = workflow.semantic_search.search("임대차 보증금", k=5)
```

### 코드에서 직접 설정

```python
import os
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig

# 환경 변수 설정
os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = './data/vector_store/v2.0.0-dynamic-dynamic-ivfpq'

# 워크플로우 초기화
config = LangGraphConfig.from_env()
workflow = EnhancedLegalQuestionWorkflow(config)
```

## 검증 방법

### 1. 인덱스 타입 확인

```python
if workflow.semantic_search and workflow.semantic_search.index:
    index_type = type(workflow.semantic_search.index).__name__
    print(f"인덱스 타입: {index_type}")
    
    if 'IndexIVFPQ' in index_type:
        print("✅ IndexIVFPQ 인덱스가 로드되었습니다!")
        if hasattr(workflow.semantic_search.index, 'pq'):
            m = workflow.semantic_search.index.pq.M
            nbits = workflow.semantic_search.index.pq.nbits
            print(f"PQ 파라미터: M={m}, nbits={nbits}")
```

### 2. 테스트 스크립트 실행

```bash
python lawfirm_langgraph/tests/scripts/test_langgraph_with_indexivfpq.py
```

### 3. 로그 확인

IndexIVFPQ 인덱스가 로드되면 다음과 같은 로그가 출력됩니다:

```
Found external index file: index.faiss
Loaded external FAISS index: IndexIVFPQ (32,583 vectors)
✅ IndexIVFPQ detected - using compressed index for memory efficiency
   PQ parameters: M=64, nbits=8
```

## 인덱스 파일 이름 지원

외부 인덱스 로드 로직은 다음 파일 이름을 자동으로 찾습니다:

1. `ml_enhanced_faiss_index.faiss` (기본 이름)
2. `index.faiss` (IndexIVFPQ 인덱스 이름)
3. `faiss_index.faiss` (대체 이름)

메타데이터 파일도 자동으로 찾습니다:

1. `ml_enhanced_faiss_index.json`
2. `index.json` (인덱스 파일 이름 기반)
3. `metadata.json`

chunk_ids 파일도 자동으로 찾습니다:

1. `ml_enhanced_faiss_index.chunk_ids.json`
2. `index.chunk_ids.json` (인덱스 파일 이름 기반)
3. `chunk_ids.json`

## 성능 비교

### 메모리 사용량

| 인덱스 타입 | 파일 크기 | 메모리 사용량 |
|------------|----------|--------------|
| IndexIVFFlat | ~95 MB | ~95 MB |
| IndexIVFPQ (M=64, nbits=8) | ~4 MB | ~2 MB |
| **압축률** | **~24배** | **~48배** |

### 검색 성능

IndexIVFPQ는 IndexIVFFlat과 유사한 검색 성능을 제공하면서 메모리 사용량을 크게 줄입니다.

## 문제 해결

### 문제 1: 인덱스 파일을 찾을 수 없음

**증상**: `External FAISS index not found`

**해결 방법**:
1. `EXTERNAL_VECTOR_STORE_BASE_PATH` 경로가 올바른지 확인
2. 디렉토리에 `index.faiss` 또는 `ml_enhanced_faiss_index.faiss` 파일이 있는지 확인

### 문제 2: 메타데이터 파일을 찾을 수 없음

**증상**: `Metadata file not found`

**해결 방법**:
1. 인덱스 디렉토리에 JSON 메타데이터 파일이 있는지 확인
2. `index.json` 또는 `ml_enhanced_faiss_index.json` 파일이 있는지 확인

### 문제 3: 검색 결과가 0개

**증상**: 검색은 성공하지만 결과가 없음

**해결 방법**:
1. `embedding_version_id` 필터링 문제일 수 있음
2. `similarity_threshold` 값을 낮춰서 테스트
3. 활성 임베딩 버전이 올바르게 설정되었는지 확인

## 관련 문서

- [외부 인덱스 설정 가이드](./external_index_config_guide.md)
- [벡터 임베딩 가이드](./embedding_guide.md)
- [버전 관리 사용법](./version_management_guide.md)

