# FAISS 인덱스 생성 방식 검토

## 현재 상태

### 1. 인덱스 타입

**현재 사용 중**: `IndexFlatIP` (기본 Flat 인덱스)

**위치**: `scripts/ingest/open_law/embedding/faiss/faiss_embedder.py`

```python
# 현재 코드 (278-292줄)
# FAISS 인덱스 저장 (Flat 인덱스, 나중에 IndexIVFFlat로 변환 가능)
faiss.normalize_L2(all_embeddings)
index = faiss.IndexFlatIP(self.dimension)  # 기본 Flat 인덱스
```

**문제점**:
- ❌ `IndexFlatIP`는 전체 벡터를 순차 검색하는 기본 인덱스
- ❌ 대용량 데이터에서 검색 속도가 느림
- ❌ 주석에 "나중에 IndexIVFFlat로 변환 가능"이라고 되어 있지만 실제로는 변환하지 않음

**권장 사항**: `IndexIVFFlat` 사용
- ✅ IVF (Inverted File) 구조로 빠른 검색
- ✅ 대용량 데이터에 적합
- ✅ 검색 속도와 정확도의 균형

### 2. MLflow 통합

**현재 상태**: MLflow를 사용하지 않음

**MLflow 관련 코드**:
- `scripts/rag/mlflow_manager.py` - MLflowFAISSManager 클래스 존재
- `scripts/rag/build_index.py` - MLflow를 사용하는 별도 빌드 스크립트
- `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py` - MLflow 인덱스 로드 기능

**현재 embedder**:
- ❌ `faiss_embedder.py`에서 MLflow를 사용하지 않음
- ❌ 로컬 파일 시스템에만 저장
- ❌ 버전 관리 없음

**권장 사항**: MLflow 통합
- ✅ 인덱스 버전 관리
- ✅ 실험 추적
- ✅ 프로덕션 배포 관리

## 개선 방안

### 1. IndexIVFFlat로 변경

`faiss_embedder.py`의 `save_embeddings` 메서드를 수정하여 `IndexIVFFlat`를 사용하도록 변경:

```python
# 개선된 코드
from scripts.ingest.open_law.embedding.faiss.faiss_indexer import FaissIndexer

# IndexIVFFlat 사용
indexer = FaissIndexer(dimension=self.dimension)
index = indexer.build_index(
    embeddings=all_embeddings,
    index_type="ivfflat",
    nlist=None  # 자동 계산
)
```

### 2. MLflow 통합 추가

`faiss_embedder.py`에 MLflow 저장 기능 추가:

```python
# MLflow 통합
from scripts.rag.mlflow_manager import MLflowFAISSManager

mlflow_manager = MLflowFAISSManager()
run_id = mlflow_manager.create_run(
    version_name=f"statutes_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags={"data_type": data_type, "domain": domain}
)
mlflow_manager.save_index(
    run_id=run_id,
    index=index,
    id_mapping=dict(enumerate(self.chunk_ids)),
    metadata=self.metadata_list
)
```

## 비교표

| 항목 | 현재 (IndexFlatIP) | 권장 (IndexIVFFlat) |
|------|-------------------|---------------------|
| 검색 속도 | 느림 (O(n)) | 빠름 (O(n/nlist)) |
| 메모리 사용 | 낮음 | 중간 |
| 정확도 | 높음 | 높음 (nprobe 조정 가능) |
| 대용량 적합성 | ❌ | ✅ |
| 학습 필요 | ❌ | ✅ |
| MLflow 통합 | ❌ | ✅ (추가 필요) |

## 결론

1. **현재는 IndexFlatIP 사용 중** - 기본 Flat 인덱스
2. **IndexIVFFlat로 변경 필요** - 성능 향상
3. **MLflow 통합 필요** - 버전 관리 및 배포 관리

