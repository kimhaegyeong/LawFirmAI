# FAISS 벡터 검색 최적화 제안서

## 개요

현재 `execute_searches_parallel` 노드에서 Phase 1 타임아웃이 발생하는 주요 원인은 FAISS 벡터 검색의 내부 로직이 최적화되지 않았기 때문입니다. 본 문서는 FAISS 인덱스와 벡터 검색 내부 로직을 최적화하는 구체적인 방법을 제안합니다.

## 현재 상태 분석

### 현재 구현 상태
- **FAISS 검색**: 단일 쿼리 벡터를 `index.search()`로 검색
- **검색 파라미터**: `search_k = k * 3`, `nprobe` 동적 조정
- **필터링**: 검색 후 DB 조회로 메타데이터 확인
- **배치 처리**: 부분적으로만 구현됨 (샘플링 단계)

### 성능 병목 지점
1. **단일 쿼리 검색**: 여러 쿼리를 순차적으로 검색
2. **DB 조회 오버헤드**: 검색 결과마다 개별 DB 조회
3. **메타데이터 캐싱 부족**: `_chunk_metadata` 캐시 활용도 낮음
4. **필터링 로직 비효율**: 검색 후 필터링으로 인한 오버헤드

## 최적화 방법 제안

### 1. 배치 검색 최적화 (높은 효과)

**목표**: 여러 쿼리를 한 번에 배치로 검색하여 오버헤드 감소

**현재 문제점**:
```python
# 현재: 각 쿼리를 개별적으로 검색
for query in queries:
    query_vec = self._embed_query(query)
    distances, indices = self.index.search(query_vec, k)
```

**개선 방법**:
```python
# 개선: 모든 쿼리를 배치로 검색
query_vectors = [self._embed_query(q) for q in queries]
query_vec_batch = np.array(query_vectors).astype('float32')
distances_batch, indices_batch = self.index.search(query_vec_batch, k)
```

**예상 효과**: 
- 검색 시간: 30-50% 감소 (쿼리 수에 따라)
- 메모리 사용: 약간 증가 (배치 크기에 따라)

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_search_with_threshold()` 메서드에 배치 검색 옵션 추가
- `semantic_search()` 메서드에서 배치 검색 호출

**우선순위**: ⭐⭐⭐⭐⭐ (최우선)

---

### 2. 메타데이터 캐싱 강화 (높은 효과)

**목표**: DB 조회를 최소화하여 필터링 속도 향상

**현재 문제점**:
- `_chunk_metadata` 캐시가 부분적으로만 사용됨
- 검색 결과마다 DB 조회 발생
- 배치 조회가 있지만 연결 풀 미사용

**개선 방법**:
```python
# 1. 메타데이터 사전 로딩 (인덱스 로드 시)
def _preload_chunk_metadata(self, chunk_ids: List[int]):
    """chunk_id 목록의 메타데이터를 사전 로드"""
    if not chunk_ids:
        return
    
    # 캐시에 없는 chunk_id만 조회
    missing_ids = [cid for cid in chunk_ids if cid not in self._chunk_metadata]
    if not missing_ids:
        return
    
    # 배치로 DB 조회 (연결 풀 사용)
    conn = self._get_connection()
    try:
        placeholders = ','.join(['?'] * len(missing_ids))
        cursor = conn.execute(
            f"SELECT id, source_type, type, source_id, source FROM text_chunks WHERE id IN ({placeholders})",
            missing_ids
        )
        for row in cursor.fetchall():
            self._chunk_metadata[row['id']] = {
                'source_type': row.get('source_type'),
                'type': row.get('type'),
                'source_id': row.get('source_id'),
                'source': row.get('source')
            }
    finally:
        if not self._connection_pool:
            conn.close()

# 2. 검색 전 메타데이터 사전 로드
def _search_with_threshold(self, ...):
    # FAISS 검색
    distances, indices = self.index.search(query_vec_np, search_k)
    
    # 검색 결과의 chunk_id 추출
    candidate_chunk_ids = [self._chunk_ids[idx] for idx in indices[0] if idx >= 0]
    
    # 메타데이터 사전 로드 (배치)
    self._preload_chunk_metadata(candidate_chunk_ids)
    
    # 이후 필터링은 캐시에서만 수행
```

**예상 효과**:
- DB 조회 시간: 70-90% 감소
- 필터링 속도: 50-70% 향상

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_preload_chunk_metadata()` 메서드 추가
- `_search_with_threshold()` 메서드 수정

**우선순위**: ⭐⭐⭐⭐⭐ (최우선)

---

### 3. FAISS 검색 파라미터 최적화 (중간 효과)

**목표**: FAISS 내부 파라미터를 최적화하여 검색 속도 향상

**현재 상태**:
- `nprobe` 동적 조정 구현됨
- `search_k = k * 3` 사용

**개선 방법**:

#### 3-1. GPU 사용 (가능한 경우)
```python
# GPU 사용 가능 여부 확인
import faiss
if faiss.get_num_gpus() > 0:
    # GPU 인덱스로 변환
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
    distances, indices = gpu_index.search(query_vec_np, search_k)
else:
    # CPU 인덱스 사용
    distances, indices = self.index.search(query_vec_np, search_k)
```

**예상 효과**: GPU 사용 시 5-10배 속도 향상

#### 3-2. 스레드 수 최적화
```python
# FAISS 스레드 수 설정
import faiss
faiss.omp_set_num_threads(4)  # CPU 코어 수에 맞게 조정
```

**예상 효과**: 멀티코어 활용 시 2-4배 속도 향상

#### 3-3. search_k 동적 조정 강화
```python
# 현재: search_k = k * 3
# 개선: 인덱스 타입과 k 값에 따라 동적 조정
if 'IndexIVFPQ' in type(self.index).__name__:
    # 압축 인덱스는 더 많은 후보 필요
    search_k = k * 4
elif 'IndexIVF' in type(self.index).__name__:
    # 일반 IVF 인덱스
    search_k = k * 3
else:
    # 정확한 인덱스 (Flat 등)
    search_k = k * 2
```

**예상 효과**: 검색 품질 유지하면서 속도 10-20% 향상

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_search_with_threshold()` 메서드 수정

**우선순위**: ⭐⭐⭐⭐ (높음)

---

### 4. 필터링 로직 최적화 (중간 효과)

**목표**: 검색 후 필터링을 최소화하여 전체 처리 시간 단축

**현재 문제점**:
- 검색 후 모든 결과에 대해 필터링 수행
- DB 조회가 개별적으로 발생

**개선 방법**:

#### 4-1. 사전 필터링 (가능한 경우)
```python
# FAISS 인덱스에 IDSelector 사용 (지원되는 경우)
if hasattr(faiss, 'IDSelector') and source_types:
    # source_type에 해당하는 chunk_id만 검색
    # (인덱스 구조에 따라 구현 방법 다름)
    pass
```

#### 4-2. 필터링 순서 최적화
```python
# 현재: 모든 필터를 순차 적용
# 개선: 빠른 필터부터 적용 (캐시 기반 필터 우선)

# 1단계: 메타데이터 캐시 기반 필터링 (가장 빠름)
filtered_indices = []
for idx, distance in zip(indices[0], distances[0]):
    if idx < 0:
        continue
    chunk_id = self._chunk_ids[idx]
    meta = self._chunk_metadata.get(chunk_id, {})
    
    # 캐시에 있으면 즉시 필터링
    if meta.get('source_type') in source_types:
        filtered_indices.append((idx, distance))
    elif not source_types:  # 필터 없으면 모두 통과
        filtered_indices.append((idx, distance))

# 2단계: DB 조회 필요한 경우만 처리
if len(filtered_indices) < k:
    # 추가 후보 확보를 위해 DB 조회
    missing_chunk_ids = [self._chunk_ids[idx] for idx, _ in filtered_indices[:k*2]]
    self._preload_chunk_metadata(missing_chunk_ids)
    # 재필터링
```

**예상 효과**: 필터링 시간 40-60% 감소

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_search_with_threshold()` 메서드의 필터링 로직 수정

**우선순위**: ⭐⭐⭐⭐ (높음)

---

### 5. 인덱스 로딩 최적화 (낮은 효과, 장기)

**목표**: 인덱스 로딩 시간 단축 및 메모리 사용 최적화

**개선 방법**:

#### 5-1. 메모리 매핑 (MMap) 사용
```python
# 대용량 인덱스의 경우 메모리 매핑 사용
if os.path.getsize(self.index_path) > 100 * 1024 * 1024:  # 100MB 이상
    self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)
else:
    self.index = faiss.read_index(str(self.index_path))
```

**예상 효과**: 메모리 사용량 감소, 로딩 시간 단축

#### 5-2. 지연 로딩
```python
# 인덱스를 처음 사용할 때만 로드
@property
def index(self):
    if self._index is None:
        self._load_index()
    return self._index
```

**예상 효과**: 초기화 시간 단축

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_load_index()` 메서드 수정

**우선순위**: ⭐⭐⭐ (중간)

---

### 6. 벡터 임베딩 캐싱 강화 (중간 효과)

**목표**: 동일 쿼리의 재임베딩 방지

**현재 상태**:
- 쿼리 임베딩 캐싱이 부분적으로만 구현됨

**개선 방법**:
```python
# 쿼리 임베딩 캐싱 강화
from functools import lru_cache
import hashlib

def _get_query_hash(self, query: str) -> str:
    """쿼리 해시 생성"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

@lru_cache(maxsize=1000)
def _embed_query_cached(self, query_hash: str, query: str) -> np.ndarray:
    """임베딩 캐싱"""
    return self._embed_query(query)

def _embed_query(self, query: str) -> np.ndarray:
    """임베딩 (캐싱 사용)"""
    query_hash = self._get_query_hash(query)
    return self._embed_query_cached(query_hash, query)
```

**예상 효과**: 동일 쿼리 재검색 시 80-90% 시간 단축

**구현 위치**: `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`
- `_embed_query()` 메서드 수정

**우선순위**: ⭐⭐⭐⭐ (높음)

---

## 구현 우선순위

### 즉시 적용 (높은 효과)
1. **배치 검색 최적화** ⭐⭐⭐⭐⭐
2. **메타데이터 캐싱 강화** ⭐⭐⭐⭐⭐

### 단기 적용 (중간 효과)
3. **FAISS 검색 파라미터 최적화** ⭐⭐⭐⭐
4. **필터링 로직 최적화** ⭐⭐⭐⭐
5. **벡터 임베딩 캐싱 강화** ⭐⭐⭐⭐

### 장기 적용 (추가 최적화)
6. **인덱스 로딩 최적화** ⭐⭐⭐

## 예상 성능 개선 효과

### 시나리오 1: 배치 검색 + 메타데이터 캐싱
- **검색 시간**: 30-50% 감소
- **필터링 시간**: 50-70% 감소
- **전체 시간**: 40-60% 감소

### 시나리오 2: 모든 최적화 적용
- **검색 시간**: 50-70% 감소
- **필터링 시간**: 60-80% 감소
- **전체 시간**: 55-75% 감소

### 시나리오 3: GPU 사용 가능한 경우
- **검색 시간**: 80-90% 감소 (GPU 사용 시)
- **전체 시간**: 70-85% 감소

## 구현 시 주의사항

1. **메모리 사용량**: 배치 검색 시 메모리 사용량 증가 주의
2. **캐시 크기**: 메타데이터 캐시 크기 제한 필요 (메모리 부족 방지)
3. **호환성**: FAISS 버전별 API 차이 확인
4. **테스트**: 각 최적화별로 성능 테스트 필수

## 참고 자료

- [FAISS 공식 문서](https://github.com/facebookresearch/faiss/wiki)
- [FAISS 성능 최적화 가이드](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS GPU 사용 가이드](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)

