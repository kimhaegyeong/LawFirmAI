# Ground Truth 생성 성능 개선 방안

## 문제점 분석

프로그램이 한 시간 넘게 실행되어도 종료되지 않는 주요 원인:

### 1. FAISS 인덱스에서 임베딩 추출 (가장 큰 병목)

**현재 코드:**
```python
for i in tqdm(range(n_total), desc="Extracting embeddings"):
    embedding = self.index.reconstruct(i)
    embeddings[i] = embedding.astype(np.float32)
```

**문제점:**
- `reconstruct()`를 하나씩 호출하면 매우 느림
- 60,000개 이상의 벡터가 있으면 시간이 매우 오래 걸림
- IndexIVFPQ 같은 압축 인덱스에서는 더 느림
- 예상 시간: 60,000개 × 0.01초 = 600초 (10분 이상)

**개선 방안:**
- 배치로 추출: `reconstruct_n()` 사용 (가능한 경우)
- 또는 문서 텍스트에서 재생성 (더 빠를 수 있음)
- 샘플링: 전체 데이터 대신 샘플만 사용

### 2. Elbow Method로 최적 클러스터 수 찾기

**현재 코드:**
```python
for k in tqdm(k_range, desc="Testing cluster numbers"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    inertias.append(kmeans.inertia_)
```

**문제점:**
- 여러 K 값(2~50)에 대해 각각 K-means 실행
- `n_init=10`으로 각 K마다 10번 초기화 시도
- 전체 데이터셋에 대해 실행하므로 매우 느림
- 예: 20개 K 값 × 10번 초기화 = 200번 K-means 실행
- 예상 시간: 60,000개 데이터 기준 각 K-means가 1-2분 → 총 20-40분

**개선 방안:**
- `n_init`을 3으로 줄이기 (10 → 3)
- 샘플링된 데이터로 Elbow method 실행 (예: 10,000개 샘플)
- 또는 사용자가 직접 클러스터 수 지정하도록 변경
- 최대 K 값을 줄이기 (예: 50 → 20)

### 3. 데이터베이스에서 청크 로드

**현재 코드:**
```python
for chunk_id in tqdm(chunk_ids, desc="Loading chunks from database"):
    cursor.execute("SELECT text, source_type, source_id, chunk_index FROM text_chunks WHERE id = ?", (chunk_id,))
    row = cursor.fetchone()
```

**문제점:**
- 각 청크마다 개별 쿼리 실행
- 60,000개 청크면 60,000번 쿼리 실행
- 네트워크/IO 오버헤드가 매우 큼
- 예상 시간: 60,000개 × 0.001초 = 60초 (1분 이상)

**개선 방안:**
- IN 절을 사용하여 배치로 로드
- 한 번의 쿼리로 모든 청크 로드
- 예: `WHERE id IN (?, ?, ..., ?)`
- 배치 크기: 1000개씩 처리

### 4. K-means 클러스터링 자체

**현재 코드:**
```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
```

**문제점:**
- 대량의 데이터(60,000개 이상)에 대해 K-means 실행
- `n_init=10`으로 10번 초기화 시도
- 수렴까지 시간이 오래 걸림
- 예상 시간: 60,000개 데이터 기준 5-10분

**개선 방안:**
- `n_init`을 3으로 줄이기
- `max_iter` 제한 설정 (예: 100)
- Mini-batch K-means 사용 고려
- 또는 샘플링된 데이터로 클러스터링

### 5. Silhouette Score 계산

**현재 코드:**
```python
silhouette_score(embeddings, labels)
```

**문제점:**
- 전체 데이터셋에 대해 계산하면 매우 느림
- O(n²) 복잡도
- 60,000개 데이터면 매우 오래 걸림
- 예상 시간: 5-10분 이상

**개선 방안:**
- 샘플링된 데이터로만 계산 (예: 5,000개)
- 또는 제거 (선택사항)

## 개선 우선순위

### 즉시 적용 가능 (High Priority)

1. **데이터베이스 배치 로드**
   - IN 절 사용으로 변경
   - 예상 개선: 60,000개 청크 기준 1분 → 1초

2. **n_init 감소**
   - 10 → 3으로 변경
   - 예상 개선: 3배 빠름

3. **Elbow method 최적화**
   - 샘플링된 데이터 사용 (예: 10,000개)
   - 최대 K 값 감소 (50 → 20)
   - 예상 개선: 10배 이상 빠름

4. **FAISS 임베딩 추출 최적화**
   - 문서 텍스트에서 재생성 (더 빠를 수 있음)
   - 또는 샘플링 사용
   - 예상 개선: 2-5배 빠름

### 중기 개선 (Medium Priority)

5. **샘플링 옵션 추가**
   - 전체 데이터 대신 샘플 사용 옵션
   - 예상 개선: 데이터 크기에 비례

6. **Silhouette Score 제거 또는 샘플링**
   - 선택사항으로 만들기
   - 샘플링된 데이터로만 계산

### 장기 개선 (Low Priority)

7. **Mini-batch K-means**
   - 대량 데이터에 더 적합
   - 메모리 효율적

8. **병렬 처리**
   - 멀티프로세싱 활용
   - GPU 가속 (가능한 경우)

## 예상 성능 개선

**현재:**
- FAISS 임베딩 추출: 10분
- Elbow method: 20-40분
- 데이터베이스 로드: 1분
- K-means 클러스터링: 5-10분
- Silhouette Score: 5-10분
- **총 예상 시간: 1시간 이상**

**개선 후 (전체 데이터):**
- FAISS 임베딩 재생성: 2-3분
- Elbow method (샘플링): 2-3분
- 데이터베이스 배치 로드: 1초
- K-means 클러스터링 (n_init=3): 2-3분
- Silhouette Score 제거: 0초
- **총 예상 시간: 5-10분**

**개선 후 (샘플링 사용, 10,000개):**
- 임베딩 재생성: 30초
- Elbow method: 30초
- 데이터베이스 로드: 0.5초
- K-means 클러스터링: 30초
- **총 예상 시간: 1-2분**

## 구현 권장사항

1. **샘플링 옵션 추가** (기본값: 전체 데이터)
   - `--sample-size` 옵션 추가
   - 기본값: None (전체), 예: 10000

2. **배치 데이터베이스 로드**
   - IN 절 사용, 배치 크기 1000

3. **n_init=3으로 변경**
   - 모든 K-means 호출에서 적용

4. **Elbow method는 샘플링된 데이터로만 실행**
   - 샘플 크기: min(10000, 전체 데이터의 20%)

5. **Silhouette Score 제거 또는 선택사항**
   - `--calculate-silhouette` 플래그 추가

6. **진행 상황 로깅 강화**
   - 각 단계별 예상 시간 표시
   - 남은 시간 추정

## 코드 수정 예시

### 데이터베이스 배치 로드
```python
# 기존 (느림)
for chunk_id in chunk_ids:
    cursor.execute("SELECT ... WHERE id = ?", (chunk_id,))

# 개선 (빠름)
batch_size = 1000
for i in range(0, len(chunk_ids), batch_size):
    batch_ids = chunk_ids[i:i+batch_size]
    placeholders = ','.join(['?'] * len(batch_ids))
    cursor.execute(f"SELECT ... WHERE id IN ({placeholders})", batch_ids)
```

### n_init 감소
```python
# 기존
kmeans = KMeans(n_clusters=k, n_init=10)

# 개선
kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100)
```

### 샘플링
```python
# 개선
if sample_size and sample_size < len(embeddings):
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    embeddings = embeddings[indices]
    labels = labels[indices]
```
