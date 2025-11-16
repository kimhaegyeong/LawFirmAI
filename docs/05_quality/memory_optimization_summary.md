# 메모리 최적화 및 성능 개선 요약

## 적용된 개선사항

### 1. 메모리 누수 방지

#### 가비지 컬렉션 추가
- `import gc` 추가
- 각 단계 완료 후 `gc.collect()` 호출
- 큰 객체 사용 후 명시적 `del` 및 `gc.collect()`

#### 명시적 메모리 해제
```python
# 임베딩 생성 후
del texts_to_process
gc.collect()

# 샘플링 후
del sample_embeddings, sample_indices
gc.collect()

# 클러스터링 후
del embeddings
gc.collect()

# Ground Truth 생성 중
if processed % batch_size == 0:
    gc.collect()
```

### 2. 성능 개선

#### 샘플링 최적화
- **이전**: 전체 문서 텍스트 로드 → 전체 임베딩 생성 → 샘플링
- **개선**: 샘플링 먼저 수행 → 샘플링된 문서만 임베딩 생성

```python
# 샘플링을 먼저 수행하여 불필요한 임베딩 생성 방지
if sample_size and sample_size < len(self.document_texts):
    indices = np.random.choice(len(self.document_texts), sample_size, replace=False)
    texts_to_process = [self.document_texts[i] for i in indices]
    embeddings = self.vector_store.generate_embeddings(texts_to_process, batch_size=64)
```

#### 배치 처리 최적화
- Ground Truth 생성 시 1000개마다 가비지 컬렉션 수행
- FAISS 인덱스 추출 시 10,000개마다 가비지 컬렉션 수행

### 3. 메모리 사용량 감소

#### 중간 변수 정리
- `cluster_dict` → `valid_clusters` 변환 후 즉시 삭제
- `sample_embeddings`, `sample_indices` 사용 후 즉시 삭제
- `embeddings` 배열 사용 후 즉시 삭제

#### 배치 단위 메모리 관리
- Ground Truth 생성: 1000개마다 GC
- FAISS 추출: 10,000개마다 GC

## 예상 효과

### 메모리 사용량
- **이전**: 전체 데이터셋 크기의 2-3배 메모리 사용
- **개선 후**: 샘플링 사용 시 메모리 사용량 70-80% 감소

### 실행 속도
- **샘플링 최적화**: 임베딩 생성 시간 80-90% 단축 (샘플링 사용 시)
- **메모리 관리**: 메모리 부족으로 인한 스왑 감소 → 전체 속도 향상

### 예시 (10,000개 샘플링)
- **이전**: 전체 60,000개 임베딩 생성 (약 30분) → 샘플링
- **개선 후**: 10,000개만 임베딩 생성 (약 5분)

## 추가 권장사항

### 1. 더 큰 샘플링 사용
- 전체 데이터가 필요하지 않은 경우 샘플링 크기 증가
- 예: `--sample-size 20000` 또는 `--sample-size 50000`

### 2. 체크포인트 활용
- 장시간 실행 시 체크포인트 기능 활용
- 중단 후 `--resume` 옵션으로 재개

### 3. 배치 크기 조정
- 메모리가 충분한 경우 `batch_size` 증가 (64 → 128)
- 메모리가 부족한 경우 `batch_size` 감소 (64 → 32)

## 모니터링

실행 중 메모리 사용량을 모니터링하려면:
```bash
# Windows PowerShell
Get-Process python | Select-Object Id, ProcessName, @{Name="Memory(MB)";Expression={[math]::Round($_.WS/1MB,2)}}
```

또는 작업 관리자에서 Python 프로세스의 메모리 사용량 확인

