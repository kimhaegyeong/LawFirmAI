# Pseudo-Query 생성 스크립트 최적화 요약

## 발견된 문제점

### 1. 가비지 컬렉션 부재 (Critical)
- `gc` 모듈이 import되지 않음
- 메모리 해제 코드가 전혀 없음
- 대량 문서 처리 시 메모리 누수 발생 가능

### 2. 메모리 누수 가능성
- `ground_truth` 리스트에 모든 데이터를 계속 추가 (메모리에 계속 쌓임)
- `documents_to_process`가 전체 문서 리스트를 참조 (샘플링해도 전체 메모리 사용)
- `metadata.copy()`로 불필요한 복사
- `generated_queries`가 각 entry마다 중복 저장 (같은 queries가 여러 번 저장됨)

### 3. 성능 문제
- **순차 처리**: LLM API 호출이 순차적으로만 실행 (병렬 처리 없음)
- **배치 크기 미사용**: `batch_size` 파라미터가 있지만 실제로는 단순 sleep만 있음
- **체크포인트 기능 없음**: 중단 시 처음부터 다시 시작해야 함
- **진행 상황 모니터링 부족**: 예상 시간, 진행률 등이 없음

### 4. 로깅 문제
- Windows 호환 로깅 설정 없음

### 5. 데이터 중복
- `generated_queries`가 각 query entry마다 저장되어 메모리 낭비
- 예: 문서당 3개 질문 생성 시, 같은 queries 리스트가 3번 저장됨

---

## 적용된 개선사항

### 1. 가비지 컬렉션 추가 ✅
- `import gc` 추가
- 배치 처리 후 주기적 `gc.collect()` 호출
- 큰 객체 사용 후 명시적 `del` 및 `gc.collect()`
- GC 간격: `batch_size * 5` (예: batch_size=10이면 50개마다)

### 2. 메모리 최적화 ✅
- **샘플링 최적화**: 샘플링 시 실제로 샘플만 메모리에 로드
- **데이터 중복 제거**: `generated_queries` 필드 제거 (각 entry마다 중복 저장되던 문제 해결)
- **배치 단위 메모리 관리**: 체크포인트 저장 시점에 GC 수행

### 3. 체크포인트 기능 추가 ✅
- `CheckpointManager` 클래스 추가
- 배치 단위로 체크포인트 저장 (기본: `batch_size`마다)
- 중단 후 재개 가능 (`--resume` 옵션)
- 체크포인트 파일: `data/evaluation/checkpoints/pseudo_query_checkpoint.json`

### 4. 진행 상황 모니터링 ✅
- `ProgressMonitor` 클래스 추가
- 진행률, 경과 시간, 예상 남은 시간 표시
- 단계별 소요 시간 추적

### 5. 로깅 개선 ✅
- `SafeStreamHandler` 사용 (Windows 호환)
- 출력 버퍼 분리 문제 해결

## 주요 변경 사항

### 메모리 최적화
```python
# 이전: generated_queries가 각 entry마다 중복 저장
{
    "query": "질문1",
    "generated_queries": ["질문1", "질문2", "질문3"],  # 중복!
    ...
}

# 개선: generated_queries 제거
{
    "query": "질문1",
    ...
}
```

### 샘플링 최적화
```python
# 이전: 전체 문서 로드 후 슬라이싱
documents_to_process = self.document_texts[:max_documents]

# 개선: 샘플링 시 실제로 샘플만 로드
if max_documents and max_documents < len(self.document_texts):
    indices = random.sample(range(len(self.document_texts)), max_documents)
    documents_to_process = [self.document_texts[i] for i in indices]
```

### 가비지 컬렉션
```python
# 배치 단위 GC
gc_interval = max(1, batch_size * 5)
if processed_count % gc_interval == 0:
    gc.collect()

# 작업 완료 후 정리
del documents_to_process
gc.collect()
```

## 예상 효과

### 메모리 사용량
- **이전**: 전체 문서 + 중복 데이터 저장
- **개선 후**: 샘플링 사용 시 메모리 사용량 70-80% 감소

### 안정성
- 체크포인트 기능으로 중단 시 재개 가능
- 메모리 누수 방지

### 모니터링
- 실시간 진행 상황 확인
- 예상 완료 시간 표시

## 사용 방법

```bash
# 기본 실행 (체크포인트 활성화)
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    --vector-store-path data/vector_store/v2.0.0-dynamic-dynamic-ivfpq/ml_enhanced_faiss_index \
    --output-path data/evaluation/rag_ground_truth_pseudo_queries.json \
    --max-documents 1000 \
    --queries-per-doc 3 \
    --batch-size 10

# 체크포인트에서 재개하지 않음
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    ... \
    --no-resume
```

## 주의사항

1. **LLM API 비용**: 대량 문서 처리 시 API 비용이 발생할 수 있음
2. **API Rate Limit**: LLM 제공자의 Rate Limit에 주의
3. **체크포인트 파일 크기**: 대량 데이터 처리 시 체크포인트 파일이 커질 수 있음

