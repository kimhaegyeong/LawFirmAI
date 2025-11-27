# 벡터 임베딩 최적화 가이드

## 개요

민사법 벡터 임베딩 생성 코드의 메모리 관리 및 리소스 활용 최적화 가이드입니다.

## 개선 사항

### 1. 가비지 컬렉션 추가

**문제점**: 배치 처리 후 메모리가 자동으로 해제되지 않아 메모리 누수 발생 가능

**해결책**: 
- `gc` 모듈 임포트 추가
- 배치 처리 후 `gc.collect()` 호출
- 큰 변수 명시적 삭제 (`del`)

**적용 파일**:
- `scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py`
- `scripts/ingest/open_law/embedding/faiss/faiss_embedder.py`

### 2. 메모리 최적화

#### pgvector_embedder.py

**개선 내용**:
```python
# 배치 처리 후 메모리 정리
del embeddings
del texts
del chunk_ids
del chunks_to_process
del chunks
gc.collect()
```

**효과**:
- 배치 처리 후 즉시 메모리 해제
- 대량 데이터 처리 시 메모리 사용량 감소
- 장시간 실행 시 안정성 향상

#### faiss_embedder.py

**개선 내용**:
```python
# 중간 변수 삭제
del embeddings
del texts
del chunk_ids
del metadata
del items
gc.collect()

# 임베딩 배열 결합 후 개별 리스트 삭제
del self.embeddings
gc.collect()

# 인덱스 생성 후 임베딩 배열 삭제
del all_embeddings
del index
gc.collect()
```

**효과**:
- 메모리에 모든 임베딩을 저장하는 문제 완화
- 인덱스 생성 후 즉시 메모리 해제
- 대량 데이터 처리 시 메모리 부족 방지

### 3. 에러 처리 시 메모리 정리

**개선 내용**:
- 예외 발생 시에도 메모리 정리 수행
- `locals()`를 사용하여 변수 존재 여부 확인 후 삭제

**효과**:
- 에러 발생 시에도 메모리 누수 방지
- 장시간 실행 시 안정성 향상

## 성능 최적화 권장 사항

### 1. 배치 크기 조정

**현재 기본값**: `batch_size=100`

**권장 사항**:
- 메모리가 충분한 경우: `batch_size=200~500`
- 메모리가 제한적인 경우: `batch_size=50~100`
- GPU 사용 시: `batch_size=500~1000`

### 2. 멀티스레딩/멀티프로세싱 활용 (향후 개선)

**현재 상태**: 단일 스레드 처리

**개선 방향**:
- 임베딩 생성 단계에서 멀티프로세싱 활용
- 데이터베이스 저장 단계는 단일 스레드 유지 (트랜잭션 안정성)

**주의사항**:
- SentenceTransformer 모델은 멀티프로세싱 시 모델 복사 필요
- 메모리 사용량 증가 가능
- CPU 코어 수에 따라 성능 향상 정도 결정

### 3. 메모리 모니터링

**권장 사항**:
- 대량 데이터 처리 시 메모리 사용량 모니터링
- 배치 크기 동적 조정 (메모리 사용량에 따라)

## 사용 예시

### pgvector 임베딩 생성

```bash
python scripts/ingest/open_law/embedding/pgvector/pgvector_embedder.py \
    --db postgresql://user:pass@host:5432/dbname \
    --data-type precedents \
    --domain civil_law \
    --batch-size 200
```

### FAISS 임베딩 생성

```bash
python scripts/ingest/open_law/embedding/faiss/faiss_embedder.py \
    --db postgresql://user:pass@host:5432/dbname \
    --data-type precedents \
    --domain civil_law \
    --batch-size 200 \
    --output-dir data/embeddings/open_law_postgresql
```

## 모니터링

### 메모리 사용량 확인

```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")
```

### 가비지 컬렉션 통계

```python
import gc

gc.collect()
stats = gc.get_stats()
print(f"가비지 컬렉션 통계: {stats}")
```

## 관련 문서

- [청킹 전략](./CHUNKING.md)
- [임베딩 비교 계획](./EMBEDDING_COMPARISON_PLAN.md)

