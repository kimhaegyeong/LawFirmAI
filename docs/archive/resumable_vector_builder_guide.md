# 중단점 복구 기능이 있는 벡터 임베딩 생성기 사용 가이드

## 개요

텍스트 임베딩 처리 중에 멈춘 경우 이어서 작업할 수 있는 `ResumableVectorBuilder`를 개발했습니다. 이 도구는 정기적으로 체크포인트를 저장하여 중단된 작업을 안전하게 복구할 수 있습니다.

## 주요 기능

### 🔄 중단점 복구 (Resume)
- 정기적으로 체크포인트 저장 (기본: 100개 문서마다)
- 중단된 지점부터 이어서 작업 가능
- 이미 처리된 파일 자동 건너뛰기

### 💾 체크포인트 관리
- `checkpoint.json`: 처리 통계 및 진행 상황
- `progress.pkl`: 벡터 스토어 상태 정보
- 작업 완료 시 체크포인트 파일 자동 정리

### 🛡️ 에러 처리
- 개별 파일/배치 에러 시에도 전체 작업 계속
- 상세한 에러 로그 및 통계
- KeyboardInterrupt (Ctrl+C) 안전 처리

## 사용법

### 1. 기본 사용법

```bash
# 새로운 작업 시작
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/ml_enhanced_resumable \
    --batch-size 10 \
    --chunk-size 100
```

### 2. 중단된 작업 이어서 진행

```bash
# 이전 작업 이어서 진행
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/ml_enhanced_resumable \
    --batch-size 10 \
    --chunk-size 100 \
    --resume
```

### 3. 고급 옵션

```bash
# 체크포인트 간격 조정 (50개 문서마다 저장)
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/ml_enhanced_resumable \
    --batch-size 5 \
    --chunk-size 50 \
    --checkpoint-interval 50 \
    --resume \
    --log-level DEBUG
```

## 명령행 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | 필수 | 입력 디렉토리 (ML-enhanced JSON 파일들) |
| `--output` | 필수 | 출력 디렉토리 (임베딩 및 인덱스) |
| `--batch-size` | 20 | 파일 배치 크기 |
| `--chunk-size` | 200 | 문서 청크 크기 |
| `--checkpoint-interval` | 100 | 체크포인트 저장 간격 (문서 수) |
| `--resume` | False | 이전 작업 이어서 진행 |
| `--log-level` | INFO | 로그 레벨 (DEBUG, INFO, WARNING, ERROR) |

## 작업 흐름

### 1. 첫 실행
```
1. 입력 디렉토리에서 ML-enhanced JSON 파일들 검색
2. 파일들을 배치로 나누어 순차 처리
3. 각 배치를 청크로 나누어 임베딩 생성
4. 정기적으로 체크포인트 저장
5. 완료 시 최종 인덱스 및 통계 저장
```

### 2. 중단 후 재시작
```
1. 체크포인트 파일 확인
2. 이미 처리된 파일들 제외
3. 남은 파일들만 처리
4. 기존 벡터 스토어에 추가
5. 완료 시 체크포인트 파일 정리
```

## 체크포인트 파일 구조

### checkpoint.json
```json
{
  "total_files_processed": 150,
  "total_laws_processed": 1200,
  "total_articles_processed": 5000,
  "total_documents_created": 5000,
  "errors": [],
  "start_time": "2025-10-14T08:00:00",
  "last_checkpoint": "2025-10-14T08:30:00",
  "processed_files": [
    "data/processed/ml_enhanced_law_001.json",
    "data/processed/ml_enhanced_law_002.json"
  ]
}
```

### progress.pkl
```python
{
  'document_count': 5000,
  'index_trained': True
}
```

## 실제 사용 예시

### 시나리오 1: 대용량 데이터 처리
```bash
# 첫 실행 (대용량 데이터)
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/large_dataset \
    --batch-size 5 \
    --chunk-size 50 \
    --checkpoint-interval 50

# 중단 후 재시작
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/large_dataset \
    --batch-size 5 \
    --chunk-size 50 \
    --checkpoint-interval 50 \
    --resume
```

### 시나리오 2: 메모리 제한 환경
```bash
# 메모리 제한 환경에서 안전한 처리
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/memory_limited \
    --batch-size 2 \
    --chunk-size 25 \
    --checkpoint-interval 25 \
    --resume
```

### 시나리오 3: 디버깅 및 모니터링
```bash
# 상세한 로그와 함께 실행
python scripts/build_resumable_vector_db.py \
    --input data/processed \
    --output data/embeddings/debug \
    --batch-size 1 \
    --chunk-size 10 \
    --checkpoint-interval 10 \
    --log-level DEBUG \
    --resume
```

## 안전한 중단 방법

### 1. KeyboardInterrupt (Ctrl+C)
```bash
# 실행 중 Ctrl+C로 안전하게 중단
# 자동으로 체크포인트 저장됨
```

### 2. 프로세스 종료
```bash
# 작업이 중단되어도 체크포인트가 있으면 복구 가능
# 다음 실행 시 --resume 플래그 사용
```

## 문제 해결

### 일반적인 문제들

1. **체크포인트 파일 손상**
   ```bash
   # 체크포인트 파일 삭제 후 처음부터 시작
   rm data/embeddings/output/checkpoint.json
   rm data/embeddings/output/progress.pkl
   ```

2. **메모리 부족**
   ```bash
   # 배치 크기와 청크 크기 줄이기
   --batch-size 1 --chunk-size 10
   ```

3. **디스크 공간 부족**
   ```bash
   # 체크포인트 간격 늘리기
   --checkpoint-interval 500
   ```

### 로그 분석

```bash
# 진행 상황 확인
tail -f logs/vector_builder.log

# 에러 확인
grep "ERROR" logs/vector_builder.log

# 체크포인트 저장 확인
grep "Checkpoint saved" logs/vector_builder.log
```

## 성능 최적화 팁

### 1. 배치 크기 조정
- **메모리 충분**: `--batch-size 20`
- **메모리 제한**: `--batch-size 5`
- **메모리 부족**: `--batch-size 1`

### 2. 청크 크기 조정
- **빠른 처리**: `--chunk-size 200`
- **안정적 처리**: `--chunk-size 100`
- **메모리 절약**: `--chunk-size 50`

### 3. 체크포인트 간격 조정
- **자주 저장**: `--checkpoint-interval 50`
- **균형**: `--checkpoint-interval 100`
- **드물게 저장**: `--checkpoint-interval 500`

## 모니터링

### 진행 상황 확인
```bash
# 실시간 진행 상황
python -c "
import json
with open('data/embeddings/output/checkpoint.json', 'r') as f:
    stats = json.load(f)
print(f'Processed: {stats[\"total_documents_created\"]} documents')
print(f'Files: {stats[\"total_files_processed\"]}')
print(f'Errors: {len(stats[\"errors\"])}')
"
```

### 통계 확인
```bash
# 최종 통계 확인
cat data/embeddings/output/ml_enhanced_stats.json
```

이 가이드를 통해 안전하고 효율적으로 대용량 텍스트 임베딩 처리를 수행할 수 있습니다!
