# Open Law API 데이터 수집 빠른 시작 가이드

## 0. 실행 전 체크

먼저 환경 설정이 올바른지 확인하세요:

```bash
# 실행 전 체크
python scripts/ingest/open_law/scripts/preflight_check.py --skip-api-test
```

문제가 발견되면 [SETUP.md](./SETUP.md)를 참고하여 환경을 설정하세요.

## 1. 환경 설정

### PostgreSQL 데이터베이스 준비

```bash
# 환경 변수 설정 (Windows PowerShell)
$env:DATABASE_URL="postgresql://user:password@host:5432/lawfirmai"
$env:LAW_OPEN_API_OC="your_email_id"  # g4c@korea.kr일 경우 oc=g4c

# 환경 변수 설정 (Linux/macOS)
export DATABASE_URL="postgresql://user:password@host:5432/lawfirmai"
export LAW_OPEN_API_OC="your_email_id"
```

또는 프로젝트 루트에 `.env` 파일을 생성하여 설정할 수 있습니다. 자세한 내용은 [SETUP.md](./SETUP.md)를 참고하세요.

### 스키마 생성

```bash
# 스키마 초기화
python scripts/migrations/init_open_law_schema.py --db $env:DATABASE_URL  # PowerShell
# 또는
python scripts/migrations/init_open_law_schema.py --db $DATABASE_URL  # Linux/macOS
```

## 2. 전체 수집 프로세스 실행

가장 간단한 방법은 배치 스크립트를 사용하는 것입니다:

```bash
# 전체 수집 프로세스 자동 실행
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc $LAW_OPEN_API_OC \
    --db $DATABASE_URL
```

이 명령은 다음을 순차적으로 실행합니다:
1. PostgreSQL 스키마 초기화
2. 민사법 현행법령 수집 (목록 → 본문)
3. 형법 현행법령 수집 (목록 → 본문)
4. 민사법 판례 수집 (목록 → 본문)
5. 형법 판례 수집 (목록 → 본문)

## 3. 단계별 실행

특정 단계만 실행하려면:

```bash
# 특정 단계만 실행
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc $LAW_OPEN_API_OC \
    --db $DATABASE_URL \
    --step civil_statutes  # 또는 criminal_statutes, civil_precedents, criminal_precedents
```

## 4. 진행 상황 확인

```bash
# 수집 진행 상황 확인
python scripts/ingest/open_law/scripts/check_collection_status.py --db $DATABASE_URL
```

## 5. 데이터 검증

```bash
# 데이터 품질 검증
python scripts/ingest/open_law/scripts/validate_data.py --db $DATABASE_URL
```

## 예상 소요 시간

- **민사법 현행법령**: 약 5-7일
- **형법 현행법령**: 약 3-5일
- **민사법 판례**: 약 5-7일
- **형법 판례**: 약 3-5일
- **전체**: 약 16-24일 (약 3-4주)

## 문제 해결

### 스키마가 이미 존재하는 경우

```bash
# --skip-schema 옵션 사용
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc $LAW_OPEN_API_OC \
    --db $DATABASE_URL \
    --skip-schema
```

### 특정 단계에서 실패한 경우

해당 단계만 다시 실행:

```bash
# 예: 민사법 현행법령만 다시 수집
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc $LAW_OPEN_API_OC \
    --db $DATABASE_URL \
    --skip-schema \
    --step civil_statutes
```

### 로그 확인

각 수집 단계의 로그는 다음 위치에 저장됩니다:
- `logs/open_law/civil_statutes_collection.log`
- `logs/open_law/criminal_statutes_collection.log`
- `logs/open_law/civil_precedents_collection.log`
- `logs/open_law/criminal_precedents_collection.log`
- `logs/open_law/batch_collection.log` (배치 실행 로그)

## 다음 단계

수집이 완료되면:
1. 데이터 검증 실행
2. [판례 데이터 청킹](./CHUNKING.md) 실행
3. [벡터 임베딩 생성 및 비교 테스트](./EMBEDDING_COMPARISON_PLAN.md) (FAISS vs pgvector)
4. 검색 인덱스 구축
5. 통합 테스트

