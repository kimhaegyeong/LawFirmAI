# Open Law API 데이터 수집

국가법령정보 공동활용 LAW OPEN DATA API를 통해 현행법령 및 판례 데이터를 수집합니다.

## 사전 준비

### 1. PostgreSQL 데이터베이스 설정

```bash
# 환경 변수 설정
export DATABASE_URL="postgresql://user:password@host:5432/lawfirmai"
```

### 2. 스키마 생성

```bash
# SQL 파일로 직접 생성
psql -U username -d lawfirmai -f scripts/migrations/create_open_law_schema.sql

# 또는 Python 스크립트 사용
python scripts/migrations/init_open_law_schema.py --db $DATABASE_URL
```

### 3. API 인증 정보 설정

```bash
# 환경 변수 설정
export LAW_OPEN_API_OC="your_email_id"  # g4c@korea.kr일 경우 oc=g4c
```

## 사용 방법

### 민사법 현행법령 수집

#### 1단계: 법령 목록 수집

```bash
python scripts/ingest/open_law/scripts/collect_civil_statutes.py \
    --oc YOUR_OC \
    --phase list \
    --output data/raw/open_law/civil_statutes_list.json
```

#### 2단계: 법령 본문 및 조문 수집

```bash
python scripts/ingest/open_law/scripts/collect_civil_statutes.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/civil_statutes_list.json \
    --db $DATABASE_URL
```

### 형법 현행법령 수집

#### 1단계: 법령 목록 수집

```bash
python scripts/ingest/open_law/scripts/collect_criminal_statutes.py \
    --oc YOUR_OC \
    --phase list \
    --output data/raw/open_law/criminal_statutes_list.json
```

#### 2단계: 법령 본문 및 조문 수집

```bash
python scripts/ingest/open_law/scripts/collect_criminal_statutes.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/criminal_statutes_list.json \
    --db $DATABASE_URL
```

### 민사법 판례 수집

#### 1단계: 판례 목록 수집

```bash
python scripts/ingest/open_law/scripts/collect_civil_precedents.py \
    --oc YOUR_OC \
    --phase list \
    --max-pages 200 \
    --output data/raw/open_law/civil_precedents_list.json
```

#### 2단계: 판례 본문 수집

```bash
python scripts/ingest/open_law/scripts/collect_civil_precedents.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/civil_precedents_list.json \
    --db $DATABASE_URL
```

### 형법 판례 수집

#### 1단계: 판례 목록 수집

```bash
python scripts/ingest/open_law/scripts/collect_criminal_precedents.py \
    --oc YOUR_OC \
    --phase list \
    --max-pages 100 \
    --output data/raw/open_law/criminal_precedents_list.json
```

#### 2단계: 판례 본문 수집

```bash
python scripts/ingest/open_law/scripts/collect_criminal_precedents.py \
    --oc YOUR_OC \
    --phase content \
    --input data/raw/open_law/criminal_precedents_list.json \
    --db $DATABASE_URL
```

## 옵션

### 공통 옵션

- `--oc`: 사용자 이메일 ID (환경변수: `LAW_OPEN_API_OC`)
- `--phase`: 수집 단계 (`list` 또는 `content`)
- `--db`: PostgreSQL 데이터베이스 URL (환경변수: `DATABASE_URL`)
- `--rate-limit`: API 요청 간 지연 시간 (초, 기본값: 0.5)

### 목록 수집 옵션 (phase=list)

- `--output`: 법령/판례 목록 저장 경로
- `--max-pages`: 최대 페이지 수

### 본문 수집 옵션 (phase=content)

- `--input`: 법령/판례 목록 JSON 파일 경로

## 데이터베이스 스키마

수집된 데이터는 다음 테이블에 저장됩니다:

- `statutes`: 법령 메타데이터
- `statutes_articles`: 조문 데이터
- `precedents`: 판례 메타데이터
- `precedent_contents`: 판례 본문

자세한 스키마는 `scripts/migrations/create_open_law_schema.sql` 참조

## 로그

수집 과정의 로그는 다음 위치에 저장됩니다:

- `logs/open_law/civil_statutes_collection.log`
- `logs/open_law/criminal_statutes_collection.log`
- `logs/open_law/civil_precedents_collection.log`
- `logs/open_law/criminal_precedents_collection.log`

## 배치 실행 (전체 프로세스 자동화)

전체 수집 프로세스를 한 번에 실행할 수 있습니다:

```bash
# 전체 수집 프로세스 실행
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc YOUR_OC \
    --db $DATABASE_URL

# 스키마 초기화 건너뛰기
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc YOUR_OC \
    --db $DATABASE_URL \
    --skip-schema

# 특정 단계만 실행
python scripts/ingest/open_law/scripts/run_collection_batch.py \
    --oc YOUR_OC \
    --db $DATABASE_URL \
    --step civil_statutes
```

## 진행 상황 확인

수집된 데이터의 통계를 확인할 수 있습니다:

```bash
# 전체 통계
python scripts/ingest/open_law/scripts/check_collection_status.py --db $DATABASE_URL

# 특정 분야만 확인
python scripts/ingest/open_law/scripts/check_collection_status.py \
    --db $DATABASE_URL \
    --domain civil_law
```

## 데이터 검증

수집된 데이터의 품질을 검증할 수 있습니다:

```bash
# 전체 데이터 검증
python scripts/ingest/open_law/scripts/validate_data.py --db $DATABASE_URL

# 특정 분야만 검증
python scripts/ingest/open_law/scripts/validate_data.py \
    --db $DATABASE_URL \
    --domain civil_law
```

## 관련 문서

- [빠른 시작 가이드](./QUICK_START.md)
- [환경 설정 가이드](./SETUP.md)
- [판례 데이터 청킹 전략](./CHUNKING.md)
- [FAISS vs pgvector 비교 테스트 개발 계획](./EMBEDDING_COMPARISON_PLAN.md)
- [빠른 수집 계획](../collection/rapid_collection_plan.md)
- [Open Law API 가이드](../guide_id_map.md)

