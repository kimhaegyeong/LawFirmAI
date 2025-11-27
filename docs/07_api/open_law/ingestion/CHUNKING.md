# 판례 데이터 청킹 전략

## 개요

PostgreSQL `precedent_contents` 테이블의 판례 데이터를 섹션 타입별로 차등 청킹하는 새로운 전략입니다.

## 청킹 전략

### 섹션 타입별 차등 청킹

| 섹션 타입 | 청킹 전략 | 설정 |
|----------|----------|------|
| **판시사항** | 최소 청킹 | 500자 이하면 청킹 안 함 |
| **판결요지** | 중간 청킹 | 1000자 단위, 200자 overlap |
| **판례내용** | 적극적 청킹 | 800자 단위, 150자 overlap |

### 레거시 전략과의 차이

- **레거시 전략**: SQLite `case_paragraphs` 테이블용 (기존 시스템 유지)
- **새 전략**: PostgreSQL `precedent_contents` 테이블용 (섹션 타입별 차등 청킹)

## 사용 방법

### 1. 스키마 생성

```bash
python scripts/migrations/init_open_law_schema.py
```

또는 직접 SQL 실행:

```bash
psql -U username -d database_name -f scripts/migrations/create_open_law_schema.sql
```

### 2. 판례 데이터 청킹

```bash
python scripts/ingest/open_law/chunk_precedents.py \
    --db postgresql://user:pass@host:5432/dbname \
    --batch-size 100 \
    --limit 1000
```

### 3. 청킹 결과 확인

#### 방법 1: 검증 스크립트 사용 (권장)

```bash
# 민사법 청킹 상태 확인
python scripts/ingest/open_law/verify_chunking_status.py --domain civil_law

# 전체 도메인 청킹 상태 확인
python scripts/ingest/open_law/verify_chunking_status.py

# 데이터베이스 URL 직접 지정
python scripts/ingest/open_law/verify_chunking_status.py \
    --db postgresql://user:pass@host:5432/dbname \
    --domain civil_law
```

검증 스크립트는 다음 정보를 제공합니다:
- 전체 통계 (판례 수, 내용 수, 청킹 완료율)
- 도메인별 통계 (civil_law, criminal_law)
- 섹션 타입별 청킹 통계
- 청킹되지 않은 데이터 확인
- 최근 청킹된 데이터 확인

#### 방법 2: SQL 쿼리 직접 실행

```sql
-- 청킹 통계
SELECT 
    section_type,
    COUNT(DISTINCT precedent_content_id) as content_count,
    COUNT(*) as chunk_count,
    AVG(chunk_length) as avg_chunk_length,
    MIN(chunk_length) as min_chunk_length,
    MAX(chunk_length) as max_chunk_length
FROM precedent_chunks pc
JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id
GROUP BY section_type;

-- 민사법 청킹 완료 여부 확인
SELECT 
    COUNT(DISTINCT pc.id) as total_contents,
    COUNT(DISTINCT CASE WHEN pch.id IS NOT NULL THEN pc.id END) as chunked_contents,
    COUNT(DISTINCT CASE WHEN pch.id IS NULL THEN pc.id END) as unchunked_contents
FROM precedent_contents pc
JOIN precedents p ON pc.precedent_id = p.id
LEFT JOIN precedent_chunks pch ON pc.id = pch.precedent_content_id
WHERE p.domain = 'civil_law';
```

## 데이터 구조

### precedent_chunks 테이블

```sql
CREATE TABLE precedent_chunks (
    id SERIAL PRIMARY KEY,
    precedent_content_id INTEGER NOT NULL REFERENCES precedent_contents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_content TEXT NOT NULL,
    chunk_length INTEGER,
    metadata JSONB,  -- 판례 메타데이터
    embedding_vector VECTOR(768),  -- 벡터 임베딩 (pgvector)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(precedent_content_id, chunk_index)
);
```

### 메타데이터 구조

```json
{
    "precedent_content_id": 123,
    "section_type": "판례내용",
    "precedent_id": 456,
    "case_name": "사건명",
    "case_number": "2020가1234",
    "decision_date": "2020-01-01",
    "court_name": "대법원",
    "domain": "civil_law",
    "referenced_articles": "민법 제750조",
    "referenced_precedents": "대법원 2019다123456"
}
```

## 성능 최적화

- 배치 처리: 기본 100개씩 처리
- 인덱스: `precedent_content_id`, `chunk_content` (FTS)
- 중복 방지: `ON CONFLICT DO NOTHING` 사용

## 주의사항

1. **pgvector 확장**: 벡터 임베딩을 사용하려면 pgvector 확장이 필요합니다.
2. **레거시 시스템**: SQLite `case_paragraphs`는 기존 청킹 전략을 계속 사용합니다.
3. **청킹 재실행**: 이미 청킹된 데이터는 건너뜁니다 (중복 방지).

## 관련 문서

- [빠른 시작 가이드](./QUICK_START.md)
- [환경 설정 가이드](./SETUP.md)
- [데이터 수집 가이드](./README.md)

