# PostgreSQL 한국어 텍스트 검색 설정 가이드

## 개요

LawFirmAI 프로젝트에서 PostgreSQL의 한국어 텍스트 검색 기능을 활용하기 위해 **PGroonga**와 **pg_trgm** 확장을 사용합니다.

**PGroonga**는 한국어 형태소 분석을 지원하여 더 정확한 검색이 가능하며, Docker 환경에서는 자동으로 설치됩니다.

## PGroonga 확장 (권장)

**PGroonga**는 한국어 전문 검색을 위한 PostgreSQL 확장입니다.

### 주요 기능

- **한국어 형태소 분석**: `to_tsvector('korean', ...)` 사용 시 형태소 분석 수행
- **정확한 검색**: 조사, 어미 제거로 핵심 키워드 추출
- **성능 최적화**: GIN 인덱스 활용 및 `text_search_vector` 컬럼 지원

### 설치

**Docker 환경**: 자동으로 설치됩니다 (별도 작업 불필요)

**로컬 환경**:
```sql
CREATE EXTENSION IF NOT EXISTS pgroonga;
```

자세한 내용은 [PGroonga 및 tsvector 사용 가이드](../../10_technical_reference/pgroonga_tsvector_guide.md)를 참조하세요.

## pg_trgm 확장

`pg_trgm` (PostgreSQL Trigram)은 trigram 기반의 텍스트 검색을 제공하는 PostgreSQL 확장입니다. 한국어 텍스트 검색에도 효과적으로 사용할 수 있습니다.

### 주요 기능

- **Trigram 기반 검색**: 3글자 단위로 텍스트를 분할하여 검색
- **유사도 검색**: `similarity()` 함수를 사용한 유사도 기반 검색
- **인덱스 지원**: `gin_trgm_ops` 연산자를 사용한 GIN 인덱스

## 자동 설치

### Docker Compose 사용 시

Docker Compose 파일에 확장 설치 스크립트가 자동으로 마운트되어 있습니다:

```yaml
volumes:
  - ../scripts/migrations/init_postgres_extensions.sql:/docker-entrypoint-initdb.d/01_init_extensions.sql:ro
```

PostgreSQL 컨테이너가 처음 시작될 때 `/docker-entrypoint-initdb.d/` 디렉토리의 SQL 파일이 자동으로 실행됩니다.

### 수동 설치

이미 생성된 데이터베이스에 확장을 설치하려면:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## 스키마 초기화

Open Law API 데이터 수집을 위한 스키마를 초기화하면 자동으로 확장이 설치됩니다:

```bash
python scripts/migrations/init_open_law_schema.py
```

## 인덱스 구조

스키마 초기화 시 다음 인덱스가 생성됩니다:

### Full-Text Search 인덱스 (GIN)
- `idx_statutes_fts`: 법령명 및 약칭 검색
- `idx_articles_fts`: 조문 내용 검색
- `idx_precedents_fts`: 판례 사건명 검색
- `idx_precedent_contents_fts`: 판례 본문 검색

### Trigram 인덱스 (GIN)
- `idx_statutes_trgm`: 법령명 trigram 검색
- `idx_articles_trgm`: 조문 내용 trigram 검색
- `idx_precedents_trgm`: 판례 사건명 trigram 검색
- `idx_precedent_contents_trgm`: 판례 본문 trigram 검색

## 검색 예시

### Full-Text Search 사용 (PGroonga)

```sql
-- 법령명 검색 (PGroonga 사용, 'korean' 설정)
SELECT * FROM statutes
WHERE to_tsvector('korean', law_name_kr || ' ' || COALESCE(law_abbrv, ''))
      @@ to_tsquery('korean', '민법');

-- 조문 내용 검색 (text_search_vector 컬럼 활용, 권장)
SELECT * FROM statutes_articles
WHERE text_search_vector @@ plainto_tsquery('korean', '계약 해지')
ORDER BY ts_rank_cd(text_search_vector, plainto_tsquery('korean', '계약 해지')) DESC;

-- 조문 내용 검색 (text_search_vector 컬럼이 없는 경우)
SELECT * FROM statutes_articles
WHERE to_tsvector('korean', article_content)
      @@ plainto_tsquery('korean', '계약 해지');
```

### Trigram 유사도 검색 사용

```sql
-- 유사도 기반 검색 (0.3 이상 유사도)
SELECT *, similarity(law_name_kr, '민법') AS sim
FROM statutes
WHERE similarity(law_name_kr, '민법') > 0.3
ORDER BY sim DESC;

-- Trigram 인덱스 활용 검색
SELECT * FROM statutes
WHERE law_name_kr % '민법';  -- % 연산자는 유사도 검색
```

### 복합 검색

```sql
-- Full-Text Search와 Trigram 검색 결합 (PGroonga 사용)
SELECT * FROM statutes_articles
WHERE (
    text_search_vector @@ plainto_tsquery('korean', '계약')
    OR article_content % '계약'
)
ORDER BY ts_rank_cd(text_search_vector, plainto_tsquery('korean', '계약')) DESC;
```

## 성능 최적화

### 인덱스 사용 확인

```sql
-- 인덱스 사용 여부 확인
EXPLAIN ANALYZE
SELECT * FROM statutes
WHERE law_name_kr % '민법';
```

### 통계 정보 업데이트

```sql
-- 인덱스 통계 정보 업데이트
ANALYZE statutes;
ANALYZE statutes_articles;
ANALYZE precedents;
ANALYZE precedent_contents;
```

## 참고 자료

### 프로젝트 내 문서

- **[PGroonga 및 tsvector 사용 가이드](../../10_technical_reference/pgroonga_tsvector_guide.md)**: 상세한 사용 방법 및 예시
- [tsvector 사용 현황 검토 보고서](../../10_technical_reference/tsvector_review_report.md)
- [Rank Score 계산 가이드](../../10_technical_reference/rank_score_calculation_guide.md)

### 공식 문서

- [PGroonga 공식 문서](https://pgroonga.github.io/)
- [PGroonga 설치 가이드](https://pgroonga.github.io/install/)
- [PostgreSQL pg_trgm 문서](https://www.postgresql.org/docs/current/pgtrgm.html)
- [PostgreSQL Full-Text Search 문서](https://www.postgresql.org/docs/current/textsearch.html)
- [PostgreSQL GIN 인덱스 문서](https://www.postgresql.org/docs/current/gin.html)

