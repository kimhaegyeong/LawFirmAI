# PGroonga 및 tsvector 사용 가이드

## 📋 개요

LawFirmAI 프로젝트에서 PostgreSQL의 한국어 전문 검색을 위해 **PGroonga**와 **tsvector**를 사용합니다. PGroonga는 한국어 형태소 분석을 지원하여 더 정확한 한국어 텍스트 검색이 가능합니다.

**✅ PostgreSQL 18 지원**: [PGroonga 4.0.4](https://github.com/pgroonga/pgroonga/releases/tag/4.0.4) (2025-10-02 릴리즈)부터 PostgreSQL 18을 공식 지원합니다. PostgreSQL 18의 `index_beginscan` API 변경사항과 ordered index scan 기능을 지원합니다.

**검토 일자**: 2025-01-XX  
**적용 버전**: PostgreSQL 18+ with PGroonga 4.0.4+

---

## 🎯 주요 특징

### PGroonga의 장점

1. **한국어 형태소 분석 지원**
   - `to_tsvector('korean', ...)` 사용 시 한국어 형태소 분석 수행
   - 조사, 어미 등을 제거하여 핵심 키워드 추출
   - 검색 정확도 향상

2. **성능 최적화**
   - GIN 인덱스 활용
   - `text_search_vector` 컬럼을 통한 인덱스 직접 사용
   - 실시간 형태소 분석 지원

3. **Docker 환경 자동 설치**
   - Docker PostgreSQL 이미지에 PGroonga 포함
   - 별도 설치 불필요

---

## 🔧 설치 및 설정

### Docker 환경 (권장)

Docker PostgreSQL 이미지는 자동으로 PGroonga를 포함합니다:

```bash
# Docker 이미지 빌드
docker-compose -f deployment/docker-compose.dev.yml build postgres

# 컨테이너 시작
docker-compose -f deployment/docker-compose.dev.yml up -d postgres
```

**Dockerfile 위치**: `deployment/postgres/Dockerfile`

**PostgreSQL 18 사용 시**: 
- [PGroonga 4.0.4](https://github.com/pgroonga/pgroonga/releases/tag/4.0.4) 이상 버전을 사용합니다 (PostgreSQL 18 공식 지원)
- PostgreSQL 18의 `index_beginscan` API 변경사항을 지원합니다
- Ordered index scan 기능 지원으로 `WHERE ... ORDER BY ... LIMIT` 쿼리 성능 향상
- 4.0.4가 없으면 최신 릴리스를 자동으로 사용합니다

### 수동 설치

로컬 PostgreSQL에 PGroonga를 설치하려면:

```bash
# Ubuntu/Debian
sudo apt-get install -y postgresql-18-pgroonga

# 또는 소스에서 빌드
# https://pgroonga.github.io/install/
```

### 확장 활성화

PostgreSQL 데이터베이스에서 확장을 활성화합니다:

```sql
-- PGroonga 확장 설치 (필수)
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- 기타 확장
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
```

**자동 설치**: Docker 환경에서는 `scripts/migrations/schema/init_postgres_extensions.sql`이 자동으로 실행됩니다.

---

## 📊 사용 방법

### 코드에서의 사용

`LegalDataConnectorV2` 클래스는 자동으로 PGroonga를 감지하고 `'korean'` 설정을 사용합니다:

```python
from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2

# 초기화 시 PGroonga 자동 감지
connector = LegalDataConnectorV2()

# 검색 시 자동으로 to_tsvector('korean', ...) 사용
results = connector.search_statutes_fts("계약 해지 사유", limit=20)
```

### 내부 동작

1. **PGroonga 감지**: `_check_pgroonga_available()` 메서드로 확장 존재 여부 확인
2. **자동 설정**: PGroonga가 있으면 `'korean'` 설정 사용, 없으면 경고 후 `'korean'` 설정 시도
3. **쿼리 생성**: `_convert_fts5_to_postgresql_fts()` 메서드가 적절한 쿼리 생성

---

## 🔍 SQL 쿼리 예시

### 기본 검색

```sql
-- 법령 조문 검색 (PGroonga 사용)
SELECT 
    sa.id,
    sa.article_no,
    sa.article_content,
    ts_rank_cd(
        to_tsvector('korean', sa.article_content),
        plainto_tsquery('korean', '계약 해지')
    ) as rank_score
FROM statutes_articles sa
WHERE to_tsvector('korean', sa.article_content) 
      @@ plainto_tsquery('korean', '계약 해지')
ORDER BY rank_score DESC
LIMIT 20;
```

### text_search_vector 컬럼 활용 (권장)

```sql
-- text_search_vector 컬럼이 있는 경우 (인덱스 직접 활용)
SELECT 
    sa.id,
    sa.article_no,
    sa.article_content,
    ts_rank_cd(
        sa.text_search_vector,
        plainto_tsquery('korean', '계약 해지')
    ) as rank_score
FROM statutes_articles sa
WHERE sa.text_search_vector 
      @@ plainto_tsquery('korean', '계약 해지')
ORDER BY rank_score DESC
LIMIT 20;
```

### OR 조건 검색

```sql
-- OR 조건 지원
SELECT 
    sa.id,
    sa.article_no,
    sa.article_content,
    ts_rank_cd(
        sa.text_search_vector,
        to_tsquery('korean', '계약 | 해지')
    ) as rank_score
FROM statutes_articles sa
WHERE sa.text_search_vector 
      @@ to_tsquery('korean', '계약 | 해지')
ORDER BY rank_score DESC
LIMIT 20;
```

---

## 🏗️ 인덱스 구조

### text_search_vector 컬럼

다음 테이블에는 `text_search_vector` 컬럼이 자동으로 생성됩니다:

- `statute_articles.text_search_vector` (v2 스키마)
- `case_paragraphs.text_search_vector` (v2 스키마)
- `decision_paragraphs.text_search_vector` (v2 스키마)
- `interpretation_paragraphs.text_search_vector` (v2 스키마)

**Open Law 스키마**: `statutes_articles`, `precedent_contents` 등은 `text_search_vector` 컬럼이 없을 수 있습니다. 이 경우 `to_tsvector('korean', ...)`를 직접 사용합니다.

### GIN 인덱스

```sql
-- text_search_vector 컬럼용 GIN 인덱스
CREATE INDEX IF NOT EXISTS idx_statute_articles_fts 
ON statute_articles USING gin(text_search_vector);

-- 또는 동적 생성 (Open Law 스키마)
CREATE INDEX IF NOT EXISTS idx_articles_fts 
ON statutes_articles USING gin(to_tsvector('korean', article_content));
```

---

## ⚙️ 트리거 함수

`text_search_vector` 컬럼은 트리거 함수를 통해 자동으로 업데이트됩니다:

```sql
-- 예시: statute_articles 테이블 트리거
CREATE OR REPLACE FUNCTION update_statute_articles_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_statute_articles_fts
BEFORE INSERT OR UPDATE ON statute_articles
FOR EACH ROW
EXECUTE FUNCTION update_statute_articles_fts();
```

**중요**: 트리거 함수에서도 `'korean'` 설정을 사용해야 합니다.

---

## 🔄 코드 구조

### _convert_fts5_to_postgresql_fts 메서드

```python
def _convert_fts5_to_postgresql_fts(
    self, 
    query: str, 
    table_alias: str = 'sa',
    text_vector_column: str = 'text_search_vector',
    text_content_column: str = None,
    table_name: str = None,
    use_pgroonga: Optional[bool] = None
) -> tuple[str, str, str, str]:
    """
    쿼리를 PostgreSQL tsvector 쿼리로 변환 (PGroonga 지원)
    
    Returns:
        (WHERE 절, ORDER BY 절, rank_score 표현식, tsquery 문자열) 튜플
    """
    # PGroonga 사용 여부 결정
    if use_pgroonga is None:
        use_pgroonga = self._check_pgroonga_available()
    
    # 항상 'korean' 설정 사용
    lang_config = 'korean'
    
    # tsvector 표현식 생성
    if text_vector_column:
        tsvector_expr = f"{table_alias}.{text_vector_column}"
    elif text_content_column:
        tsvector_expr = f"to_tsvector('{lang_config}', {table_alias}.{text_content_column})"
    else:
        tsvector_expr = f"{table_alias}.text_search_vector"
    
    # WHERE 절, ORDER BY 절, rank_score 표현식 생성
    where_clause = f"{tsvector_expr} @@ plainto_tsquery('{lang_config}', %s)"
    rank_score_expr = f"ts_rank_cd({tsvector_expr}, plainto_tsquery('{lang_config}', %s))"
    order_clause = f"{rank_score_expr} DESC"
    
    return where_clause, order_clause, rank_score_expr, query_clean
```

### PGroonga 감지 메서드

```python
def _check_pgroonga_available(self) -> bool:
    """
    PGroonga 확장이 설치되어 있고 사용 가능한지 확인
    
    Returns:
        PGroonga 사용 가능 여부
    """
    # pg_extension 테이블에서 확인
    # pg_proc 테이블에서 함수 존재 여부 확인
    # ...
```

---

## 📈 성능 최적화

### 1. text_search_vector 컬럼 활용

**권장**: `text_search_vector` 컬럼이 있으면 항상 사용

```sql
-- ✅ 좋은 예: 인덱스 직접 활용
WHERE sa.text_search_vector @@ plainto_tsquery('korean', '검색어')

-- ❌ 나쁜 예: 매번 tsvector 재계산
WHERE to_tsvector('korean', sa.article_content) @@ plainto_tsquery('korean', '검색어')
```

### 2. 인덱스 사용 확인

```sql
-- 실행 계획 확인
EXPLAIN ANALYZE
SELECT sa.id, sa.article_content
FROM statutes_articles sa
WHERE sa.text_search_vector @@ plainto_tsquery('korean', '계약 해지')
ORDER BY ts_rank_cd(sa.text_search_vector, plainto_tsquery('korean', '계약 해지')) DESC
LIMIT 20;

-- 예상 결과: Bitmap Index Scan on idx_statute_articles_fts
```

### 3. 통계 정보 업데이트

```sql
-- 인덱스 통계 정보 업데이트
ANALYZE statutes_articles;
ANALYZE precedent_contents;
```

---

## ⚠️ 주의사항

### 1. PGroonga 필수

- Docker 환경에서는 PGroonga가 자동으로 설치됩니다.
- 로컬 환경에서는 PGroonga 설치가 필요합니다.
- PGroonga가 없으면 `'korean'` 설정이 작동하지 않을 수 있습니다.

### 2. 설정 일관성

- 모든 곳에서 `'korean'` 설정을 사용해야 합니다:
  - 트리거 함수: `to_tsvector('korean', ...)`
  - 쿼리: `to_tsvector('korean', ...)`, `plainto_tsquery('korean', ...)`
  - 인덱스: `to_tsvector('korean', ...)` (Open Law 스키마)

### 3. text_search_vector 컬럼

- `text_search_vector` 컬럼이 있으면 항상 사용 (성능 최적화)
- 컬럼이 없으면 `to_tsvector('korean', ...)` 직접 사용
- 컬럼 존재 여부는 `_check_column_exists()` 메서드로 확인

---

## 🐛 문제 해결

### PGroonga가 감지되지 않는 경우

```python
# 로그 확인
# "⚠️ PGroonga is not available. Korean text search ('korean' config) requires PGroonga."

# 해결 방법
# 1. PostgreSQL에 PGroonga 확장 설치 확인
SELECT * FROM pg_extension WHERE extname = 'pgroonga';

# 2. 확장 설치
CREATE EXTENSION IF NOT EXISTS pgroonga;

# 3. 함수 존재 확인
SELECT proname FROM pg_proc WHERE proname LIKE '%pgroonga%';
```

### 인덱스가 사용되지 않는 경우

```sql
-- 인덱스 존재 확인
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'statutes_articles' 
AND indexname LIKE '%fts%';

-- 인덱스 재생성
DROP INDEX IF EXISTS idx_articles_fts;
CREATE INDEX idx_articles_fts 
ON statutes_articles USING gin(to_tsvector('korean', article_content));
```

### 검색 결과가 없는 경우

```sql
-- 쿼리 테스트
SELECT plainto_tsquery('korean', '계약 해지');
-- 결과: '계약' & '해지'

-- tsvector 생성 테스트
SELECT to_tsvector('korean', '계약을 해지할 수 있다');
-- 결과: 형태소 분석된 토큰들
```

---

## 📚 참고 자료

### 공식 문서

- [PGroonga 공식 문서](https://pgroonga.github.io/)
- [PGroonga 설치 가이드](https://pgroonga.github.io/install/)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [PostgreSQL tsvector 및 tsquery](https://www.postgresql.org/docs/current/datatype-textsearch.html)

### 프로젝트 내 문서

- [tsvector 사용 현황 검토 보고서](./tsvector_review_report.md)
- [Rank Score 계산 가이드](./rank_score_calculation_guide.md)
- [데이터베이스 스키마](./database_schema.md)

---

## 🔄 마이그레이션 가이드

### 기존 'simple' 설정에서 'korean' 설정으로 변경

1. **PGroonga 설치 확인**
   ```sql
   CREATE EXTENSION IF NOT EXISTS pgroonga;
   ```

2. **트리거 함수 업데이트**
   ```sql
   -- 기존: to_tsvector('simple', ...)
   -- 변경: to_tsvector('korean', ...)
   CREATE OR REPLACE FUNCTION update_statute_articles_fts()
   RETURNS TRIGGER AS $$
   BEGIN
       NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
       RETURN NEW;
   END;
   $$ LANGUAGE plpgsql;
   ```

3. **text_search_vector 컬럼 재생성**
   ```sql
   -- 기존 데이터 업데이트
   UPDATE statute_articles
   SET text_search_vector = to_tsvector('korean', COALESCE(text, ''));
   ```

4. **인덱스 재생성** (Open Law 스키마)
   ```sql
   DROP INDEX IF EXISTS idx_articles_fts;
   CREATE INDEX idx_articles_fts 
   ON statutes_articles USING gin(to_tsvector('korean', article_content));
   ```

---

## ✅ 체크리스트

### 개발 환경 설정

- [ ] Docker PostgreSQL 이미지에 PGroonga 포함 확인
- [ ] `init_postgres_extensions.sql`에 PGroonga 확장 추가 확인
- [ ] 트리거 함수에서 `'korean'` 설정 사용 확인

### 코드 검증

- [ ] `_check_pgroonga_available()` 메서드 정상 작동 확인
- [ ] `_convert_fts5_to_postgresql_fts()` 메서드에서 `'korean'` 설정 사용 확인
- [ ] 모든 검색 메서드에서 `rank_score_expr` 사용 확인

### 성능 최적화

- [ ] `text_search_vector` 컬럼 존재 여부 확인
- [ ] GIN 인덱스 생성 확인
- [ ] 실행 계획에서 인덱스 사용 확인

---

## 📝 변경 이력

- **2025-01-XX**: PGroonga 지원 추가, 'korean' 설정 사용으로 변경
- **2025-01-XX**: Docker PostgreSQL에 PGroonga 자동 설치 추가
- **2025-01-XX**: text_search_vector 컬럼 활용 개선

---

이 문서는 LawFirmAI 프로젝트에서 PGroonga와 tsvector를 사용한 한국어 전문 검색의 구현 및 사용 방법을 설명합니다.

