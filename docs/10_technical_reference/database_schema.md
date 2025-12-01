# PostgreSQL 데이터베이스 스키마

## 개요

LawFirmAI 프로젝트는 **PostgreSQL 12+** 데이터베이스를 사용하며, 다음 기능을 지원합니다:
- **pgvector**: 벡터 유사도 검색
- **Full-Text Search (FTS)**: 한국어 텍스트 검색
- **JSONB**: 유연한 메타데이터 저장

본 문서는 **실제 PostgreSQL 데이터베이스에 존재하는** 테이블 스키마를 상세히 설명합니다.

## 데이터베이스 요구사항

- **PostgreSQL 버전**: 12.0 이상 (권장: 18+)
- **필수 확장 프로그램**: 
  - `vector` (pgvector) - 벡터 검색
  - `pgroonga` - 한국어 전문 검색 (형태소 분석 지원) ⭐ **권장**
  - `pg_trgm` - 한국어 텍스트 검색 (trigram 기반, 보조)

## 확장 프로그램

### pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**용도**: 벡터 유사도 검색을 위한 확장 프로그램
- `VECTOR(768)` 타입 지원
- 코사인 유사도 연산 (`<=>`, `<->`)
- IVFFlat 및 HNSW 인덱스 지원

### PGroonga ⭐ (권장)

```sql
CREATE EXTENSION IF NOT EXISTS pgroonga;
```

**용도**: 한국어 전문 검색을 위한 확장 (형태소 분석 지원)
- `to_tsvector('korean', ...)` 사용 시 한국어 형태소 분석
- 조사, 어미 제거로 핵심 키워드 추출
- 검색 정확도 향상

**자세한 내용**: [PGroonga 및 tsvector 사용 가이드](./pgroonga_tsvector_guide.md)

### pg_trgm

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**용도**: 한국어 텍스트 검색을 위한 trigram 확장 (보조)
- `gin_trgm_ops` 연산자 클래스
- LIKE 검색 성능 향상
- 유사도 검색 지원

## 테이블 구조

### 1. 도메인 및 소스 관리

#### domains (도메인)

법률 도메인을 관리하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 도메인 ID (자동 증가) |
| name | VARCHAR(255) | NOT NULL, UNIQUE | 도메인 이름 (예: '민사법', '형사법') |

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `name`

**사용 예시:**
```sql
-- 도메인 조회
SELECT * FROM domains WHERE name = '민사법';
```

#### sources (소스 추적)

데이터 소스 정보를 추적하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 소스 ID |
| source_type | VARCHAR(50) | NOT NULL | 소스 타입 |
| path | TEXT | NOT NULL | 파일 경로 |
| hash | VARCHAR(64) | | 파일 해시 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

**인덱스:**
- PRIMARY KEY: `id`

### 2. 법률 데이터

#### statutes (법률 - Open Law API)

Open Law API에서 수집한 법률 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 법률 ID |
| law_id | INTEGER | NOT NULL, UNIQUE | 법령ID (Open Law API) |
| law_name_kr | TEXT | NOT NULL | 법령명(한글) |
| law_name_hanja | TEXT | | 법령명(한자) |
| law_name_en | TEXT | | 법령명(영어) |
| law_abbrv | TEXT | | 법령약칭 |
| law_type | TEXT | | 법령종류 |
| law_type_code | TEXT | | 법종구분코드 |
| proclamation_date | DATE | | 공포일자 |
| proclamation_number | INTEGER | | 공포번호 |
| effective_date | DATE | | 시행일자 |
| ministry_code | INTEGER | | 소관부처코드 |
| ministry_name | TEXT | | 소관부처명 |
| amendment_type | TEXT | | 제개정구분 |
| domain | TEXT | | 분야 (civil_law, criminal_law) |
| raw_response_json | JSONB | | 원본 JSON 응답 |
| collected_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수집 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |

**제약조건:**
- PRIMARY KEY: `id`
- UNIQUE: `law_id`

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `law_id`
- `idx_statutes_domain`: `domain` (B-Tree)
- `idx_statutes_law_id`: `law_id` (B-Tree)
- `idx_statutes_law_name`: `law_name_kr` (B-Tree)
- `idx_statutes_fts`: Full-Text Search (GIN 인덱스)
- `idx_statutes_trgm`: `law_name_kr` (GIN 인덱스, trigram)

#### statutes_articles (법률 조문 - Open Law API)

Open Law API에서 수집한 법률 조문을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 조문 ID |
| statute_id | INTEGER | NOT NULL | 법률 ID (FK → statutes(id)) |
| article_no | TEXT | NOT NULL | 조문번호 (예: "000200" = 제2조) |
| article_title | TEXT | | 조문제목 |
| article_content | TEXT | NOT NULL | 조문내용 |
| clause_no | TEXT | | 항번호 |
| clause_content | TEXT | | 항내용 |
| item_no | TEXT | | 호번호 |
| item_content | TEXT | | 호내용 |
| sub_item_no | TEXT | | 목번호 |
| sub_item_content | TEXT | | 목내용 |
| effective_date | DATE | | 조문시행일자 |
| raw_response_json | JSONB | | 원본 JSON 응답 |
| collected_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수집 일시 |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `statute_id` → `statutes(id)` ON DELETE CASCADE
- UNIQUE: `(statute_id, article_no, clause_no, item_no, sub_item_no)`

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `(statute_id, article_no, clause_no, item_no, sub_item_no)`
- `idx_articles_statute_id`: `statute_id` (B-Tree)
- `idx_articles_article_no`: `article_no` (B-Tree)
- `idx_articles_fts`: Full-Text Search (GIN 인덱스)
- `idx_articles_trgm`: `article_content` (GIN 인덱스, trigram)

#### statute_articles (법률 조문 - v2 스키마)

v2 스키마의 법률 조문을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 조문 ID |
| statute_id | INTEGER | NOT NULL | 법률 ID (FK → statutes(id)) |
| article_no | VARCHAR(50) | NOT NULL | 조 번호 (예: '750', '제750조') |
| clause_no | VARCHAR(50) | | 항 번호 |
| item_no | VARCHAR(50) | | 호 번호 |
| heading | TEXT | | 조문 제목 |
| text | TEXT | NOT NULL | 조문 내용 |
| version_effective_date | TEXT | | 버전 시행일 |
| text_search_vector | tsvector | | Full-Text Search용 벡터 (자동 생성) |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `statute_id` → `statutes(id)` ON DELETE CASCADE

**인덱스:**
- PRIMARY KEY: `id`
- `idx_statute_articles_keys`: `(statute_id, article_no, clause_no, item_no)` (복합 B-Tree)
- `idx_statute_articles_fts`: `text_search_vector` (GIN 인덱스)

**트리거:**
- `trigger_statute_articles_fts_update`: `text` 컬럼이 INSERT/UPDATE될 때 자동으로 `text_search_vector`를 갱신

**⚠️ 중요 사항:**
- `article_no`는 VARCHAR(50) 타입이므로 문자열로 비교해야 합니다: `article_no = '750'` (숫자 비교 금지)
- `text_search_vector`는 트리거에 의해 자동으로 생성되므로 수동 업데이트 불필요

### 3. 판례 데이터

#### precedents (판례)

판례 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 판례 ID |
| precedent_id | INTEGER | NOT NULL, UNIQUE | 판례정보일련번호 |
| case_name | TEXT | NOT NULL | 사건명 |
| case_number | TEXT | | 사건번호 |
| decision_date | DATE | | 선고일자 |
| court_name | TEXT | | 법원명 |
| court_type_code | INTEGER | | 법원종류코드 |
| case_type_name | TEXT | | 사건종류명 |
| case_type_code | INTEGER | | 사건종류코드 |
| decision_type | TEXT | | 판결유형 |
| decision_result | TEXT | | 선고 |
| domain | TEXT | | 분야 (civil_law, criminal_law) |
| raw_response_json | JSONB | | 원본 JSON 응답 |
| collected_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수집 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |

**제약조건:**
- PRIMARY KEY: `id`
- UNIQUE: `precedent_id`

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `precedent_id`
- `idx_precedents_domain`: `domain` (B-Tree)
- `idx_precedents_precedent_id`: `precedent_id` (B-Tree)
- `idx_precedents_case_name`: `case_name` (B-Tree)
- `idx_precedents_decision_date`: `decision_date` (B-Tree)
- `idx_precedents_fts`: Full-Text Search (GIN 인덱스)
- `idx_precedents_trgm`: `case_name` (GIN 인덱스, trigram)

**사용 예시:**
```sql
-- 특정 판례 조회
SELECT * FROM precedents WHERE precedent_id = 12345;

-- 도메인별 판례 통계
SELECT domain, COUNT(*) as count
FROM precedents
GROUP BY domain;
```

#### precedent_contents (판례 본문)

판례 본문을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 본문 ID |
| precedent_id | INTEGER | NOT NULL | 판례 ID (FK → precedents(id)) |
| section_type | TEXT | NOT NULL | 섹션 유형 (판시사항, 판결요지, 판례내용) |
| section_content | TEXT | NOT NULL | 섹션 내용 |
| referenced_articles | TEXT | | 참조조문 |
| referenced_precedents | TEXT | | 참조판례 |
| raw_response_json | JSONB | | 원본 JSON 응답 |
| collected_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수집 일시 |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `precedent_id` → `precedents(id)` ON DELETE CASCADE
- UNIQUE: `(precedent_id, section_type)` - 판례별 섹션 유형 유일성

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `(precedent_id, section_type)`
- `idx_precedent_contents_precedent_id`: `precedent_id` (B-Tree)
- `idx_precedent_contents_section_type`: `section_type` (B-Tree)
- `idx_precedent_contents_fts`: Full-Text Search (GIN 인덱스)
- `idx_precedent_contents_trgm`: `section_content` (GIN 인덱스, trigram)

**사용 예시:**
```sql
-- 특정 판례의 모든 섹션 조회
SELECT section_type, section_content
FROM precedent_contents
WHERE precedent_id = 12345
ORDER BY section_type;

-- Full-Text Search 사용
SELECT p.case_name, pc.section_type, pc.section_content
FROM precedent_contents pc
JOIN precedents p ON pc.precedent_id = p.id
WHERE pc.section_content LIKE '%불법행위%'
LIMIT 20;
```

#### precedent_chunks (판례 청크)

청킹된 판례 내용을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 청크 ID |
| precedent_content_id | INTEGER | NOT NULL | 판례 본문 ID (FK → precedent_contents(id)) |
| chunk_index | INTEGER | NOT NULL | 청크 인덱스 |
| chunk_content | TEXT | NOT NULL | 청크 내용 |
| chunk_length | INTEGER | | 청크 길이 |
| metadata | JSONB | | 메타데이터 |
| embedding_vector | VECTOR(768) | | 벡터 임베딩 (pgvector) |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| embedding_version | INTEGER | NOT NULL, DEFAULT 1 | 임베딩 버전 |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `precedent_content_id` → `precedent_contents(id)` ON DELETE CASCADE
- UNIQUE: `(precedent_content_id, chunk_index)`

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `(precedent_content_id, chunk_index)`
- `idx_precedent_chunks_content_id`: `precedent_content_id` (B-Tree)
- `idx_precedent_chunks_fts`: Full-Text Search (GIN 인덱스)
- `idx_precedent_chunks_vector_ivfflat`: `embedding_vector` (IVFFlat 인덱스)
- `idx_precedent_chunks_embedding_vector_hnsw`: `embedding_vector` (HNSW 인덱스)
- `idx_precedent_chunks_version`: `embedding_version` (B-Tree)

### 4. 벡터 검색 관련

**참고**: `text_chunks` 테이블은 더 이상 사용되지 않습니다. PostgreSQL 환경에서는 각 소스 타입별 전용 테이블(`precedent_chunks` 등)을 사용합니다.

#### embeddings (임베딩)

pgvector를 사용한 벡터 임베딩을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 임베딩 ID |
| chunk_id | INTEGER | NOT NULL | 청크 ID (다양한 소스 테이블 참조 가능) |
| model | VARCHAR(255) | NOT NULL | 모델명 |
| dim | INTEGER | NOT NULL | 벡터 차원 (예: 768) |
| version_id | INTEGER | | 임베딩 버전 ID (FK → embedding_versions(id)) |
| vector | VECTOR(768) | NOT NULL | 벡터 데이터 (pgvector 타입) |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `version_id` → `embedding_versions(id)` ON DELETE SET NULL
- `chunk_id`는 외래 키 제약조건 없음 (다양한 소스 테이블 참조 가능)

**인덱스:**
- PRIMARY KEY: `id`
- `idx_embeddings_chunk`: `chunk_id` (B-Tree)
- `idx_embeddings_vector`: `vector` (IVFFlat 인덱스, cosine 연산)
- `idx_embeddings_vector_hnsw`: `vector` (HNSW 인덱스, cosine 연산)

**⚠️ 중요 사항:**
- `vector` 컬럼은 pgvector 확장의 `VECTOR(768)` 타입입니다.
- IVFFlat과 HNSW 인덱스 모두 생성되어 있어 근사 검색 성능이 최적화되어 있습니다.

#### statute_embeddings (법률 조문 임베딩)

법률 조문의 벡터 임베딩을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 임베딩 ID |
| article_id | INTEGER | NOT NULL | 조문 ID (FK → statutes_articles(id)) |
| embedding_vector | VECTOR(768) | | 벡터 데이터 (pgvector 타입) |
| metadata | JSONB | | 메타데이터 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| embedding_version | INTEGER | NOT NULL, DEFAULT 1 | 임베딩 버전 |

**제약조건:**
- PRIMARY KEY: `id`
- FOREIGN KEY: `article_id` → `statutes_articles(id)` ON DELETE CASCADE
- UNIQUE: `(article_id, embedding_version)`

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `(article_id, embedding_version)`
- `idx_statute_embeddings_article_id`: `article_id` (B-Tree)
- `idx_statute_embeddings_version`: `embedding_version` (B-Tree)
- `idx_statute_embeddings_vector`: `embedding_vector` (IVFFlat 인덱스)
- `idx_statute_embeddings_embedding_vector_hnsw`: `embedding_vector` (HNSW 인덱스)

#### embedding_versions (임베딩 버전)

임베딩 버전 정보를 관리하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 버전 ID |
| version | INTEGER | NOT NULL | 버전 번호 |
| model_name | VARCHAR(255) | NOT NULL | 모델명 |
| dim | INTEGER | NOT NULL | 벡터 차원 |
| data_type | VARCHAR(20) | NOT NULL | 데이터 타입 (statutes, precedents 등) |
| chunking_strategy | VARCHAR(50) | | 청킹 전략 |
| description | TEXT | | 설명 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| is_active | BOOLEAN | DEFAULT FALSE | 활성 여부 |
| metadata | JSONB | | 메타데이터 |

**제약조건:**
- PRIMARY KEY: `id`
- UNIQUE: `(version, data_type)` - 버전과 데이터 타입 조합 유일성

**인덱스:**
- PRIMARY KEY: `id`
- UNIQUE: `(version, data_type)`
- `idx_embedding_versions_active`: `id` (부분 인덱스, `is_active = TRUE`인 경우만)

**사용 예시:**
```sql
-- 활성 버전 조회
SELECT * FROM embedding_versions WHERE is_active = TRUE;

-- 데이터 타입별 활성 버전 조회
SELECT * FROM embedding_versions 
WHERE is_active = TRUE AND data_type = 'statutes';
```

#### retrieval_cache (검색 캐시)

### 5. API 서버용 테이블 (인증 및 세션 관리)

#### users (회원 정보)

회원 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| user_id | VARCHAR(255) | PRIMARY KEY | 사용자 ID (Google OAuth2 sub) |
| email | VARCHAR(255) | | 이메일 주소 |
| name | TEXT | | 사용자 이름 |
| picture | TEXT | | 프로필 사진 URL |
| provider | VARCHAR(50) | | 인증 제공자 (google 등) |
| google_access_token | TEXT | | Google OAuth2 Access Token |
| google_refresh_token | TEXT | | Google OAuth2 Refresh Token |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |

**제약조건:**
- PRIMARY KEY: `user_id`

**인덱스:**
- PRIMARY KEY: `user_id`
- `idx_users_email`: `email` (B-Tree)
- `idx_users_provider`: `provider` (B-Tree)

**사용 예시:**
```sql
-- 사용자 조회
SELECT * FROM users WHERE email = 'user@example.com';

-- Google OAuth2 사용자 조회
SELECT * FROM users WHERE provider = 'google' AND user_id = 'google_user_id';
```

#### sessions (세션 정보)

채팅 세션 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| session_id | VARCHAR(255) | PRIMARY KEY | 세션 ID (UUID) |
| title | TEXT | | 세션 제목 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| message_count | INTEGER | DEFAULT 0 | 메시지 개수 |
| user_id | VARCHAR(255) | | 사용자 ID (FK → users(user_id)) |
| ip_address | VARCHAR(45) | | IP 주소 (IPv4/IPv6) |

**제약조건:**
- PRIMARY KEY: `session_id`

**인덱스:**
- PRIMARY KEY: `session_id`
- `idx_sessions_user_id`: `user_id` (B-Tree)
- `idx_sessions_updated_at`: `updated_at` (B-Tree)

**사용 예시:**
```sql
-- 사용자별 세션 조회
SELECT * FROM sessions WHERE user_id = 'user_id' ORDER BY updated_at DESC;

-- 최근 세션 조회
SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 10;
```

#### messages (메시지 정보)

채팅 메시지를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| message_id | VARCHAR(255) | PRIMARY KEY | 메시지 ID (UUID) |
| session_id | VARCHAR(255) | NOT NULL, FK | 세션 ID (FK → sessions(session_id)) |
| role | VARCHAR(50) | NOT NULL | 메시지 역할 (user, assistant, progress) |
| content | TEXT | NOT NULL | 메시지 내용 |
| timestamp | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 메시지 시간 |
| metadata | JSONB | | 메타데이터 (sources_by_type, sources_detail 등) |

**제약조건:**
- PRIMARY KEY: `message_id`
- FOREIGN KEY: `session_id` → `sessions(session_id)` ON DELETE CASCADE

**인덱스:**
- PRIMARY KEY: `message_id`
- `idx_messages_session_id`: `session_id` (B-Tree)
- `idx_messages_timestamp`: `timestamp` (B-Tree)
- `idx_messages_metadata_gin`: `metadata` (GIN 인덱스, JSONB 쿼리 최적화)

**사용 예시:**
```sql
-- 세션별 메시지 조회
SELECT * FROM messages WHERE session_id = 'session_id' ORDER BY timestamp ASC;

-- 메타데이터에서 sources_by_type 조회
SELECT message_id, metadata->'sources_by_type' as sources
FROM messages
WHERE metadata ? 'sources_by_type';
```

검색 결과 캐시를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| query_hash | VARCHAR(255) | PRIMARY KEY | 쿼리 해시 |
| topk_ids | TEXT | | 상위 K개 ID 목록 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

**제약조건:**
- PRIMARY KEY: `query_hash`

## Full-Text Search (FTS)

PostgreSQL의 Full-Text Search 기능을 사용하여 한국어 텍스트 검색을 지원합니다.

### 트리거 함수

다음 테이블의 텍스트 컬럼은 자동으로 `tsvector`로 변환됩니다:
- `statute_articles.text` → `text_search_vector`

각 테이블에는 **BEFORE INSERT/UPDATE 트리거**가 설정되어 있어, 텍스트가 변경될 때 자동으로 `text_search_vector`가 갱신됩니다.

### FTS 쿼리 예시 (PGroonga 사용)

```sql
-- 법률 조문 검색 (PGroonga 사용, 'korean' 설정)
SELECT 
    s.law_name_kr,
    sa.article_no,
    sa.article_content,
    ts_rank_cd(
        to_tsvector('korean', sa.article_content),
        plainto_tsquery('korean', '계약 해지')
    ) as rank_score
FROM statutes_articles sa
JOIN statutes s ON sa.statute_id = s.id
WHERE to_tsvector('korean', sa.article_content) 
      @@ plainto_tsquery('korean', '계약 해지')
ORDER BY rank_score DESC
LIMIT 10;

-- text_search_vector 컬럼 활용 (권장, 인덱스 직접 사용)
SELECT 
    s.law_name_kr,
    sa.article_no,
    sa.article_content,
    ts_rank_cd(
        sa.text_search_vector,
        plainto_tsquery('korean', '계약 해지')
    ) as rank_score
FROM statute_articles sa
JOIN statutes s ON sa.statute_id = s.id
WHERE sa.text_search_vector 
      @@ plainto_tsquery('korean', '계약 해지')
ORDER BY rank_score DESC
LIMIT 10;

-- 판례 본문 검색
SELECT 
    p.case_name,
    pc.section_type,
    pc.section_content
FROM precedent_contents pc
JOIN precedents p ON pc.precedent_id = p.id
WHERE pc.section_content LIKE '%불법행위%'
LIMIT 20;
```

## 벡터 검색 (pgvector)

pgvector를 사용하여 벡터 유사도 검색을 수행합니다.

### 벡터 인덱스

다음 테이블의 `vector` 컬럼에는 벡터 인덱스가 생성되어 있습니다:
- `embeddings.vector`: IVFFlat 및 HNSW 인덱스
- `precedent_chunks.embedding_vector`: IVFFlat 및 HNSW 인덱스
- `statute_embeddings.embedding_vector`: IVFFlat 및 HNSW 인덱스

### 벡터 검색 예시

```sql
-- 코사인 유사도 검색 (거리 기반)
-- embeddings 테이블 벡터 검색 (chunk_id는 다른 소스 테이블 참조)
-- 참고: text_chunks 테이블은 더 이상 사용되지 않으므로, 
-- 실제 사용 시에는 chunk_id가 참조하는 테이블과 JOIN 필요
SELECT 
    e.id,
    e.chunk_id,
    e.vector <=> query_vector as distance,
    1 - (e.vector <=> query_vector) as similarity
FROM embeddings e
WHERE e.version_id = 1
ORDER BY e.vector <=> query_vector
LIMIT 10;

-- 판례 청크 벡터 검색
SELECT 
    pc.id,
    pc.chunk_content,
    1 - (pc.embedding_vector <=> query_vector) as similarity
FROM precedent_chunks pc
WHERE pc.embedding_version = 1
ORDER BY pc.embedding_vector <=> query_vector
LIMIT 10;
```

## 외래 키 관계

### 주요 관계도

```
domains
  └── (간접 참조)

statutes (Open Law API)
  └── statutes_articles
        └── statute_embeddings

statutes (v2 스키마)
  └── statute_articles

precedents
  └── precedent_contents
        └── precedent_chunks

embedding_versions
  └── embeddings
```

### 외래 키 정책

- **ON DELETE CASCADE**: 
  - `statutes` → `statutes_articles`
  - `statutes_articles` → `statute_embeddings`
  - `precedents` → `precedent_contents`
  - `precedent_contents` → `precedent_chunks`
  - 부모 레코드 삭제 시 자식 레코드도 자동 삭제

- **ON DELETE SET NULL**:
  - `embedding_versions` → `embeddings.version_id`
  - 부모 레코드 삭제 시 외래 키를 NULL로 설정

## 제약조건

### UNIQUE 제약조건

- `domains.name`
- `statutes.law_id` (Open Law API)
- `statutes_articles(statute_id, article_no, clause_no, item_no, sub_item_no)` (Open Law API)
- `precedents.precedent_id`
- `precedent_contents(precedent_id, section_type)`
- `precedent_chunks(precedent_content_id, chunk_index)`
- `embedding_versions(version, data_type)`
- `statute_embeddings(article_id, embedding_version)`

## 인덱스 전략

### B-Tree 인덱스

- **외래 키 컬럼**: 모든 외래 키에 인덱스 생성
- **자주 조회되는 컬럼**: `law_id`, `precedent_id`, `domain` 등
- **정렬에 사용되는 컬럼**: `decision_date`, `created_at` 등
- **UNIQUE 제약조건**: 자동으로 UNIQUE 인덱스 생성

### GIN 인덱스

- **Full-Text Search용 tsvector 컬럼**: 
  - `statute_articles.text_search_vector`
- **JSONB 메타데이터 컬럼**: 
  - `embedding_versions.metadata`
  - `precedent_chunks.metadata`
- **Trigram 인덱스 (한국어 검색)**:
  - `statutes.law_name_kr`
  - `statutes_articles.article_content`
  - `precedents.case_name`
  - `precedent_contents.section_content`

### IVFFlat 인덱스

- **벡터 검색용 vector 컬럼**: 
  - `embeddings.vector`
  - `precedent_chunks.embedding_vector`
  - `statute_embeddings.embedding_vector`
  - 연산자: `vector_cosine_ops` (코사인 유사도)

### HNSW 인덱스

- **벡터 검색용 vector 컬럼** (더 빠른 검색):
  - `embeddings.vector`
  - `precedent_chunks.embedding_vector`
  - `statute_embeddings.embedding_vector`
  - 연산자: `vector_cosine_ops` (코사인 유사도)

### 부분 인덱스

- `embedding_versions`: `is_active = TRUE`인 경우만 인덱싱
  ```sql
  CREATE INDEX idx_embedding_versions_active 
  ON embedding_versions (id) WHERE is_active = TRUE;
  ```

## TEXT2SQL 사용 가이드

### 데이터 타입 주의사항

1. **날짜 필드**:
   - `statutes.proclamation_date`, `statutes.effective_date`: DATE 타입
   - `statutes_articles.effective_date`: DATE 타입
   - `precedents.decision_date`: DATE 타입
   - 날짜 비교 시 DATE 타입으로 비교:
     ```sql
     WHERE decision_date >= '2022-01-01'::DATE
     WHERE decision_date >= DATE '2022-01-01'
     ```

2. **article_no (TEXT/VARCHAR 타입)**:
   - `statutes_articles.article_no`: TEXT 타입 (Open Law API)
   - `statute_articles.article_no`: VARCHAR(50) 타입 (v2 스키마)
   - 문자열로 비교해야 합니다:
     ```sql
     WHERE article_no = '750'
     WHERE article_no LIKE '750%'
     ```

### TEXT2SQL 쿼리 예시

```sql
-- 법령 조문 검색 (Open Law API 스키마)
SELECT s.law_name_kr, sa.article_no, sa.article_content
FROM statutes_articles sa
JOIN statutes s ON sa.statute_id = s.id
WHERE s.law_name_kr LIKE '%민법%' AND sa.article_no LIKE '%000750%'
LIMIT 5;

-- 판례 검색
SELECT p.case_name, p.case_number, p.decision_date, pc.section_content
FROM precedent_contents pc
JOIN precedents p ON pc.precedent_id = p.id
WHERE pc.section_content LIKE '%불법행위%'
LIMIT 20;

-- 날짜 범위 검색
SELECT p.case_name, p.decision_date, p.court_name
FROM precedents p
WHERE p.decision_date >= '2022-01-01'::DATE
ORDER BY p.decision_date DESC
LIMIT 10;
```

## 참고 사항

### 타임스탬프

- 모든 TIMESTAMP 컬럼은 UTC 기준입니다.
- 애플리케이션에서 시간대 변환이 필요할 수 있습니다.

### JSONB 활용

- JSONB 타입을 사용하여 유연한 메타데이터 저장이 가능합니다.
- GIN 인덱스를 사용하여 JSONB 쿼리 성능을 최적화합니다.
- 예: `WHERE raw_response_json @> '{"law_id": 123}'::jsonb`

### 데이터 무결성

- 외래 키 제약조건으로 데이터 무결성을 보장합니다.
- CASCADE/SET NULL 정책으로 데이터 일관성을 유지합니다.
- UNIQUE 제약조건으로 중복 데이터를 방지합니다.

### 트랜잭션

- PostgreSQL의 ACID 특성을 활용하여 데이터 일관성을 보장합니다.
- 트랜잭션 내에서 여러 테이블에 대한 작업을 원자적으로 수행할 수 있습니다.

## 마이그레이션 및 스키마 변경

스키마 변경 시 다음을 고려하세요:

1. **외래 키 제약조건**: 기존 데이터와의 호환성 확인
2. **인덱스 재생성**: 대용량 테이블의 경우 시간이 소요될 수 있음
3. **트리거 함수**: 스키마 변경 시 트리거 함수도 함께 업데이트 필요
4. **벡터 인덱스**: IVFFlat/HNSW 인덱스 재생성 시 데이터 재분포 필요

## 관련 문서

### 프로젝트 내 문서

- **[PGroonga 및 tsvector 사용 가이드](./pgroonga_tsvector_guide.md)**: 상세한 사용 방법 및 예시
- [tsvector 사용 현황 검토 보고서](./tsvector_review_report.md)
- [Rank Score 계산 가이드](./rank_score_calculation_guide.md)

### 공식 문서

- PostgreSQL 공식 문서: https://www.postgresql.org/docs/
- pgvector 문서: https://github.com/pgvector/pgvector
- PGroonga 공식 문서: https://pgroonga.github.io/
- PostgreSQL Full-Text Search: https://www.postgresql.org/docs/current/textsearch.html
- pg_trgm 문서: https://www.postgresql.org/docs/current/pgtrgm.html
