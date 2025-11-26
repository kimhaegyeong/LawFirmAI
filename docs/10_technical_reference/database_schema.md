# PostgreSQL 데이터베이스 스키마

## 개요

LawFirmAI 프로젝트는 PostgreSQL 데이터베이스를 사용하며, pgvector 확장을 통해 벡터 검색을 지원합니다. 본 문서는 현재 사용 중인 PostgreSQL 데이터베이스 스키마를 설명합니다.

## 확장 프로그램

### pgvector
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

벡터 유사도 검색을 위한 pgvector 확장을 사용합니다.

## 테이블 구조

### 1. 도메인 및 소스 관리

#### domains (도메인)
법률 도메인을 관리하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 도메인 ID |
| name | VARCHAR(255) | NOT NULL, UNIQUE | 도메인 이름 |

#### sources (소스 추적)
데이터 소스 정보를 추적하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 소스 ID |
| source_type | VARCHAR(50) | NOT NULL | 소스 타입 (statute, case, decision, interpretation) |
| path | TEXT | NOT NULL | 파일 경로 |
| hash | VARCHAR(64) | | 파일 해시 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

**인덱스:**
- `idx_sources_type`: source_type
- `idx_sources_path`: path

### 2. 법률 데이터

#### statutes (법률)
법률 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 법률 ID |
| domain_id | INTEGER | NOT NULL, FK → domains(id) | 도메인 ID |
| name | VARCHAR(255) | NOT NULL | 법률명 |
| abbrv | VARCHAR(100) | | 약칭 |
| statute_type | VARCHAR(50) | | 법률 유형 |
| proclamation_date | TEXT | | 공포일 |
| effective_date | TEXT | | 시행일 |
| category | VARCHAR(50) | | 카테고리 |
| | | UNIQUE(domain_id, name) | 도메인별 법률명 유일성 |

**인덱스:**
- `idx_statutes_domain`: domain_id
- `idx_statutes_category`: category

#### statute_articles (법률 조문)
법률 조문을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 조문 ID |
| statute_id | INTEGER | NOT NULL, FK → statutes(id) | 법률 ID |
| article_no | VARCHAR(50) | NOT NULL | 조 번호 (제n조) |
| clause_no | VARCHAR(50) | | 항 번호 |
| item_no | VARCHAR(50) | | 호 번호 |
| heading | TEXT | | 조문 제목 |
| text | TEXT | NOT NULL | 조문 내용 |
| version_effective_date | TEXT | | 버전 시행일 |
| text_search_vector | tsvector | | Full-Text Search용 벡터 |

**인덱스:**
- `idx_statute_articles_keys`: (statute_id, article_no, clause_no, item_no)
- `idx_statute_articles_fts`: text_search_vector (GIN 인덱스)

**트리거:**
- `trigger_statute_articles_fts_update`: text 업데이트 시 text_search_vector 자동 갱신

### 3. 판례 데이터

#### cases (판례)
판례 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 판례 ID |
| domain_id | INTEGER | NOT NULL, FK → domains(id) | 도메인 ID |
| doc_id | VARCHAR(255) | NOT NULL, UNIQUE | 문서 ID |
| court | VARCHAR(100) | | 법원 |
| case_type | VARCHAR(50) | | 사건 유형 |
| casenames | TEXT | | 사건명 |
| announce_date | TEXT | | 선고일 |

**인덱스:**
- `idx_cases_domain`: domain_id
- `idx_cases_doc_id`: doc_id

#### case_paragraphs (판례 문단)
판례 문단을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 문단 ID |
| case_id | INTEGER | NOT NULL, FK → cases(id) | 판례 ID |
| para_index | INTEGER | NOT NULL | 문단 인덱스 |
| text | TEXT | NOT NULL | 문단 내용 |
| text_search_vector | tsvector | | Full-Text Search용 벡터 |
| | | UNIQUE(case_id, para_index) | 판례별 문단 인덱스 유일성 |

**인덱스:**
- `idx_case_paragraphs_case`: (case_id, para_index)
- `idx_case_paragraphs_fts`: text_search_vector (GIN 인덱스)

**트리거:**
- `trigger_case_paragraphs_fts_update`: text 업데이트 시 text_search_vector 자동 갱신

### 4. 심결례 데이터

#### decisions (심결례)
심결례 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 심결례 ID |
| domain_id | INTEGER | NOT NULL, FK → domains(id) | 도메인 ID |
| org | VARCHAR(255) | | 기관 |
| doc_id | VARCHAR(255) | NOT NULL, UNIQUE | 문서 ID |
| decision_date | TEXT | | 결정일 |
| result | TEXT | | 결과 |

**인덱스:**
- `idx_decisions_domain`: domain_id
- `idx_decisions_doc_id`: doc_id

#### decision_paragraphs (심결례 문단)
심결례 문단을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 문단 ID |
| decision_id | INTEGER | NOT NULL, FK → decisions(id) | 심결례 ID |
| para_index | INTEGER | NOT NULL | 문단 인덱스 |
| text | TEXT | NOT NULL | 문단 내용 |
| text_search_vector | tsvector | | Full-Text Search용 벡터 |
| | | UNIQUE(decision_id, para_index) | 심결례별 문단 인덱스 유일성 |

**인덱스:**
- `idx_decision_paragraphs_decision`: (decision_id, para_index)
- `idx_decision_paragraphs_fts`: text_search_vector (GIN 인덱스)

**트리거:**
- `trigger_decision_paragraphs_fts_update`: text 업데이트 시 text_search_vector 자동 갱신

### 5. 유권해석 데이터

#### interpretations (유권해석)
유권해석 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 해석 ID |
| domain_id | INTEGER | NOT NULL, FK → domains(id) | 도메인 ID |
| org | VARCHAR(255) | | 기관 |
| doc_id | VARCHAR(255) | NOT NULL, UNIQUE | 문서 ID |
| title | TEXT | | 제목 |
| response_date | TEXT | | 회신일 |

**인덱스:**
- `idx_interpretations_domain`: domain_id
- `idx_interpretations_doc_id`: doc_id

#### interpretation_paragraphs (해석례 문단)
해석례 문단을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 문단 ID |
| interpretation_id | INTEGER | NOT NULL, FK → interpretations(id) | 해석 ID |
| para_index | INTEGER | NOT NULL | 문단 인덱스 |
| text | TEXT | NOT NULL | 문단 내용 |
| text_search_vector | tsvector | | Full-Text Search용 벡터 |
| | | UNIQUE(interpretation_id, para_index) | 해석별 문단 인덱스 유일성 |

**인덱스:**
- `idx_interpretation_paragraphs_interp`: (interpretation_id, para_index)
- `idx_interpretation_paragraphs_fts`: text_search_vector (GIN 인덱스)

**트리거:**
- `trigger_interpretation_paragraphs_fts_update`: text 업데이트 시 text_search_vector 자동 갱신

### 6. 벡터 검색 관련

#### text_chunks (텍스트 청크)
RAG를 위한 텍스트 청크를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 청크 ID |
| source_type | VARCHAR(50) | NOT NULL | 소스 타입 (statute_article, case_paragraph, decision_paragraph, interpretation_paragraph) |
| source_id | INTEGER | NOT NULL | 소스 ID (해당 테이블의 FK) |
| level | VARCHAR(50) | | 레벨 (article/clause/item or paragraph) |
| chunk_index | INTEGER | NOT NULL | 청크 인덱스 |
| start_char | INTEGER | | 시작 문자 위치 |
| end_char | INTEGER | | 종료 문자 위치 |
| overlap_chars | INTEGER | | 겹치는 문자 수 |
| text | TEXT | NOT NULL | 청크 텍스트 |
| token_count | INTEGER | | 토큰 수 |
| embedding_version_id | INTEGER | FK → embedding_versions(id) | 임베딩 버전 ID |
| chunk_size_category | VARCHAR(20) | | 청크 크기 카테고리 |
| chunk_group_id | VARCHAR(255) | | 청크 그룹 ID |
| chunking_strategy | VARCHAR(50) | | 청킹 전략 |
| meta | JSONB | | 메타데이터 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| | | UNIQUE(source_type, source_id, chunk_index, embedding_version_id) | 소스별 청크 유일성 |

**제약조건:**
- `chk_text_chunks_source_type`: source_type은 ('statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph') 중 하나

**인덱스:**
- `idx_text_chunks_source`: (source_type, source_id, chunk_index, embedding_version_id)
- `idx_text_chunks_meta_gin`: meta (GIN 인덱스)
- `idx_text_chunks_version_type`: (embedding_version_id, source_type)

#### embeddings (임베딩)
pgvector를 사용한 벡터 임베딩을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 임베딩 ID |
| chunk_id | INTEGER | NOT NULL, FK → text_chunks(id) | 청크 ID |
| model | VARCHAR(255) | NOT NULL | 모델명 |
| dim | INTEGER | NOT NULL | 벡터 차원 |
| version_id | INTEGER | FK → embedding_versions(id) | 임베딩 버전 ID |
| vector | VECTOR(768) | NOT NULL | 벡터 데이터 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

**인덱스:**
- `idx_embeddings_chunk`: chunk_id
- `idx_embeddings_vector`: vector (IVFFlat 인덱스, cosine 연산)

#### embedding_versions (임베딩 버전)
임베딩 버전 정보를 관리하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 버전 ID |
| version_name | VARCHAR(255) | NOT NULL, UNIQUE | 버전 이름 |
| chunking_strategy | VARCHAR(50) | NOT NULL | 청킹 전략 |
| model_name | VARCHAR(255) | NOT NULL | 모델명 |
| description | TEXT | | 설명 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| is_active | BOOLEAN | DEFAULT FALSE | 활성 여부 |
| metadata | JSONB | | 메타데이터 |

**인덱스:**
- `idx_embedding_versions_active`: id (부분 인덱스, is_active = TRUE인 경우만)

#### retrieval_cache (검색 캐시)
검색 결과 캐시를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| query_hash | VARCHAR(255) | PRIMARY KEY | 쿼리 해시 |
| topk_ids | TEXT | | 상위 K개 ID 목록 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

### 7. 파일 처리 이력

#### processed_files (처리된 파일)
데이터 처리 이력을 추적하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 레코드 ID |
| file_path | TEXT | NOT NULL, UNIQUE | 파일 경로 |
| file_hash | VARCHAR(64) | NOT NULL | 파일 해시 |
| data_type | VARCHAR(50) | NOT NULL | 데이터 타입 |
| processed_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 처리 일시 |
| processing_status | VARCHAR(50) | DEFAULT 'completed' | 처리 상태 |
| record_count | INTEGER | DEFAULT 0 | 레코드 수 |
| processing_version | VARCHAR(50) | DEFAULT '1.0' | 처리 버전 |
| error_message | TEXT | | 에러 메시지 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |

**인덱스:**
- `idx_processed_files_path`: file_path
- `idx_processed_files_type`: data_type
- `idx_processed_files_status`: processing_status

### 8. 채팅/세션 관리

#### sessions (세션)
사용자 세션 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| session_id | VARCHAR(255) | PRIMARY KEY | 세션 ID |
| title | TEXT | | 세션 제목 |
| category | TEXT | | 카테고리 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| message_count | INTEGER | DEFAULT 0 | 메시지 수 |
| user_id | VARCHAR(255) | | 사용자 ID |
| ip_address | VARCHAR(45) | | IP 주소 |
| metadata | JSONB | | 메타데이터 |

**인덱스:**
- `idx_sessions_updated_at`: updated_at
- `idx_sessions_user_id`: user_id

#### messages (메시지)
채팅 메시지를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| message_id | VARCHAR(255) | PRIMARY KEY | 메시지 ID |
| session_id | VARCHAR(255) | NOT NULL, FK → sessions(session_id) | 세션 ID |
| role | VARCHAR(50) | NOT NULL | 역할 (user, assistant 등) |
| content | TEXT | NOT NULL | 메시지 내용 |
| timestamp | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 타임스탬프 |
| metadata | JSONB | | 메타데이터 |

**인덱스:**
- `idx_messages_session_id`: session_id

### 9. 키워드 및 패턴 관리

#### keywords (키워드)
질문 분류를 위한 키워드를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 키워드 ID |
| question_type | TEXT | NOT NULL | 질문 유형 |
| keyword | TEXT | NOT NULL | 키워드 |
| weight_level | TEXT | NOT NULL | 가중치 레벨 (high, medium, low) |
| weight_value | REAL | NOT NULL, DEFAULT 1.0 | 가중치 값 |
| category | TEXT | | 카테고리 |
| description | TEXT | | 설명 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| is_active | BOOLEAN | DEFAULT TRUE | 활성 여부 |
| | | UNIQUE(question_type, keyword) | 질문 유형별 키워드 유일성 |

#### patterns (패턴)
질문 분류를 위한 패턴을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 패턴 ID |
| question_type | TEXT | NOT NULL | 질문 유형 |
| pattern | TEXT | NOT NULL | 패턴 |
| pattern_type | TEXT | NOT NULL | 패턴 타입 (regex, keyword, phrase) |
| priority | INTEGER | DEFAULT 1 | 우선순위 |
| description | TEXT | | 설명 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| is_active | BOOLEAN | DEFAULT TRUE | 활성 여부 |

#### question_types (질문 유형)
질문 유형 메타데이터를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 유형 ID |
| type_name | TEXT | UNIQUE, NOT NULL | 유형 이름 |
| display_name | TEXT | NOT NULL | 표시 이름 |
| description | TEXT | | 설명 |
| parent_type | TEXT | | 부모 유형 |
| priority | INTEGER | DEFAULT 1 | 우선순위 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |
| is_active | BOOLEAN | DEFAULT TRUE | 활성 여부 |

#### keyword_stats (키워드 통계)
키워드 사용 통계를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 통계 ID |
| keyword_id | INTEGER | NOT NULL, FK → keywords(id) | 키워드 ID |
| match_count | INTEGER | DEFAULT 0 | 매칭 횟수 |
| success_count | INTEGER | DEFAULT 0 | 성공 횟수 |
| last_matched_at | TIMESTAMP | | 마지막 매칭 일시 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 수정 일시 |

### 10. 동의어 관리

#### synonyms (동의어)
동의어 매핑을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 동의어 ID |
| keyword | TEXT | NOT NULL | 키워드 |
| synonym | TEXT | NOT NULL | 동의어 |
| domain | TEXT | DEFAULT 'general' | 도메인 |
| context | TEXT | DEFAULT 'general' | 컨텍스트 |
| confidence | REAL | DEFAULT 0.0 | 신뢰도 |
| usage_count | INTEGER | DEFAULT 0 | 사용 횟수 |
| user_rating | REAL | DEFAULT 0.0 | 사용자 평가 |
| source | TEXT | DEFAULT 'unknown' | 소스 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |
| last_used | TIMESTAMP | | 마지막 사용 일시 |
| is_active | BOOLEAN | DEFAULT TRUE | 활성 여부 |
| | | UNIQUE(keyword, synonym, domain, context) | 동의어 유일성 |

**인덱스:**
- `idx_synonyms_keyword`: keyword
- `idx_synonyms_domain`: domain
- `idx_synonyms_usage`: usage_count (DESC)
- `idx_synonyms_active`: is_active
- `idx_synonyms_confidence`: confidence (DESC)

#### synonym_usage_stats (동의어 사용 통계)
동의어 사용 통계를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 통계 ID |
| synonym_id | INTEGER | FK → synonyms(id) | 동의어 ID |
| usage_date | DATE | | 사용 일자 |
| usage_count | INTEGER | DEFAULT 0 | 사용 횟수 |
| success_rate | REAL | DEFAULT 0.0 | 성공률 |
| | | UNIQUE(synonym_id, usage_date) | 동의어별 일자 유일성 |

#### synonym_quality_metrics (동의어 품질 평가)
동의어 품질 평가 메트릭을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | SERIAL | PRIMARY KEY | 평가 ID |
| synonym_id | INTEGER | FK → synonyms(id) | 동의어 ID |
| semantic_similarity | REAL | DEFAULT 0.0 | 의미적 유사도 |
| context_relevance | REAL | DEFAULT 0.0 | 컨텍스트 관련성 |
| domain_relevance | REAL | DEFAULT 0.0 | 도메인 관련성 |
| user_feedback_score | REAL | DEFAULT 0.0 | 사용자 피드백 점수 |
| overall_score | REAL | DEFAULT 0.0 | 전체 점수 |
| evaluated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 평가 일시 |

### 11. 피드백 관리

#### feedback (피드백)
사용자 피드백을 저장하는 테이블입니다.

| 컬럼명 | 타입 | 제약조건 | 설명 |
|--------|------|----------|------|
| id | TEXT | PRIMARY KEY | 피드백 ID |
| timestamp | TEXT | NOT NULL | 타임스탬프 |
| session_id | TEXT | | 세션 ID |
| user_id | TEXT | | 사용자 ID |
| feedback_type | TEXT | NOT NULL | 피드백 타입 |
| rating | INTEGER | | 평점 |
| text_content | TEXT | | 텍스트 내용 |
| question | TEXT | | 질문 |
| answer | TEXT | | 답변 |
| context | TEXT | | 컨텍스트 |
| metadata | TEXT | | 메타데이터 |

**인덱스:**
- `idx_feedback_timestamp`: timestamp
- `idx_feedback_type`: feedback_type
- `idx_feedback_rating`: rating
- `idx_feedback_session`: session_id

## Full-Text Search (FTS)

PostgreSQL의 Full-Text Search 기능을 사용하여 한국어 텍스트 검색을 지원합니다.

### 트리거 함수

다음 테이블의 텍스트 컬럼은 자동으로 `tsvector`로 변환됩니다:
- `statute_articles.text` → `text_search_vector`
- `case_paragraphs.text` → `text_search_vector`
- `decision_paragraphs.text` → `text_search_vector`
- `interpretation_paragraphs.text` → `text_search_vector`

각 테이블에는 BEFORE INSERT/UPDATE 트리거가 설정되어 있어, 텍스트가 변경될 때 자동으로 `text_search_vector`가 갱신됩니다.

### FTS 쿼리 예시

```sql
-- 법률 조문 검색
SELECT id, text, heading
FROM statute_articles
WHERE text_search_vector @@ to_tsquery('korean', '계약 & 해지')
ORDER BY ts_rank(text_search_vector, to_tsquery('korean', '계약 & 해지')) DESC;
```

## 벡터 검색 (pgvector)

pgvector를 사용하여 벡터 유사도 검색을 수행합니다.

### 벡터 인덱스

`embeddings` 테이블의 `vector` 컬럼에는 IVFFlat 인덱스가 생성되어 있습니다:
- 인덱스 타입: IVFFlat
- 연산자: cosine
- lists: 100

### 벡터 검색 예시

```sql
-- 코사인 유사도 검색
SELECT 
    e.id,
    tc.text,
    1 - (e.vector <=> query_vector) as similarity
FROM embeddings e
JOIN text_chunks tc ON e.chunk_id = tc.id
WHERE e.version_id = 1
ORDER BY e.vector <=> query_vector
LIMIT 10;
```

## 외래 키 관계

### 주요 관계도

```
domains
  ├── statutes
  │     └── statute_articles
  ├── cases
  │     └── case_paragraphs
  ├── decisions
  │     └── decision_paragraphs
  └── interpretations
        └── interpretation_paragraphs

embedding_versions
  ├── text_chunks
  └── embeddings
        └── text_chunks

sessions
  └── messages

keywords
  └── keyword_stats

synonyms
  ├── synonym_usage_stats
  └── synonym_quality_metrics
```

## 제약조건

### CHECK 제약조건

- `text_chunks.source_type`: 'statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph' 중 하나
- `keywords.weight_level`: 'high', 'medium', 'low' 중 하나
- `patterns.pattern_type`: 'regex', 'keyword', 'phrase' 중 하나

### UNIQUE 제약조건

- `domains.name`
- `statutes(domain_id, name)`
- `cases.doc_id`
- `decisions.doc_id`
- `interpretations.doc_id`
- `text_chunks(source_type, source_id, chunk_index, embedding_version_id)`
- `embedding_versions.version_name`
- `keywords(question_type, keyword)`
- `question_types.type_name`
- `synonyms(keyword, synonym, domain, context)`
- `processed_files.file_path`

## 인덱스 전략

### B-Tree 인덱스
- 외래 키 컬럼
- 자주 조회되는 컬럼 (doc_id, domain_id 등)
- 정렬에 사용되는 컬럼

### GIN 인덱스
- Full-Text Search용 tsvector 컬럼
- JSONB 메타데이터 컬럼

### IVFFlat 인덱스
- 벡터 검색용 vector 컬럼

### 부분 인덱스
- `embedding_versions`: is_active = TRUE인 경우만 인덱싱

## 성능 최적화

### 인덱스 활용
- 자주 사용되는 쿼리 패턴에 맞춰 인덱스 설계
- 복합 인덱스로 다중 컬럼 조회 최적화
- 부분 인덱스로 불필요한 인덱스 크기 감소

### 벡터 검색 최적화
- IVFFlat 인덱스로 빠른 근사 검색
- lists 파라미터 조정으로 정확도와 성능 균형

### Full-Text Search 최적화
- GIN 인덱스로 빠른 텍스트 검색
- 한국어 텍스트 검색 지원

## 참고 사항

- 모든 TIMESTAMP 컬럼은 UTC 기준입니다.
- JSONB 타입을 사용하여 유연한 메타데이터 저장이 가능합니다.
- 외래 키 제약조건으로 데이터 무결성을 보장합니다.
- CASCADE/SET NULL 정책으로 데이터 일관성을 유지합니다.

