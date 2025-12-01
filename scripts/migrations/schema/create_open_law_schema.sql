-- Open Law API 데이터 수집을 위한 PostgreSQL 스키마
-- 실행: psql -U username -d database_name -f create_open_law_schema.sql

-- 법령 메타데이터 테이블
CREATE TABLE IF NOT EXISTS statutes (
    id SERIAL PRIMARY KEY,
    law_id INTEGER NOT NULL UNIQUE,              -- 법령ID
    law_name_kr TEXT NOT NULL,                   -- 법령명(한글)
    law_name_hanja TEXT,                         -- 법령명(한자)
    law_name_en TEXT,                            -- 법령명(영어)
    law_abbrv TEXT,                              -- 법령약칭
    law_type TEXT,                               -- 법령종류
    law_type_code TEXT,                          -- 법종구분코드
    proclamation_date DATE,                      -- 공포일자
    proclamation_number INTEGER,                 -- 공포번호
    effective_date DATE,                         -- 시행일자
    ministry_code INTEGER,                       -- 소관부처코드
    ministry_name TEXT,                          -- 소관부처명
    amendment_type TEXT,                         -- 제개정구분
    domain TEXT,                                 -- 분야 (civil_law, criminal_law)
    raw_response_json JSONB,                     -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 조문 테이블
CREATE TABLE IF NOT EXISTS statutes_articles (
    id SERIAL PRIMARY KEY,
    statute_id INTEGER NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
    article_no TEXT NOT NULL,                    -- 조문번호 (예: "000200" = 제2조)
    article_title TEXT,                          -- 조문제목
    article_content TEXT NOT NULL,               -- 조문내용
    clause_no TEXT,                              -- 항번호
    clause_content TEXT,                         -- 항내용
    item_no TEXT,                                -- 호번호
    item_content TEXT,                           -- 호내용
    sub_item_no TEXT,                            -- 목번호
    sub_item_content TEXT,                       -- 목내용
    effective_date DATE,                         -- 조문시행일자
    raw_response_json JSONB,                     -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(statute_id, article_no, clause_no, item_no, sub_item_no)
);

-- 판례 메타데이터 테이블
CREATE TABLE IF NOT EXISTS precedents (
    id SERIAL PRIMARY KEY,
    precedent_id INTEGER NOT NULL UNIQUE,        -- 판례정보일련번호
    case_name TEXT NOT NULL,                     -- 사건명
    case_number TEXT,                            -- 사건번호
    decision_date DATE,                          -- 선고일자
    court_name TEXT,                             -- 법원명
    court_type_code INTEGER,                     -- 법원종류코드
    case_type_name TEXT,                         -- 사건종류명
    case_type_code INTEGER,                      -- 사건종류코드
    decision_type TEXT,                          -- 판결유형
    decision_result TEXT,                         -- 선고
    domain TEXT,                                 -- 분야 (civil_law, criminal_law)
    raw_response_json JSONB,                     -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 판례 본문 테이블
CREATE TABLE IF NOT EXISTS precedent_contents (
    id SERIAL PRIMARY KEY,
    precedent_id INTEGER NOT NULL REFERENCES precedents(id) ON DELETE CASCADE,
    section_type TEXT NOT NULL,                  -- 섹션 유형 (판시사항, 판결요지, 판례내용)
    section_content TEXT NOT NULL,               -- 섹션 내용
    referenced_articles TEXT,                    -- 참조조문
    referenced_precedents TEXT,                  -- 참조판례
    raw_response_json JSONB,                     -- 원본 JSON 응답
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 청킹된 판례 내용을 저장할 테이블
CREATE TABLE IF NOT EXISTS precedent_chunks (
    id SERIAL PRIMARY KEY,
    precedent_content_id INTEGER NOT NULL REFERENCES precedent_contents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_content TEXT NOT NULL,
    chunk_length INTEGER,
    metadata JSONB,  -- 메타데이터 저장
    embedding_vector VECTOR(768),  -- 벡터 임베딩 (pgvector 사용 시)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(precedent_content_id, chunk_index)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_statutes_domain ON statutes(domain);
CREATE INDEX IF NOT EXISTS idx_statutes_law_id ON statutes(law_id);
CREATE INDEX IF NOT EXISTS idx_statutes_law_name ON statutes(law_name_kr);
CREATE INDEX IF NOT EXISTS idx_articles_statute_id ON statutes_articles(statute_id);
CREATE INDEX IF NOT EXISTS idx_articles_article_no ON statutes_articles(article_no);
CREATE INDEX IF NOT EXISTS idx_precedents_domain ON precedents(domain);
CREATE INDEX IF NOT EXISTS idx_precedents_precedent_id ON precedents(precedent_id);
CREATE INDEX IF NOT EXISTS idx_precedents_case_name ON precedents(case_name);
CREATE INDEX IF NOT EXISTS idx_precedents_decision_date ON precedents(decision_date);
CREATE INDEX IF NOT EXISTS idx_precedent_contents_precedent_id ON precedent_contents(precedent_id);
CREATE INDEX IF NOT EXISTS idx_precedent_contents_section_type ON precedent_contents(section_type);
-- 복합 인덱스: 중복 체크 및 조회 성능 향상
CREATE UNIQUE INDEX IF NOT EXISTS idx_precedent_contents_precedent_section ON precedent_contents(precedent_id, section_type);
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_content_id ON precedent_chunks(precedent_content_id);
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_fts ON precedent_chunks USING gin(to_tsvector('simple', chunk_content));

-- 한국어 텍스트 검색 확장 설치
-- pg_trgm: trigram 기반 한국어 텍스트 검색 지원
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 벡터 임베딩 확장 설치 (pgvector)
-- 주의: pgvector 확장이 설치되어 있어야 합니다
-- 설치 방법: CREATE EXTENSION IF NOT EXISTS vector;
-- 만약 pgvector가 설치되지 않은 경우, embedding_vector 컬럼은 NULL로 유지됩니다

-- Full-Text Search 인덱스 (PostgreSQL)
-- 한국어 텍스트 검색을 위한 인덱스
-- pg_trgm 확장을 사용하여 한국어 텍스트 검색 성능 향상
CREATE INDEX IF NOT EXISTS idx_statutes_fts ON statutes USING gin(to_tsvector('simple', law_name_kr || ' ' || COALESCE(law_abbrv, '')));
CREATE INDEX IF NOT EXISTS idx_statutes_trgm ON statutes USING gin(law_name_kr gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_articles_fts ON statutes_articles USING gin(to_tsvector('simple', article_content));
CREATE INDEX IF NOT EXISTS idx_articles_trgm ON statutes_articles USING gin(article_content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_precedents_fts ON precedents USING gin(to_tsvector('simple', case_name));
CREATE INDEX IF NOT EXISTS idx_precedents_trgm ON precedents USING gin(case_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_precedent_contents_fts ON precedent_contents USING gin(to_tsvector('simple', section_content));
CREATE INDEX IF NOT EXISTS idx_precedent_contents_trgm ON precedent_contents USING gin(section_content gin_trgm_ops);

