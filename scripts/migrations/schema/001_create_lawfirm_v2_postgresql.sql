-- lawfirm_v2 PostgreSQL schema
-- PostgreSQL + pgvector + Full-Text Search

-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- Domains and Sources
CREATE TABLE IF NOT EXISTS domains (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(50) NOT NULL, -- statute, case, decision, interpretation
    path TEXT NOT NULL,
    hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ⚠️ 사용 중단 예정 테이블 (DEPRECATED)
-- Open Law 스키마의 테이블들로 대체되었습니다.
-- 삭제 계획: docs/10_technical_reference/postgresql_migration_improvement_plan.md 참조
-- ============================================

-- Statutes (DEPRECATED - Open Law 스키마의 statutes, statutes_articles로 대체)
-- CREATE TABLE IF NOT EXISTS statutes (
--     id SERIAL PRIMARY KEY,
--     domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
--     name VARCHAR(255) NOT NULL,
--     abbrv VARCHAR(100),
--     statute_type VARCHAR(50),
--     proclamation_date TEXT,
--     effective_date TEXT,
--     category VARCHAR(50),
--     UNIQUE(domain_id, name)
-- );

-- CREATE TABLE IF NOT EXISTS statute_articles (
--     id SERIAL PRIMARY KEY,
--     statute_id INTEGER NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
--     article_no VARCHAR(50) NOT NULL,   -- 제n조
--     clause_no VARCHAR(50),             -- 항
--     item_no VARCHAR(50),               -- 호
--     heading TEXT,                      -- 조문 제목
--     text TEXT NOT NULL,
--     version_effective_date TEXT,
--     -- PostgreSQL FTS를 위한 tsvector 컬럼
--     text_search_vector tsvector
-- );

-- CREATE INDEX IF NOT EXISTS idx_statute_articles_keys
-- ON statute_articles (statute_id, article_no, clause_no, item_no);

-- -- GIN 인덱스 (Full-Text Search)
-- CREATE INDEX IF NOT EXISTS idx_statute_articles_fts 
-- ON statute_articles USING GIN (text_search_vector);

-- Cases (DEPRECATED - Open Law 스키마의 precedents, precedent_contents로 대체)
-- CREATE TABLE IF NOT EXISTS cases (
--     id SERIAL PRIMARY KEY,
--     domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
--     doc_id VARCHAR(255) NOT NULL UNIQUE,
--     court VARCHAR(100),
--     case_type VARCHAR(50),
--     casenames TEXT,
--     announce_date TEXT
-- );

-- CREATE TABLE IF NOT EXISTS case_paragraphs (
--     id SERIAL PRIMARY KEY,
--     case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
--     para_index INTEGER NOT NULL,
--     text TEXT NOT NULL,
--     -- PostgreSQL FTS를 위한 tsvector 컬럼
--     text_search_vector tsvector,
--     UNIQUE(case_id, para_index)
-- );

-- CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case
-- ON case_paragraphs (case_id, para_index);

-- -- GIN 인덱스 (Full-Text Search)
-- CREATE INDEX IF NOT EXISTS idx_case_paragraphs_fts 
-- ON case_paragraphs USING GIN (text_search_vector);

-- Decisions (DEPRECATED - 더 이상 사용하지 않음)
-- CREATE TABLE IF NOT EXISTS decisions (
--     id SERIAL PRIMARY KEY,
--     domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
--     org VARCHAR(255),                   -- 기관
--     doc_id VARCHAR(255) NOT NULL UNIQUE,
--     decision_date TEXT,
--     result TEXT
-- );

-- CREATE TABLE IF NOT EXISTS decision_paragraphs (
--     id SERIAL PRIMARY KEY,
--     decision_id INTEGER NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
--     para_index INTEGER NOT NULL,
--     text TEXT NOT NULL,
--     -- PostgreSQL FTS를 위한 tsvector 컬럼
--     text_search_vector tsvector,
--     UNIQUE(decision_id, para_index)
-- );

-- CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision
-- ON decision_paragraphs (decision_id, para_index);

-- -- GIN 인덱스 (Full-Text Search)
-- CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_fts 
-- ON decision_paragraphs USING GIN (text_search_vector);

-- Interpretations (DEPRECATED - 더 이상 사용하지 않음)
-- CREATE TABLE IF NOT EXISTS interpretations (
--     id SERIAL PRIMARY KEY,
--     domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
--     org VARCHAR(255),
--     doc_id VARCHAR(255) NOT NULL UNIQUE,
--     title TEXT,
--     response_date TEXT
-- );

-- CREATE TABLE IF NOT EXISTS interpretation_paragraphs (
--     id SERIAL PRIMARY KEY,
--     interpretation_id INTEGER NOT NULL REFERENCES interpretations(id) ON DELETE CASCADE,
--     para_index INTEGER NOT NULL,
--     text TEXT NOT NULL,
--     -- PostgreSQL FTS를 위한 tsvector 컬럼
--     text_search_vector tsvector,
--     UNIQUE(interpretation_id, para_index)
-- );

-- CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interp
-- ON interpretation_paragraphs (interpretation_id, para_index);

-- -- GIN 인덱스 (Full-Text Search)
-- CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_fts 
-- ON interpretation_paragraphs USING GIN (text_search_vector);

-- Embeddings (pgvector 사용)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL,  -- chunk_id는 다른 테이블 참조 (precedent_chunks 등)
    model VARCHAR(255) NOT NULL,
    dim INTEGER NOT NULL,
    version_id INTEGER REFERENCES embedding_versions(id) ON DELETE SET NULL,
    vector VECTOR(768) NOT NULL,  -- pgvector 타입 (차원은 모델에 따라 조정)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunk
ON embeddings (chunk_id);

-- 벡터 검색 인덱스 (IVFFlat - 빠른 검색)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON embeddings USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

-- 또는 HNSW (더 빠르지만 더 많은 메모리 사용) - 주석 처리
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
-- ON embeddings USING hnsw (vector vector_cosine_ops);

-- Embedding versions
CREATE TABLE IF NOT EXISTS embedding_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR(255) NOT NULL UNIQUE,
    chunking_strategy VARCHAR(50) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,  -- PostgreSQL 네이티브 BOOLEAN
    metadata JSONB,                    -- JSONB 타입
    UNIQUE(version_name)
);

-- 활성 버전 조회 최적화 (부분 인덱스)
CREATE INDEX IF NOT EXISTS idx_embedding_versions_active 
ON embedding_versions (id) WHERE is_active = TRUE;

-- Optional cache
CREATE TABLE IF NOT EXISTS retrieval_cache (
    query_hash VARCHAR(255) PRIMARY KEY,
    topk_ids TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ⚠️ 사용 중단 예정 FTS 트리거 함수들 (DEPRECATED)
-- 사용 중단 테이블과 함께 제거 예정
-- ============================================

-- -- Statute Articles FTS 트리거 (DEPRECATED)
-- CREATE OR REPLACE FUNCTION update_statute_articles_fts() RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.text_search_vector := to_tsvector('simple', COALESCE(NEW.text, ''));
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- CREATE TRIGGER trigger_statute_articles_fts_update
-- BEFORE INSERT OR UPDATE ON statute_articles
-- FOR EACH ROW EXECUTE FUNCTION update_statute_articles_fts();

-- -- Case Paragraphs FTS 트리거 (DEPRECATED)
-- CREATE OR REPLACE FUNCTION update_case_paragraphs_fts() RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.text_search_vector := to_tsvector('simple', COALESCE(NEW.text, ''));
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- CREATE TRIGGER trigger_case_paragraphs_fts_update
-- BEFORE INSERT OR UPDATE ON case_paragraphs
-- FOR EACH ROW EXECUTE FUNCTION update_case_paragraphs_fts();

-- -- Decision Paragraphs FTS 트리거 (DEPRECATED)
-- CREATE OR REPLACE FUNCTION update_decision_paragraphs_fts() RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.text_search_vector := to_tsvector('simple', COALESCE(NEW.text, ''));
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- CREATE TRIGGER trigger_decision_paragraphs_fts_update
-- BEFORE INSERT OR UPDATE ON decision_paragraphs
-- FOR EACH ROW EXECUTE FUNCTION update_decision_paragraphs_fts();

-- -- Interpretation Paragraphs FTS 트리거 (DEPRECATED)
-- CREATE OR REPLACE FUNCTION update_interpretation_paragraphs_fts() RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.text_search_vector := to_tsvector('simple', COALESCE(NEW.text, ''));
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- CREATE TRIGGER trigger_interpretation_paragraphs_fts_update
-- BEFORE INSERT OR UPDATE ON interpretation_paragraphs
-- FOR EACH ROW EXECUTE FUNCTION update_interpretation_paragraphs_fts();

-- 제약조건 추가
-- (text_chunks 테이블은 더 이상 사용되지 않으므로 제거됨)

ALTER TABLE embeddings 
ADD CONSTRAINT IF NOT EXISTS fk_embeddings_version 
FOREIGN KEY (version_id) 
REFERENCES embedding_versions(id) 
ON DELETE SET NULL;

-- ============================================
-- API 서버용 테이블 (인증 및 세션 관리)
-- ============================================

-- Users (회원 정보)
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255),
    name TEXT,
    picture TEXT,
    provider VARCHAR(50),
    google_access_token TEXT,
    google_refresh_token TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users 인덱스
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_provider ON users(provider);

-- Sessions (세션 정보)
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    user_id VARCHAR(255),
    ip_address VARCHAR(45)
);

-- Sessions 인덱스
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

-- Messages (메시지 정보)
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Messages 외래 키 및 인덱스
ALTER TABLE messages 
ADD CONSTRAINT IF NOT EXISTS fk_messages_session_id 
FOREIGN KEY (session_id) 
REFERENCES sessions(session_id) 
ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_metadata_gin ON messages USING GIN (metadata);

