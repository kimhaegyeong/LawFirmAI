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

-- Statutes
CREATE TABLE IF NOT EXISTS statutes (
    id SERIAL PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    name VARCHAR(255) NOT NULL,
    abbrv VARCHAR(100),
    statute_type VARCHAR(50),
    proclamation_date TEXT,
    effective_date TEXT,
    category VARCHAR(50),
    UNIQUE(domain_id, name)
);

CREATE TABLE IF NOT EXISTS statute_articles (
    id SERIAL PRIMARY KEY,
    statute_id INTEGER NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
    article_no VARCHAR(50) NOT NULL,   -- 제n조
    clause_no VARCHAR(50),             -- 항
    item_no VARCHAR(50),               -- 호
    heading TEXT,                      -- 조문 제목
    text TEXT NOT NULL,
    version_effective_date TEXT,
    -- PostgreSQL FTS를 위한 tsvector 컬럼
    text_search_vector tsvector
);

CREATE INDEX IF NOT EXISTS idx_statute_articles_keys
ON statute_articles (statute_id, article_no, clause_no, item_no);

-- GIN 인덱스 (Full-Text Search)
CREATE INDEX IF NOT EXISTS idx_statute_articles_fts 
ON statute_articles USING GIN (text_search_vector);

-- Cases (판결문)
CREATE TABLE IF NOT EXISTS cases (
    id SERIAL PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    doc_id VARCHAR(255) NOT NULL UNIQUE,
    court VARCHAR(100),
    case_type VARCHAR(50),
    casenames TEXT,
    announce_date TEXT
);

CREATE TABLE IF NOT EXISTS case_paragraphs (
    id SERIAL PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    -- PostgreSQL FTS를 위한 tsvector 컬럼
    text_search_vector tsvector,
    UNIQUE(case_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case
ON case_paragraphs (case_id, para_index);

-- GIN 인덱스 (Full-Text Search)
CREATE INDEX IF NOT EXISTS idx_case_paragraphs_fts 
ON case_paragraphs USING GIN (text_search_vector);

-- Decisions (심결례 등)
CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    org VARCHAR(255),                   -- 기관
    doc_id VARCHAR(255) NOT NULL UNIQUE,
    decision_date TEXT,
    result TEXT
);

CREATE TABLE IF NOT EXISTS decision_paragraphs (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    -- PostgreSQL FTS를 위한 tsvector 컬럼
    text_search_vector tsvector,
    UNIQUE(decision_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision
ON decision_paragraphs (decision_id, para_index);

-- GIN 인덱스 (Full-Text Search)
CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_fts 
ON decision_paragraphs USING GIN (text_search_vector);

-- Interpretations (유권해석)
CREATE TABLE IF NOT EXISTS interpretations (
    id SERIAL PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    org VARCHAR(255),
    doc_id VARCHAR(255) NOT NULL UNIQUE,
    title TEXT,
    response_date TEXT
);

CREATE TABLE IF NOT EXISTS interpretation_paragraphs (
    id SERIAL PRIMARY KEY,
    interpretation_id INTEGER NOT NULL REFERENCES interpretations(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    -- PostgreSQL FTS를 위한 tsvector 컬럼
    text_search_vector tsvector,
    UNIQUE(interpretation_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interp
ON interpretation_paragraphs (interpretation_id, para_index);

-- GIN 인덱스 (Full-Text Search)
CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_fts 
ON interpretation_paragraphs USING GIN (text_search_vector);

-- Vector store meta
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(50) NOT NULL,           -- statute_article | case_paragraph | decision_paragraph | interpretation_paragraph
    source_id INTEGER NOT NULL,                 -- FK to corresponding table
    level VARCHAR(50),                          -- article/clause/item or paragraph
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    overlap_chars INTEGER,
    text TEXT NOT NULL,
    token_count INTEGER,
    embedding_version_id INTEGER REFERENCES embedding_versions(id) ON DELETE SET NULL,
    chunk_size_category VARCHAR(20),
    chunk_group_id VARCHAR(255),
    chunking_strategy VARCHAR(50),
    meta JSONB,                                 -- JSONB 타입 (인덱싱 및 쿼리 최적화)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, source_id, chunk_index, embedding_version_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_text_chunks_source
ON text_chunks (source_type, source_id, chunk_index, embedding_version_id);

-- JSONB 인덱스 (메타데이터 쿼리 최적화)
CREATE INDEX IF NOT EXISTS idx_text_chunks_meta_gin 
ON text_chunks USING GIN (meta);

-- 복합 인덱스
CREATE INDEX IF NOT EXISTS idx_text_chunks_version_type 
ON text_chunks (embedding_version_id, source_type);

-- Embeddings (pgvector 사용)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES text_chunks(id) ON DELETE CASCADE,
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

-- FTS 트리거 함수들
-- Statute Articles FTS 트리거
CREATE OR REPLACE FUNCTION update_statute_articles_fts() RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_statute_articles_fts_update
BEFORE INSERT OR UPDATE ON statute_articles
FOR EACH ROW EXECUTE FUNCTION update_statute_articles_fts();

-- Case Paragraphs FTS 트리거
CREATE OR REPLACE FUNCTION update_case_paragraphs_fts() RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_case_paragraphs_fts_update
BEFORE INSERT OR UPDATE ON case_paragraphs
FOR EACH ROW EXECUTE FUNCTION update_case_paragraphs_fts();

-- Decision Paragraphs FTS 트리거
CREATE OR REPLACE FUNCTION update_decision_paragraphs_fts() RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_decision_paragraphs_fts_update
BEFORE INSERT OR UPDATE ON decision_paragraphs
FOR EACH ROW EXECUTE FUNCTION update_decision_paragraphs_fts();

-- Interpretation Paragraphs FTS 트리거
CREATE OR REPLACE FUNCTION update_interpretation_paragraphs_fts() RETURNS TRIGGER AS $$
BEGIN
    NEW.text_search_vector := to_tsvector('korean', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_interpretation_paragraphs_fts_update
BEFORE INSERT OR UPDATE ON interpretation_paragraphs
FOR EACH ROW EXECUTE FUNCTION update_interpretation_paragraphs_fts();

-- 제약조건 추가
-- text_chunks source_type CHECK 제약조건
ALTER TABLE text_chunks 
ADD CONSTRAINT IF NOT EXISTS chk_text_chunks_source_type 
CHECK (source_type IN ('statute_article', 'case_paragraph', 'decision_paragraph', 'interpretation_paragraph'));

-- 외래 키 제약조건 명시 (이미 위에서 정의했지만 명시적으로 추가)
ALTER TABLE text_chunks 
ADD CONSTRAINT IF NOT EXISTS fk_text_chunks_embedding_version 
FOREIGN KEY (embedding_version_id) 
REFERENCES embedding_versions(id) 
ON DELETE SET NULL;

ALTER TABLE embeddings 
ADD CONSTRAINT IF NOT EXISTS fk_embeddings_version 
FOREIGN KEY (version_id) 
REFERENCES embedding_versions(id) 
ON DELETE SET NULL;

