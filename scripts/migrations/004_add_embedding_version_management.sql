-- Migration: Add embedding version management
-- Date: 2024
-- Description: 임베딩 버전 관리 시스템 추가 (청킹 전략별)

-- 1. embedding_versions 테이블 생성
CREATE TABLE IF NOT EXISTS embedding_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name TEXT NOT NULL UNIQUE,
    chunking_strategy TEXT NOT NULL,
    model_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 0,
    metadata TEXT
);

-- 2. text_chunks 테이블에 embedding_version_id 컬럼 추가
ALTER TABLE text_chunks ADD COLUMN embedding_version_id INTEGER REFERENCES embedding_versions(id);

-- 3. embeddings 테이블에 version_id 컬럼 추가 (선택적, 추적용)
ALTER TABLE embeddings ADD COLUMN version_id INTEGER;

-- 4. 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_text_chunks_embedding_version 
ON text_chunks (embedding_version_id);

CREATE INDEX IF NOT EXISTS idx_embedding_versions_strategy 
ON embedding_versions (chunking_strategy, is_active);

CREATE INDEX IF NOT EXISTS idx_embedding_versions_active 
ON embedding_versions (is_active);

-- 5. 기본 버전 등록 (기존 데이터용)
INSERT OR IGNORE INTO embedding_versions (version_name, chunking_strategy, model_name, description, is_active)
VALUES 
    ('v1.0.0-standard', 'standard', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS', '기본 표준 청킹 전략', 1),
    ('v1.0.0-dynamic', 'dynamic', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS', '동적 청킹 전략', 0),
    ('v1.0.0-hybrid', 'hybrid', 'snunlp/KR-SBERT-V40K-klueNLI-augSTS', '하이브리드 청킹 전략', 0);

-- 6. 기존 청크에 기본 버전 ID 할당
UPDATE text_chunks 
SET embedding_version_id = (
    SELECT id FROM embedding_versions 
    WHERE chunking_strategy = COALESCE(text_chunks.chunking_strategy, 'standard') 
    AND is_active = 1 
    LIMIT 1
)
WHERE embedding_version_id IS NULL;

