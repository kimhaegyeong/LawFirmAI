-- Migration: Add chunking metadata columns to text_chunks table
-- Date: 2024
-- Description: 청킹 전략 및 메타데이터를 위한 컬럼 추가

-- 1. text_chunks 테이블에 청킹 메타데이터 컬럼 추가
ALTER TABLE text_chunks ADD COLUMN chunking_strategy TEXT;
ALTER TABLE text_chunks ADD COLUMN chunk_size_category TEXT;
ALTER TABLE text_chunks ADD COLUMN query_type TEXT;
ALTER TABLE text_chunks ADD COLUMN chunk_group_id TEXT;
ALTER TABLE text_chunks ADD COLUMN original_document_id INTEGER;

-- 2. 인덱스 추가 (검색 성능 향상)
CREATE INDEX IF NOT EXISTS idx_text_chunks_chunk_group_id 
ON text_chunks (chunk_group_id);

CREATE INDEX IF NOT EXISTS idx_text_chunks_chunking_strategy_category 
ON text_chunks (chunking_strategy, chunk_size_category);

CREATE INDEX IF NOT EXISTS idx_text_chunks_source_lookup 
ON text_chunks (source_type, source_id, chunk_group_id);

-- 3. 기존 데이터에 기본값 설정 (선택적)
-- 기존 청크는 'standard' 전략으로 설정
UPDATE text_chunks 
SET chunking_strategy = 'standard',
    chunk_size_category = CASE 
        WHEN LENGTH(text) < 800 THEN 'small'
        WHEN LENGTH(text) < 1500 THEN 'medium'
        ELSE 'large'
    END
WHERE chunking_strategy IS NULL;

