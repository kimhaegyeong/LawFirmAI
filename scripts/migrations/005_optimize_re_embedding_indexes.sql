-- Migration: Optimize indexes for re-embedding performance
-- Date: 2024
-- Description: 재임베딩 성능 향상을 위한 인덱스 최적화

-- 1. case_paragraphs 테이블 인덱스
CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case_id_para_index 
ON case_paragraphs(case_id, para_index);

-- 2. decision_paragraphs 테이블 인덱스
CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision_id_para_index 
ON decision_paragraphs(decision_id, para_index);

-- 3. interpretation_paragraphs 테이블 인덱스
CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interpretation_id_para_index 
ON interpretation_paragraphs(interpretation_id, para_index);

-- 4. text_chunks 테이블 인덱스 (재임베딩 확인용)
CREATE INDEX IF NOT EXISTS idx_text_chunks_source_lookup 
ON text_chunks(source_type, source_id, chunking_strategy, embedding_version_id);

-- 5. text_chunks 테이블 인덱스 (chunk_index 조회용)
CREATE INDEX IF NOT EXISTS idx_text_chunks_source_chunk_index 
ON text_chunks(source_type, source_id, chunk_index);

