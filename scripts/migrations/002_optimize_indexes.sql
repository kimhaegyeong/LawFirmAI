-- lawfirm_v2.db 인덱스 최적화
-- Phase 3: 데이터베이스 구조 최적화

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- 1. embeddings 테이블 인덱스 최적화
-- model별 인덱스 추가 (여러 모델 사용 시 성능 향상)
CREATE INDEX IF NOT EXISTS idx_embeddings_model_chunk
ON embeddings (model, chunk_id);

-- chunk_id 인덱스는 이미 있지만, 검색 성능 향상을 위해 확인
-- CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings (chunk_id);

-- 2. text_chunks 테이블 인덱스 최적화
-- source_type과 source_id 복합 인덱스 (이미 존재하지만 확인)
-- CREATE UNIQUE INDEX IF NOT EXISTS idx_text_chunks_source
-- ON text_chunks (source_type, source_id, chunk_index);

-- 검색 쿼리 최적화를 위한 추가 인덱스
CREATE INDEX IF NOT EXISTS idx_text_chunks_source_type
ON text_chunks (source_type);

CREATE INDEX IF NOT EXISTS idx_text_chunks_source_id
ON text_chunks (source_id);

-- text_chunks와 embeddings 조인 성능 향상
CREATE INDEX IF NOT EXISTS idx_text_chunks_id
ON text_chunks (id);

-- 3. statute_articles 인덱스 최적화 (검색 성능 향상)
CREATE INDEX IF NOT EXISTS idx_statute_articles_statute_text
ON statute_articles (statute_id, text);

-- 4. case_paragraphs 인덱스 최적화
CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case_text
ON case_paragraphs (case_id, text);

-- 5. decision_paragraphs 인덱스 최적화
CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision_text
ON decision_paragraphs (decision_id, text);

-- 6. interpretation_paragraphs 인덱스 최적화
CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interpretation_text
ON interpretation_paragraphs (interpretation_id, text);

-- 7. FTS5 인덱스 최적화 (가능한 경우)
-- FTS5는 자체 인덱스를 사용하지만, 통계 업데이트 권장
-- ANALYZE는 런타임에 실행

-- 8. 통계 업데이트 (쿼리 최적화를 위해)
ANALYZE;

-- 완료 로그
SELECT 'Index optimization completed successfully' AS status;
