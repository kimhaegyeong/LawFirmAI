-- 데이터베이스 연결 최적화를 위한 인덱스 추가
-- 실행: psql -U postgres -d lawfirm_db -f scripts/migrations/add_connection_optimization_indexes.sql
-- 또는: python scripts/migrations/scripts/init/run_postgresql_migration.py

BEGIN;

-- ============================================
-- 1. embeddings 테이블 인덱스
-- ============================================

-- version_id로 필터링하는 쿼리 최적화 (부분 인덱스)
CREATE INDEX IF NOT EXISTS idx_embeddings_version_id 
ON embeddings(version_id) 
WHERE version_id IS NOT NULL;

-- chunk_id와 version_id 복합 인덱스 (자주 함께 사용됨)
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_version 
ON embeddings(chunk_id, version_id);

-- model로 필터링하는 쿼리 최적화
CREATE INDEX IF NOT EXISTS idx_embeddings_model 
ON embeddings(model) 
WHERE model IS NOT NULL;

-- chunk_id 단독 인덱스 (이미 있을 수 있지만 확인)
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id 
ON embeddings(chunk_id);

-- ============================================
-- 2. precedent_chunks 테이블 인덱스
-- ============================================

-- embedding_vector가 NULL이 아닌 경우만 인덱싱 (부분 인덱스)
-- 자주 사용되는 조건: WHERE embedding_vector IS NOT NULL
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_has_embedding 
ON precedent_chunks(id, embedding_version) 
WHERE embedding_vector IS NOT NULL;

-- embedding_version으로 필터링 (부분 인덱스)
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_embedding_version 
ON precedent_chunks(embedding_version) 
WHERE embedding_version IS NOT NULL;

-- precedent_content_id와 embedding_version 복합 인덱스
-- JOIN 및 필터링 최적화
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_content_version 
ON precedent_chunks(precedent_content_id, embedding_version);

-- id 단독 인덱스 (이미 PRIMARY KEY가 있지만 명시적으로 추가)
-- WHERE id IN (...) 쿼리 최적화
-- 주의: PRIMARY KEY가 이미 인덱스를 제공하므로 중복이지만 명시적으로 유지

-- ============================================
-- 3. precedent_contents 테이블 인덱스
-- ============================================

-- section_content가 비어있지 않은 경우 인덱스
CREATE INDEX IF NOT EXISTS idx_precedent_contents_has_section_content 
ON precedent_contents(id) 
WHERE section_content IS NOT NULL AND section_content != '';

-- precedent_id와 section_type 복합 인덱스 (이미 있을 수 있지만 확인)
-- CREATE INDEX IF NOT EXISTS idx_precedent_contents_precedent_section 
-- ON precedent_contents(precedent_id, section_type);

-- ============================================
-- 4. statute_embeddings 테이블 인덱스
-- ============================================

-- embedding_vector가 NULL이 아닌 경우만 인덱싱 (부분 인덱스)
CREATE INDEX IF NOT EXISTS idx_statute_embeddings_has_embedding 
ON statute_embeddings(article_id, embedding_version) 
WHERE embedding_vector IS NOT NULL;

-- embedding_version으로 필터링
CREATE INDEX IF NOT EXISTS idx_statute_embeddings_embedding_version 
ON statute_embeddings(embedding_version) 
WHERE embedding_version IS NOT NULL;

-- article_id 단독 인덱스 (WHERE article_id IN (...) 최적화)
CREATE INDEX IF NOT EXISTS idx_statute_embeddings_article_id 
ON statute_embeddings(article_id);

-- ============================================
-- 5. embedding_versions 테이블 인덱스
-- ============================================

-- is_active와 data_type 복합 인덱스 (자주 함께 사용됨)
CREATE INDEX IF NOT EXISTS idx_embedding_versions_active_data_type 
ON embedding_versions(is_active, data_type) 
WHERE is_active = TRUE;

-- version 단독 인덱스
CREATE INDEX IF NOT EXISTS idx_embedding_versions_version 
ON embedding_versions(version);

COMMIT;

-- ============================================
-- 인덱스 생성 확인
-- ============================================
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('embeddings', 'precedent_chunks', 'precedent_contents', 'statute_embeddings', 'embedding_versions')
  AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;

-- ============================================
-- 인덱스 크기 확인
-- ============================================
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE tablename IN ('embeddings', 'precedent_chunks', 'precedent_contents', 'statute_embeddings', 'embedding_versions')
  AND indexname LIKE 'idx_%'
ORDER BY pg_relation_size(indexname::regclass) DESC;

