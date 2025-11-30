-- 성능 최적화를 위한 인덱스 추가
-- 실행: psql -U postgres -d lawfirmai_local -f scripts/migrations/add_performance_indexes.sql
-- 또는: python scripts/migrations/scripts/init/run_performance_indexes_migration.py

-- 🔥 메모리 설정: 인덱스 생성에 필요한 메모리 증가
-- 인덱스 생성 시 더 많은 메모리 사용 허용 (세션 레벨)
SET maintenance_work_mem = '256MB';

-- ============================================
-- 1. pgvector 복합 부분 인덱스 (CRITICAL)
-- ============================================

-- precedent_chunks 테이블: 벡터 검색 + embedding_version 필터 최적화
-- 데이터 크기: 약 114,203개 청크 → lists = sqrt(114203) ≈ 338
-- 🔥 수정: WITH 절은 WHERE 절 앞에 와야 함
-- 🔥 메모리 부족 방지를 위해 lists 값을 150으로 설정 (338에서 감소)
BEGIN;
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_vector_version 
ON precedent_chunks USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 150)
WHERE embedding_vector IS NOT NULL AND embedding_version IS NOT NULL;
COMMIT;

-- statute_embeddings 테이블: 벡터 검색 + embedding_version 필터 최적화
-- 데이터 크기에 따라 lists 조정 (기본 100)
BEGIN;
CREATE INDEX IF NOT EXISTS idx_statute_embeddings_vector_version 
ON statute_embeddings USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 100)
WHERE embedding_vector IS NOT NULL AND embedding_version IS NOT NULL;
COMMIT;

-- ============================================
-- 2. JOIN 최적화 인덱스 (HIGH)
-- ============================================

-- precedent_contents 테이블: JOIN 최적화
-- precedent_chunks → precedent_contents JOIN 성능 향상
BEGIN;
CREATE INDEX IF NOT EXISTS idx_precedent_contents_id_precedent_id 
ON precedent_contents(id, precedent_id);
COMMIT;

-- precedents 테이블: domain 필터링 최적화
BEGIN;
CREATE INDEX IF NOT EXISTS idx_precedents_domain_id 
ON precedents(domain, id) 
WHERE domain IS NOT NULL;
COMMIT;

-- precedent_chunks 테이블: JOIN + 필터링 복합 인덱스
BEGIN;
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_content_join 
ON precedent_chunks(precedent_content_id, embedding_version, id)
WHERE embedding_vector IS NOT NULL;
COMMIT;

-- ============================================
-- 3. 통계 정보 업데이트 (쿼리 플래너 최적화)
-- ============================================

ANALYZE precedent_chunks;
ANALYZE statute_embeddings;
ANALYZE precedent_contents;
ANALYZE precedents;
ANALYZE embedding_versions;

-- ============================================
-- 인덱스 생성 확인
-- ============================================
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('precedent_chunks', 'statute_embeddings', 'precedent_contents', 'precedents')
  AND indexname LIKE 'idx_%'
  AND (indexname LIKE '%vector%' OR indexname LIKE '%join%' OR indexname LIKE '%domain%')
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
WHERE tablename IN ('precedent_chunks', 'statute_embeddings', 'precedent_contents', 'precedents')
  AND indexname LIKE 'idx_%'
  AND (indexname LIKE '%vector%' OR indexname LIKE '%join%' OR indexname LIKE '%domain%')
ORDER BY pg_relation_size(indexname::regclass) DESC;
