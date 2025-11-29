-- PGroonga 인덱스 생성 스크립트
-- PostgreSQL에서 한국어 텍스트 검색 성능 향상을 위한 PGroonga 인덱스 생성

-- PGroonga 확장 확인 및 설치
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- 기존 'simple' 인덱스는 유지하고, PGroonga 인덱스를 추가로 생성
-- PGroonga는 한국어 형태소 분석을 지원하여 더 정확한 검색이 가능합니다

-- 1. statutes_articles 테이블 - article_content 컬럼
-- PGroonga 인덱스 생성 (한국어 형태소 분석 지원)
CREATE INDEX IF NOT EXISTS idx_statutes_articles_pgroonga 
ON statutes_articles USING pgroonga (article_content);

-- 2. precedent_contents 테이블 - section_content 컬럼
-- PGroonga 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_precedent_contents_pgroonga 
ON precedent_contents USING pgroonga (section_content);

-- 3. precedent_chunks 테이블 - chunk_content 컬럼
-- PGroonga 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_precedent_chunks_pgroonga 
ON precedent_chunks USING pgroonga (chunk_content);

-- 4. statutes 테이블 - law_name_kr 컬럼
-- PGroonga 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_statutes_pgroonga 
ON statutes USING pgroonga (law_name_kr);

-- 5. precedents 테이블 - case_name 컬럼
-- PGroonga 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_precedents_pgroonga 
ON precedents USING pgroonga (case_name);

-- 인덱스 생성 완료 로그
DO $$
BEGIN
    RAISE NOTICE 'PGroonga indexes created successfully';
END $$;

