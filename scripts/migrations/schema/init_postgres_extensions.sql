-- PostgreSQL 확장 설치 스크립트
-- Docker PostgreSQL 컨테이너 시작 시 자동 실행

-- 한국어 텍스트 검색을 위한 pg_trgm 확장
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 벡터 검색을 위한 pgvector 확장
CREATE EXTENSION IF NOT EXISTS vector;

