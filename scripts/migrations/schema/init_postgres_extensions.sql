-- PostgreSQL 확장 설치 스크립트
-- Docker PostgreSQL 컨테이너 시작 시 자동 실행

-- 한국어 텍스트 검색을 위한 pg_trgm 확장
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 한국어 전문 검색을 위한 PGroonga 확장 (필수)
-- PGroonga는 한국어 형태소 분석을 지원하여 더 정확한 검색이 가능합니다.
-- Docker 환경에서는 PGroonga가 필수로 설치되어 있습니다.
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- 벡터 검색을 위한 pgvector 확장
CREATE EXTENSION IF NOT EXISTS vector;

