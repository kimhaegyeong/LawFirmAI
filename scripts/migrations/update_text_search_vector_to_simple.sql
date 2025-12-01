-- 기존 text_search_vector 컬럼을 'simple' 설정으로 재생성
-- 트리거 함수가 'korean'에서 'simple'로 변경되었으므로, 기존 데이터도 업데이트 필요

-- Statute Articles
UPDATE statute_articles
SET text_search_vector = to_tsvector('simple', COALESCE(text, ''))
WHERE text_search_vector IS NOT NULL;

-- Case Paragraphs
UPDATE case_paragraphs
SET text_search_vector = to_tsvector('simple', COALESCE(text, ''))
WHERE text_search_vector IS NOT NULL;

-- Decision Paragraphs
UPDATE decision_paragraphs
SET text_search_vector = to_tsvector('simple', COALESCE(text, ''))
WHERE text_search_vector IS NOT NULL;

-- Interpretation Paragraphs
UPDATE interpretation_paragraphs
SET text_search_vector = to_tsvector('simple', COALESCE(text, ''))
WHERE text_search_vector IS NOT NULL;

-- 인덱스 재생성 (선택적, 성능 최적화)
-- REINDEX INDEX idx_statute_articles_fts;
-- REINDEX INDEX idx_case_paragraphs_fts;
-- REINDEX INDEX idx_decision_paragraphs_fts;
-- REINDEX INDEX idx_interpretation_paragraphs_fts;

