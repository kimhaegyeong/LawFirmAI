-- Migration: Create view for chunk to source document mapping
-- Date: 2024
-- Description: 원본 문서와 청크를 연결하는 뷰 생성

-- 기존 뷰가 있으면 삭제
DROP VIEW IF EXISTS v_chunk_to_source;

-- 원본 문서와 청크를 연결하는 뷰 생성
CREATE VIEW v_chunk_to_source AS
SELECT 
    tc.id as chunk_id,
    tc.source_type,
    tc.source_id,
    tc.chunk_index,
    tc.text as chunk_text,
    tc.chunking_strategy,
    tc.chunk_size_category,
    tc.chunk_group_id,
    tc.original_document_id,
    tc.start_char,
    tc.end_char,
    tc.overlap_chars,
    -- 원본 문서 정보 (source_type별로 JOIN)
    CASE 
        WHEN tc.source_type = 'statute_article' THEN 
            sa.statute_name || ' 제' || COALESCE(sa.article_no, '') || '조'
        WHEN tc.source_type = 'case_paragraph' THEN 
            COALESCE(cp.court, '') || ' ' || COALESCE(cp.casenames, '')
        WHEN tc.source_type = 'decision_paragraph' THEN 
            COALESCE(dp.court, '') || ' ' || COALESCE(dp.casenames, '')
        WHEN tc.source_type = 'interpretation_paragraph' THEN 
            COALESCE(ip.title, '')
        ELSE 'Unknown'
    END as source_title,
    CASE 
        WHEN tc.source_type = 'statute_article' THEN sa.text
        WHEN tc.source_type = 'case_paragraph' THEN cp.text
        WHEN tc.source_type = 'decision_paragraph' THEN dp.text
        WHEN tc.source_type = 'interpretation_paragraph' THEN ip.text
        ELSE NULL
    END as original_text,
    -- 추가 메타데이터
    CASE 
        WHEN tc.source_type = 'statute_article' THEN sa.article_no
        WHEN tc.source_type = 'case_paragraph' THEN cp.doc_id
        WHEN tc.source_type = 'decision_paragraph' THEN dp.doc_id
        WHEN tc.source_type = 'interpretation_paragraph' THEN ip.id
        ELSE NULL
    END as source_identifier
FROM text_chunks tc
LEFT JOIN statute_articles sa 
    ON tc.source_type = 'statute_article' AND tc.source_id = sa.id
LEFT JOIN case_paragraphs cp 
    ON tc.source_type = 'case_paragraph' AND tc.source_id = cp.id
LEFT JOIN decision_paragraphs dp 
    ON tc.source_type = 'decision_paragraph' AND tc.source_id = dp.id
LEFT JOIN interpretation_paragraphs ip 
    ON tc.source_type = 'interpretation_paragraph' AND tc.source_id = ip.id;

