#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""재임베딩된 Version 5 인덱스를 외부 인덱스로 사용하기 위한 메타데이터 파일 생성"""

import sys
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_external_index_metadata(
    db_path: str,
    chunk_ids_json_path: str,
    output_json_path: str,
    embedding_version_id: int = 5
):
    """
    재임베딩된 인덱스를 위한 외부 인덱스 메타데이터 파일 생성
    
    Args:
        db_path: 데이터베이스 경로
        chunk_ids_json_path: chunk_ids.json 파일 경로
        output_json_path: 출력할 ml_enhanced_faiss_index.json 파일 경로
        embedding_version_id: 임베딩 버전 ID
    """
    logger.info("="*80)
    logger.info("외부 인덱스 메타데이터 파일 생성")
    logger.info("="*80)
    
    # chunk_ids.json 로드
    logger.info(f"Loading chunk_ids from {chunk_ids_json_path}")
    with open(chunk_ids_json_path, 'r', encoding='utf-8') as f:
        chunk_ids = json.load(f)
    
    logger.info(f"Loaded {len(chunk_ids)} chunk_ids")
    
    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    document_metadata = []
    document_texts = []
    
    logger.info("Fetching metadata and texts from database...")
    
    # 배치로 처리
    batch_size = 1000
    for i in range(0, len(chunk_ids), batch_size):
        batch_chunk_ids = chunk_ids[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch_chunk_ids))
        
        # text_chunks와 관련 테이블 조인하여 메타데이터 조회
        query = f"""
            SELECT 
                tc.id as chunk_id,
                tc.source_type,
                tc.source_id,
                tc.text,
                tc.chunk_index,
                tc.embedding_version_id,
                CASE 
                    WHEN tc.source_type = 'case_paragraph' THEN c.doc_id
                    WHEN tc.source_type = 'statute_article' THEN s.abbrv || ' 제' || sa.article_no || '조'
                    ELSE NULL
                END as doc_id,
                CASE 
                    WHEN tc.source_type = 'case_paragraph' THEN c.casenames
                    ELSE NULL
                END as casenames,
                CASE 
                    WHEN tc.source_type = 'case_paragraph' THEN c.court
                    ELSE NULL
                END as court
            FROM text_chunks tc
            LEFT JOIN cases c ON tc.source_type = 'case_paragraph' AND tc.source_id = c.id
            LEFT JOIN statute_articles sa ON tc.source_type = 'statute_article' AND tc.source_id = sa.id
            LEFT JOIN statutes s ON sa.statute_id = s.id
            WHERE tc.id IN ({placeholders})
            AND tc.embedding_version_id = ?
            ORDER BY tc.id
        """
        
        params = list(batch_chunk_ids) + [embedding_version_id]
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            chunk_id = row['chunk_id']
            source_type = row['source_type']
            source_id = row['source_id']
            text = row['text'] or ''
            doc_id = row['doc_id'] or ''
            
            # 메타데이터 구성
            metadata = {
                'chunk_id': chunk_id,
                'source_type': source_type,
                'source_id': source_id,
                'chunk_index': row['chunk_index'],
                'embedding_version_id': row['embedding_version_id']
            }
            
            # source_type별 추가 메타데이터
            if source_type == 'case_paragraph':
                metadata['case_id'] = doc_id if doc_id else f"case_{source_id}"
                if row['casenames']:
                    metadata['casenames'] = row['casenames']
                if row['court']:
                    metadata['court'] = row['court']
            elif source_type == 'statute_article':
                metadata['statute_id'] = source_id
                if doc_id:
                    metadata['statute_name'] = doc_id
            
            document_metadata.append(metadata)
            document_texts.append(text)
        
        if (i + batch_size) % 5000 == 0 or i + batch_size >= len(chunk_ids):
            logger.info(f"Processed {min(i + batch_size, len(chunk_ids))}/{len(chunk_ids)} chunks")
    
    conn.close()
    
    # JSON 파일 생성
    logger.info(f"Creating metadata JSON file: {output_json_path}")
    output_data = {
        'document_metadata': document_metadata,
        'document_texts': document_texts,
        'version': f'v{embedding_version_id}',
        'total_documents': len(document_metadata),
        'created_at': str(Path(chunk_ids_json_path).stat().st_mtime) if Path(chunk_ids_json_path).exists() else None
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Metadata file created: {output_json_path}")
    logger.info(f"   Total documents: {len(document_metadata)}")
    logger.info(f"   Total texts: {len(document_texts)}")
    
    return output_json_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="외부 인덱스 메타데이터 파일 생성")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--chunk-ids", default="data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.chunk_ids.json", help="chunk_ids.json 파일 경로")
    parser.add_argument("--output", default="data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.json", help="출력 JSON 파일 경로")
    parser.add_argument("--version-id", type=int, default=5, help="임베딩 버전 ID")
    
    args = parser.parse_args()
    
    create_external_index_metadata(
        db_path=args.db,
        chunk_ids_json_path=args.chunk_ids,
        output_json_path=args.output,
        embedding_version_id=args.version_id
    )

