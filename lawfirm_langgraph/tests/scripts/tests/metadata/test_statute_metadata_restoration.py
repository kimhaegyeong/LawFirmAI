# -*- coding: utf-8 -*-
"""법령 메타데이터 복원 문제 디버깅"""

import sys
import os
from pathlib import Path
import sqlite3

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
    scripts_dir = script_dir.parent
    tests_dir = scripts_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 설정
os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = str(project_root / "data" / "vector_store" / "v2.0.0-dynamic-dynamic-ivfpq")

logger.info("=" * 60)
logger.info("법령 메타데이터 복원 문제 디버깅")
logger.info("=" * 60)

# SemanticSearchEngineV2 초기화
config = Config()
search_engine = SemanticSearchEngineV2(
    db_path=config.database_path,
    use_external_index=config.use_external_vector_store,
    external_index_path=config.external_vector_store_base_path
)

# 검색 쿼리
query = "임대차 보증금 반환"
logger.info(f"\n검색 쿼리: {query}")

# 법령만 검색
statute_results = search_engine.search(
    query, 
    k=5, 
    similarity_threshold=0.15,
    source_types=["statute_article"]
)

logger.info(f"\n법령 검색 결과: {len(statute_results)}개")

if statute_results:
    for i, result in enumerate(statute_results, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"결과 {i}:")
        logger.info(f"  chunk_id: {result.get('metadata', {}).get('chunk_id')}")
        logger.info(f"  source_id: {result.get('metadata', {}).get('source_id')}")
        logger.info(f"  source_type: {result.get('type')}")
        logger.info(f"  score: {result.get('score', 0):.4f}")
        logger.info(f"  statute_name: {result.get('statute_name')}")
        logger.info(f"  law_name: {result.get('law_name')}")
        logger.info(f"  article_no: {result.get('article_no')}")
        logger.info(f"  article_number: {result.get('article_number')}")
        logger.info(f"  metadata.statute_name: {result.get('metadata', {}).get('statute_name')}")
        logger.info(f"  metadata.article_no: {result.get('metadata', {}).get('article_no')}")
        
        # DB에서 직접 확인
        chunk_id = result.get('metadata', {}).get('chunk_id')
        source_id = result.get('metadata', {}).get('source_id')
        
        if chunk_id and source_id:
            conn = sqlite3.connect(config.database_path)
            conn.row_factory = sqlite3.Row
            
            # text_chunks 확인
            cursor = conn.execute("""
                SELECT source_type, source_id, meta
                FROM text_chunks
                WHERE id = ?
            """, (chunk_id,))
            chunk_row = cursor.fetchone()
            if chunk_row:
                logger.info(f"\n  DB text_chunks:")
                logger.info(f"    source_type: {chunk_row['source_type']}")
                logger.info(f"    source_id: {chunk_row['source_id']}")
                logger.info(f"    meta: {chunk_row['meta']}")
            
            # statute_articles 확인
            cursor = conn.execute("""
                SELECT sa.id, sa.article_no, s.name as statute_name
                FROM statute_articles sa
                JOIN statutes s ON sa.statute_id = s.id
                WHERE sa.id = ?
            """, (source_id,))
            statute_row = cursor.fetchone()
            if statute_row:
                logger.info(f"\n  DB statute_articles:")
                logger.info(f"    id: {statute_row['id']}")
                logger.info(f"    article_no: {statute_row['article_no']}")
                logger.info(f"    statute_name: {statute_row['statute_name']}")
            else:
                logger.warning(f"  ⚠️  statute_articles에서 source_id={source_id}를 찾을 수 없습니다")
            
            # _get_source_metadata 테스트
            source_meta = search_engine._get_source_metadata(conn, "statute_article", source_id)
            logger.info(f"\n  _get_source_metadata 결과:")
            logger.info(f"    {source_meta}")
            
            conn.close()
else:
    logger.warning("⚠️  법령 검색 결과가 없습니다")

