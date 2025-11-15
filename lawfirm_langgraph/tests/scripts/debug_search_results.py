# -*- coding: utf-8 -*-
"""
검색 결과 디버깅 스크립트
"""

import sys
import os
from pathlib import Path
import logging

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_search_results():
    """검색 결과 디버깅"""
    db_path = project_root / "data" / "lawfirm_v2.db"
    if not db_path.exists():
        logger.error(f"데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return
    
    # 검색 엔진 초기화
    logger.info("검색 엔진 초기화 중...")
    try:
        search_engine = SemanticSearchEngineV2(db_path=str(db_path))
        logger.info("✅ 검색 엔진 초기화 완료")
    except Exception as e:
        logger.error(f"검색 엔진 초기화 실패: {e}")
        return
    
    # 테스트 쿼리
    query = "임대차 보증금"
    logger.info(f"\n테스트 쿼리: {query}")
    
    try:
        # 검색 실행
        results = search_engine.search(
            query=query,
            k=3,
            similarity_threshold=0.3
        )
        
        logger.info(f"\n검색 결과: {len(results)}개")
        
        # 첫 번째 결과 상세 분석
        if results:
            result = results[0]
            logger.info("\n첫 번째 결과 상세:")
            logger.info(f"  id: {result.get('id')}")
            logger.info(f"  embedding_version_id (최상위): {result.get('embedding_version_id')}")
            logger.info(f"  embedding_version_id (metadata): {result.get('metadata', {}).get('embedding_version_id')}")
            logger.info(f"  chunk_id (metadata): {result.get('metadata', {}).get('chunk_id')}")
            logger.info(f"  source_id (metadata): {result.get('metadata', {}).get('source_id')}")
            logger.info(f"  source_type: {result.get('type')}")
            logger.info(f"  metadata keys: {list(result.get('metadata', {}).keys())}")
            
            # chunk_id로 직접 DB 조회
            chunk_id = result.get('metadata', {}).get('chunk_id')
            if chunk_id:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT id, embedding_version_id, source_type, source_id FROM text_chunks WHERE id = ?",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    logger.info(f"\n  DB에서 직접 조회:")
                    logger.info(f"    chunk_id: {row['id']}")
                    logger.info(f"    embedding_version_id: {row['embedding_version_id']}")
                    logger.info(f"    source_type: {row['source_type']}")
                    logger.info(f"    source_id: {row['source_id']}")
                conn.close()
                
    except Exception as e:
        logger.error(f"검색 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    debug_search_results()

