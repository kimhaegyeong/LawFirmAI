# -*- coding: utf-8 -*-
"""
FAISS 인덱스와 chunk_id 매핑 문제 조사 스크립트
"""

import sys
import os
from pathlib import Path
import sqlite3
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def investigate_faiss_mapping():
    """FAISS 인덱스와 chunk_id 매핑 조사"""
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
    
    # FAISS 인덱스 상태 확인
    logger.info("\n" + "="*80)
    logger.info("FAISS 인덱스 상태 확인")
    logger.info("="*80)
    logger.info(f"  index is None: {search_engine.index is None}")
    if search_engine.index:
        logger.info(f"  index.ntotal: {search_engine.index.ntotal}")
    logger.info(f"  _chunk_ids 길이: {len(search_engine._chunk_ids) if hasattr(search_engine, '_chunk_ids') and search_engine._chunk_ids else 0}")
    
    # _chunk_ids 샘플 확인
    if hasattr(search_engine, '_chunk_ids') and search_engine._chunk_ids:
        logger.info(f"  _chunk_ids 샘플 (처음 10개): {search_engine._chunk_ids[:10]}")
        logger.info(f"  _chunk_ids 샘플 (마지막 10개): {search_engine._chunk_ids[-10:]}")
    
    # 검색 실행하여 실제 chunk_id 확인
    logger.info("\n" + "="*80)
    logger.info("검색 실행 및 chunk_id 확인")
    logger.info("="*80)
    query = "임대차 보증금"
    try:
        results = search_engine.search(
            query=query,
            k=5,
            similarity_threshold=0.3
        )
        
        logger.info(f"검색 결과: {len(results)}개")
        
        for i, result in enumerate(results[:5], 1):
            result_id = result.get('id', '')
            chunk_id_meta = result.get('metadata', {}).get('chunk_id')
            embedding_version_id = result.get('embedding_version_id')
            
            logger.info(f"\n  결과 {i}:")
            logger.info(f"    id: {result_id}")
            logger.info(f"    metadata.chunk_id: {chunk_id_meta}")
            logger.info(f"    embedding_version_id: {embedding_version_id}")
            
            # id에서 chunk_id 추출
            if result_id.startswith('chunk_'):
                try:
                    extracted_id = int(result_id.replace('chunk_', ''))
                    logger.info(f"    추출된 chunk_id (id에서): {extracted_id}")
                    
                    # _chunk_ids에서 확인
                    if hasattr(search_engine, '_chunk_ids') and search_engine._chunk_ids:
                        if extracted_id < len(search_engine._chunk_ids):
                            mapped_chunk_id = search_engine._chunk_ids[extracted_id]
                            logger.info(f"    _chunk_ids[{extracted_id}] = {mapped_chunk_id}")
                            
                            # 실제 DB에서 확인
                            conn = sqlite3.connect(str(db_path))
                            conn.row_factory = sqlite3.Row
                            cursor = conn.execute(
                                "SELECT id, source_type, source_id, embedding_version_id FROM text_chunks WHERE id = ?",
                                (mapped_chunk_id,)
                            )
                            row = cursor.fetchone()
                            if row:
                                logger.info(f"    ✅ 매핑된 chunk_id={mapped_chunk_id}는 DB에 존재")
                            else:
                                logger.warning(f"    ❌ 매핑된 chunk_id={mapped_chunk_id}는 DB에 없음")
                            conn.close()
                        else:
                            logger.warning(f"    ⚠️  extracted_id={extracted_id}가 _chunk_ids 길이({len(search_engine._chunk_ids)})를 초과")
                except ValueError:
                    logger.warning(f"    ⚠️  id에서 chunk_id 추출 실패: {result_id}")
                    
    except Exception as e:
        logger.error(f"검색 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    investigate_faiss_mapping()

