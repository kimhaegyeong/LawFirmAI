# -*- coding: utf-8 -*-
"""
메타데이터 복원 검증 스크립트
- 검색 결과에 메타데이터가 포함되어 있는지 확인
"""

import sys
import os
import sqlite3
import json
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_metadata_restoration():
    """메타데이터 복원 검증"""
    logger.info("="*80)
    logger.info("메타데이터 복원 검증 테스트")
    logger.info("="*80)
    
    # 환경 변수 설정 (IndexIVFPQ 인덱스 사용)
    if not os.getenv('USE_EXTERNAL_VECTOR_STORE'):
        os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
    
    if not os.getenv('EXTERNAL_VECTOR_STORE_BASE_PATH'):
        possible_paths = [
            "data/vector_store/v2.0.0-dynamic-dynamic-ivfpq",
            "./data/vector_store/v2.0.0-dynamic-dynamic-ivfpq",
            str(project_root / "data" / "vector_store" / "v2.0.0-dynamic-dynamic-ivfpq"),
        ]
        detected_path = None
        for p in possible_paths:
            if Path(p).exists():
                detected_path = p
                break
        
        if detected_path:
            os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = detected_path
            logger.info(f"✅ IndexIVFPQ 인덱스 경로: {detected_path}")
        else:
            logger.warning("⚠️  IndexIVFPQ 인덱스 경로를 찾을 수 없습니다.")
    
    # SemanticSearchEngineV2 초기화
    config = Config()
    search_engine = SemanticSearchEngineV2(
        db_path=config.database_path,
        use_external_index=config.use_external_vector_store,
        external_index_path=config.external_vector_store_base_path
    )
    
    # 검색 쿼리
    query = "임대차 보증금 반환"
    logger.info(f"\n📝 검색 쿼리: {query}")
    
    # 검색 실행
    results = search_engine.search(query, k=10, similarity_threshold=0.2)
    
    logger.info(f"\n✅ 검색 결과: {len(results)}개")
    
    if not results:
        logger.warning("⚠️  검색 결과가 없습니다.")
        return
    
    # 메타데이터 검증
    logger.info("\n" + "="*80)
    logger.info("검색 결과 메타데이터 검증")
    logger.info("="*80)
    
    metadata_stats = {
        'has_doc_id': 0,
        'has_casenames': 0,
        'has_court': 0,
        'has_org': 0,
        'has_all_metadata': 0,
        'missing_metadata': 0,
        'text_too_short': 0
    }
    
    for i, result in enumerate(results[:10], 1):
        logger.info(f"\n--- 결과 {i} ---")
        # chunk_id는 metadata 안에 있음
        metadata = result.get('metadata', {})
        chunk_id = metadata.get('chunk_id') or result.get('chunk_id')
        source_type = metadata.get('source_type') or result.get('type') or result.get('source_type')
        logger.info(f"  chunk_id: {chunk_id}")
        logger.info(f"  source_type: {source_type}")
        logger.info(f"  score: {result.get('score', 0):.4f}")
        
        # 메타데이터 확인 (최상위 레벨과 metadata 모두 확인)
        doc_id = result.get('doc_id') or metadata.get('doc_id')
        casenames = result.get('casenames') or metadata.get('casenames')
        court = result.get('court') or metadata.get('court')
        org = result.get('org') or metadata.get('org')
        
        if doc_id:
            metadata_stats['has_doc_id'] += 1
            logger.info(f"  ✅ doc_id: {doc_id}")
        else:
            logger.warning(f"  ⚠️  doc_id: 없음")
        
        # source_type에 따라 필요한 메타데이터 확인
        if source_type == 'case_paragraph':
            if casenames:
                metadata_stats['has_casenames'] += 1
                logger.info(f"  ✅ casenames: {casenames[:50]}...")
            else:
                logger.warning(f"  ⚠️  casenames: 없음")
            
            if court:
                metadata_stats['has_court'] += 1
                logger.info(f"  ✅ court: {court}")
            else:
                logger.warning(f"  ⚠️  court: 없음")
            
            # case_paragraph는 doc_id, casenames, court 모두 필요
            if doc_id and casenames and court:
                metadata_stats['has_all_metadata'] += 1
            else:
                metadata_stats['missing_metadata'] += 1
        elif source_type == 'decision_paragraph':
            if org:
                metadata_stats['has_org'] = metadata_stats.get('has_org', 0) + 1
                logger.info(f"  ✅ org: {org}")
            else:
                logger.warning(f"  ⚠️  org: 없음")
            
            # decision_paragraph는 doc_id와 org 필요
            if doc_id and org:
                metadata_stats['has_all_metadata'] += 1
            else:
                metadata_stats['missing_metadata'] += 1
        else:
            # 다른 타입은 doc_id만 확인
            if doc_id:
                metadata_stats['has_all_metadata'] += 1
            else:
                metadata_stats['missing_metadata'] += 1
        
        # 텍스트 길이 확인
        text = result.get('text') or result.get('content') or ''
        if len(text) < 100:
            metadata_stats['text_too_short'] += 1
            logger.warning(f"  ⚠️  텍스트가 너무 짧음: {len(text)}자")
        else:
            logger.info(f"  ✅ 텍스트 길이: {len(text)}자")
    
    # 통계 출력
    logger.info("\n" + "="*80)
    logger.info("메타데이터 복원 통계")
    logger.info("="*80)
    logger.info(f"전체 결과: {len(results)}개")
    logger.info(f"doc_id 있음: {metadata_stats['has_doc_id']}개")
    logger.info(f"casenames 있음: {metadata_stats['has_casenames']}개")
    logger.info(f"court 있음: {metadata_stats['has_court']}개")
    logger.info(f"org 있음: {metadata_stats.get('has_org', 0)}개")
    logger.info(f"모든 메타데이터 있음: {metadata_stats['has_all_metadata']}개")
    logger.info(f"메타데이터 누락: {metadata_stats['missing_metadata']}개")
    logger.info(f"텍스트 너무 짧음: {metadata_stats['text_too_short']}개")
    
    # 데이터베이스에서 직접 확인
    logger.info("\n" + "="*80)
    logger.info("데이터베이스 직접 확인")
    logger.info("="*80)
    
    conn = sqlite3.connect(config.database_path)
    conn.row_factory = sqlite3.Row
    
    # 첫 번째 결과의 chunk_id 가져오기 (metadata 안에 있음)
    sample_chunk_id = None
    if results:
        sample_metadata = results[0].get('metadata', {})
        sample_chunk_id = sample_metadata.get('chunk_id') or results[0].get('chunk_id')
    
    if sample_chunk_id:
        cursor = conn.execute("""
            SELECT id, source_type, source_id, meta, LENGTH(text) as text_length
            FROM text_chunks
            WHERE id = ?
        """, (sample_chunk_id,))
        row = cursor.fetchone()
        
        if row:
            logger.info(f"청크 ID {sample_chunk_id}:")
            logger.info(f"  source_type: {row['source_type']}")
            logger.info(f"  source_id: {row['source_id']}")
            logger.info(f"  text_length: {row['text_length']}자")
            
            if row['meta']:
                try:
                    meta_json = json.loads(row['meta'])
                    logger.info(f"  ✅ meta 컬럼에 메타데이터 있음:")
                    logger.info(f"    doc_id: {meta_json.get('doc_id', '없음')}")
                    logger.info(f"    casenames: {meta_json.get('casenames', '없음')[:50]}...")
                    logger.info(f"    court: {meta_json.get('court', '없음')}")
                except Exception as e:
                    logger.warning(f"  ⚠️  meta JSON 파싱 실패: {e}")
            else:
                logger.warning(f"  ⚠️  meta 컬럼이 비어있음")
    
    conn.close()
    
    logger.info("\n" + "="*80)
    logger.info("검증 완료")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        test_metadata_restoration()
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        sys.exit(1)

