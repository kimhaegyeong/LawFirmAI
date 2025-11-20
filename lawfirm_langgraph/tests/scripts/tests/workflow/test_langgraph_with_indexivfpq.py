#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangGraph에서 IndexIVFPQ 인덱스 사용 테스트"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_langgraph_with_indexivfpq():
    """LangGraph에서 IndexIVFPQ 인덱스 사용 테스트"""
    logger.info("="*80)
    logger.info("LangGraph에서 IndexIVFPQ 인덱스 사용 테스트")
    logger.info("="*80)
    
    # LangGraph 설정
    config = LangGraphConfig.from_env()
    
    # IndexIVFPQ 인덱스 경로 설정
    indexivfpq_path = "data/vector_store/v2.0.0-dynamic-dynamic-ivfpq"
    
    # Config에 IndexIVFPQ 인덱스 경로 설정
    import os
    os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
    os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = indexivfpq_path
    
    # Config 재로드
    from lawfirm_langgraph.core.utils.config import Config
    config_obj = Config()
    
    logger.info(f"\n설정 정보:")
    logger.info(f"  USE_EXTERNAL_VECTOR_STORE: {config_obj.use_external_vector_store}")
    logger.info(f"  EXTERNAL_VECTOR_STORE_BASE_PATH: {config_obj.external_vector_store_base_path}")
    
    # EnhancedLegalQuestionWorkflow 초기화
    logger.info("\nEnhancedLegalQuestionWorkflow 초기화 중...")
    workflow = EnhancedLegalQuestionWorkflow(config)
    
    # SemanticSearchEngineV2 확인
    if hasattr(workflow, 'semantic_search') and workflow.semantic_search:
        logger.info("\n✅ SemanticSearchEngineV2 초기화 완료")
        
        if workflow.semantic_search.index:
            index_type = type(workflow.semantic_search.index).__name__
            logger.info(f"  인덱스 타입: {index_type}")
            logger.info(f"  인덱스 벡터 수: {workflow.semantic_search.index.ntotal:,}")
            
            if 'IndexIVFPQ' in index_type:
                logger.info(f"  ✅ IndexIVFPQ 인덱스 감지됨!")
                if hasattr(workflow.semantic_search.index, 'pq'):
                    m = workflow.semantic_search.index.pq.M if hasattr(workflow.semantic_search.index.pq, 'M') else 'unknown'
                    nbits = workflow.semantic_search.index.pq.nbits if hasattr(workflow.semantic_search.index.pq, 'nbits') else 'unknown'
                    logger.info(f"     PQ parameters: M={m}, nbits={nbits}")
        else:
            logger.warning("⚠️  인덱스가 로드되지 않았습니다")
    else:
        logger.warning("⚠️  SemanticSearchEngineV2가 초기화되지 않았습니다")
    
    # 검색 테스트
    test_query = "임대차 보증금 반환"
    logger.info(f"\n📝 검색 테스트 쿼리: {test_query}")
    logger.info("-" * 80)
    
    try:
        if hasattr(workflow, 'semantic_search') and workflow.semantic_search:
            results = workflow.semantic_search.search(
                query=test_query,
                k=5,
                similarity_threshold=0.0
            )
            
            logger.info(f"✅ 검색 결과: {len(results)}개")
            
            if results:
                for i, result in enumerate(results[:3], 1):
                    score = result.get('score', 0)
                    chunk_id = result.get('metadata', {}).get('chunk_id', 'N/A')
                    source_type = result.get('type', 'N/A')
                    text_preview = result.get('text', '')[:100] if result.get('text') else 'N/A'
                    
                    logger.info(f"  {i}. score={score:.4f}, chunk_id={chunk_id}, type={source_type}")
                    logger.info(f"     text: {text_preview}...")
            else:
                logger.warning("⚠️  검색 결과가 없습니다")
        else:
            logger.warning("⚠️  SemanticSearchEngineV2를 사용할 수 없습니다")
            
    except Exception as e:
        logger.error(f"❌ 검색 중 오류 발생: {e}", exc_info=True)
    
    logger.info("\n" + "="*80)
    logger.info("테스트 완료")
    logger.info("="*80)


if __name__ == "__main__":
    test_langgraph_with_indexivfpq()

