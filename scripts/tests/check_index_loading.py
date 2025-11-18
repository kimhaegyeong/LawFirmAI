#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAISS 인덱스 로드 상태 확인"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_index_loading():
    """인덱스 로드 상태 확인"""
    logger.info("="*80)
    logger.info("FAISS 인덱스 로드 상태 확인")
    logger.info("="*80)
    
    engine = SemanticSearchEngineV2(
        db_path="data/lawfirm_v2.db",
        use_external_index=False
    )
    
    logger.info(f"\n인덱스 정보:")
    logger.info(f"  index_path: {engine.index_path}")
    logger.info(f"  index 존재: {Path(engine.index_path).exists()}")
    logger.info(f"  index 로드됨: {engine.index is not None}")
    
    if engine.index:
        logger.info(f"  index.ntotal: {engine.index.ntotal:,}")
    
    logger.info(f"  _chunk_ids 길이: {len(engine._chunk_ids) if hasattr(engine, '_chunk_ids') and engine._chunk_ids else 0}")
    
    # chunk_ids.json 확인
    chunk_ids_path = Path(engine.index_path).with_suffix('.chunk_ids.json')
    logger.info(f"\nchunk_ids.json:")
    logger.info(f"  경로: {chunk_ids_path}")
    logger.info(f"  존재: {chunk_ids_path.exists()}")
    
    if chunk_ids_path.exists():
        import json
        with open(chunk_ids_path, 'r', encoding='utf-8') as f:
            chunk_ids = json.load(f)
        logger.info(f"  chunk_ids 개수: {len(chunk_ids)}")
        logger.info(f"  샘플: {chunk_ids[:5]}")
        
        # FAISS 인덱스 크기와 비교
        if engine.index:
            if len(chunk_ids) == engine.index.ntotal:
                logger.info(f"  ✅ chunk_ids 길이와 FAISS 인덱스 크기 일치")
            else:
                logger.warning(f"  ⚠️  chunk_ids 길이({len(chunk_ids)}) != FAISS 인덱스 크기({engine.index.ntotal})")
    
    # _chunk_ids와 chunk_ids.json 비교
    if hasattr(engine, '_chunk_ids') and engine._chunk_ids and chunk_ids_path.exists():
        import json
        with open(chunk_ids_path, 'r', encoding='utf-8') as f:
            saved_chunk_ids = json.load(f)
        
        if len(engine._chunk_ids) == len(saved_chunk_ids):
            if engine._chunk_ids[:5] == saved_chunk_ids[:5]:
                logger.info(f"  ✅ _chunk_ids와 chunk_ids.json 일치")
            else:
                logger.warning(f"  ⚠️  _chunk_ids와 chunk_ids.json 내용 불일치")
                logger.info(f"    _chunk_ids 샘플: {engine._chunk_ids[:5]}")
                logger.info(f"    chunk_ids.json 샘플: {saved_chunk_ids[:5]}")
        else:
            logger.warning(f"  ⚠️  _chunk_ids 길이({len(engine._chunk_ids)}) != chunk_ids.json 길이({len(saved_chunk_ids)})")


if __name__ == "__main__":
    check_index_loading()

