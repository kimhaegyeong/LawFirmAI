#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""특정 embedding version에 대한 FAISS 인덱스 빌드 스크립트"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_faiss_index_for_version(version_id: int, db_path: str = "data/lawfirm_v2.db"):
    """
    특정 embedding version에 대한 FAISS 인덱스 빌드
    
    Args:
        version_id: Embedding version ID
        db_path: 데이터베이스 경로
    """
    logger.info("="*80)
    logger.info(f"FAISS 인덱스 빌드 시작 (embedding_version_id={version_id})")
    logger.info("="*80)
    
    # SemanticSearchEngineV2 초기화
    try:
        engine = SemanticSearchEngineV2(
            db_path=db_path,
            use_external_index=False  # 내부 인덱스 빌드
        )
        logger.info("✅ 검색 엔진 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 검색 엔진 초기화 실패: {e}")
        return False
    
    # FAISS 인덱스 빌드
    try:
        logger.info(f"\n📦 Embedding version {version_id}의 임베딩 벡터를 로드하여 FAISS 인덱스 빌드 중...")
        success = engine._build_faiss_index_sync(embedding_version_id=version_id)
        
        if success:
            logger.info("\n✅ FAISS 인덱스 빌드 완료!")
            logger.info(f"   인덱스 경로: {engine.index_path}")
            if engine.index and hasattr(engine.index, 'ntotal'):
                logger.info(f"   인덱스 크기: {engine.index.ntotal:,}개 벡터")
            return True
        else:
            logger.error("\n❌ FAISS 인덱스 빌드 실패")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ FAISS 인덱스 빌드 중 오류 발생: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="특정 embedding version에 대한 FAISS 인덱스 빌드")
    parser.add_argument("--version-id", type=int, required=True, help="Embedding version ID")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    
    args = parser.parse_args()
    
    success = build_faiss_index_for_version(args.version_id, args.db)
    sys.exit(0 if success else 1)

