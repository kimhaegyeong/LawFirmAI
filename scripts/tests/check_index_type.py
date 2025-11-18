#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현재 로드된 FAISS 인덱스 타입 확인"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_index_type(index_path: str):
    """인덱스 타입 확인"""
    if not FAISS_AVAILABLE:
        logger.error("FAISS not available")
        return
    
    if not Path(index_path).exists():
        logger.error(f"Index file not found: {index_path}")
        return
    
    try:
        index = faiss.read_index(str(index_path))
        index_type = type(index).__name__
        
        logger.info("="*80)
        logger.info("FAISS 인덱스 정보")
        logger.info("="*80)
        logger.info(f"인덱스 경로: {index_path}")
        logger.info(f"인덱스 타입: {index_type}")
        logger.info(f"벡터 개수: {index.ntotal:,}")
        logger.info(f"벡터 차원: {index.d}")
        
        # IndexIVF 계열 확인
        if hasattr(index, 'nprobe'):
            logger.info(f"nprobe: {index.nprobe}")
            if hasattr(index, 'nlist'):
                logger.info(f"nlist: {index.nlist}")
        
        # IndexIVFPQ 확인
        if 'IndexIVFPQ' in index_type:
            logger.info("✅ IndexIVFPQ 인덱스 감지됨")
            if hasattr(index, 'pq'):
                m = index.pq.M if hasattr(index.pq, 'M') else 'unknown'
                nbits = index.pq.nbits if hasattr(index.pq, 'nbits') else 'unknown'
                logger.info(f"   PQ parameters: M={m}, nbits={nbits}")
        elif 'IndexIVF' in index_type:
            logger.info(f"ℹ️  IndexIVF 계열 인덱스 (IndexIVFPQ 아님)")
        else:
            logger.info(f"ℹ️  다른 타입의 인덱스")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to load index: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS 인덱스 타입 확인")
    parser.add_argument("--index", default="data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.faiss", help="인덱스 파일 경로")
    
    args = parser.parse_args()
    
    check_index_type(args.index)

