#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IndexIVFPQ 인덱스 생성 스크립트"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from scripts.utils.embedding_version_manager import EmbeddingVersionManager
from scripts.utils.faiss_version_manager import FAISSVersionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_indexivfpq(
    db_path: str,
    vector_store_path: str,
    embedding_version_id: int,
    output_path: Optional[str] = None,
    m: int = 64,
    nbits: int = 8,
    nlist: Optional[int] = None
):
    """
    IndexIVFPQ 인덱스 생성
    
    Args:
        db_path: 데이터베이스 경로
        vector_store_path: 벡터 스토어 경로
        embedding_version_id: 임베딩 버전 ID
        output_path: 출력 인덱스 파일 경로 (선택사항)
        m: Product Quantization의 서브벡터 개수 (기본값: 64)
        nbits: 각 서브벡터의 비트 수 (기본값: 8)
        nlist: 클러스터 수 (선택사항, 자동 계산)
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        logger.error("FAISS or NumPy not available")
        return False
    
    logger.info("="*80)
    logger.info("IndexIVFPQ 인덱스 생성")
    logger.info("="*80)
    
    # SemanticSearchEngineV2 인스턴스 생성
    engine = SemanticSearchEngineV2(
        db_path=db_path,
        use_external_index=False
    )
    
    # EmbeddingVersionManager를 통해 버전 정보 가져오기
    evm = EmbeddingVersionManager(db_path)
    version_info = evm.get_version_statistics(embedding_version_id)
    
    if not version_info:
        logger.error(f"❌ 버전 ID {embedding_version_id}를 찾을 수 없습니다.")
        return False
    
    logger.info(f"📦 Embedding version {embedding_version_id}의 임베딩 벡터를 로드하여 IndexIVFPQ 인덱스 빌드 중...")
    
    # 벡터 로드
    chunk_vectors = engine._load_chunk_vectors(embedding_version_id=embedding_version_id)
    if not chunk_vectors:
        logger.error("❌ 임베딩 벡터를 로드할 수 없습니다.")
        return False
    
    logger.info(f"✅ {len(chunk_vectors)}개의 벡터 로드 완료")
    
    # numpy 배열 생성
    chunk_ids_sorted = sorted(chunk_vectors.keys())
    vectors = np.array([
        chunk_vectors[chunk_id]
        for chunk_id in chunk_ids_sorted
    ]).astype('float32')
    
    dimension = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    logger.info(f"벡터 차원: {dimension}, 벡터 개수: {num_vectors:,}")
    
    # nlist 자동 계산 (지정되지 않은 경우)
    if nlist is None:
        nlist = min(1000, max(100, num_vectors // 100))
        logger.info(f"nlist 자동 계산: {nlist}")
    
    # Product Quantization 파라미터 검증
    if dimension % m != 0:
        logger.warning(f"⚠️  벡터 차원({dimension})이 m({m})으로 나누어떨어지지 않습니다. m을 조정합니다.")
        # m을 dimension의 약수로 조정
        for candidate_m in [32, 48, 64, 96, 128]:
            if dimension % candidate_m == 0:
                m = candidate_m
                logger.info(f"m을 {m}으로 조정했습니다.")
                break
        else:
            logger.error(f"❌ 벡터 차원({dimension})에 적합한 m 값을 찾을 수 없습니다.")
            return False
    
    logger.info(f"IndexIVFPQ 파라미터: nlist={nlist}, m={m}, nbits={nbits}")
    
    # IndexIVFPQ 인덱스 생성
    logger.info("IndexIVFPQ 인덱스 생성 중...")
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
    
    # 학습
    logger.info(f"IndexIVFPQ 인덱스 학습 중... (nlist={nlist})")
    index.train(vectors)
    
    # 벡터 추가
    logger.info(f"벡터 추가 중... ({num_vectors:,}개)")
    index.add(vectors)
    
    # nprobe 설정
    optimal_nprobe = engine._calculate_optimal_nprobe(10, num_vectors)
    index.nprobe = optimal_nprobe
    logger.info(f"nprobe 설정: {optimal_nprobe}")
    
    # 출력 경로 결정
    if output_path is None:
        version_name = version_info.get('version_name', f'v{embedding_version_id}')
        chunking_strategy = version_info.get('chunking_strategy', 'standard')
        output_dir = Path(vector_store_path) / f"{version_name}-{chunking_strategy}-ivfpq"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "index.faiss")
    else:
        output_path = str(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 인덱스 저장
    logger.info(f"IndexIVFPQ 인덱스 저장 중: {output_path}")
    faiss.write_index(index, output_path)
    
    # chunk_ids.json 저장
    chunk_ids_path = Path(output_path).with_suffix('.chunk_ids.json')
    import json
    with open(chunk_ids_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_ids_sorted, f, indent=2)
    logger.info(f"chunk_ids 저장: {chunk_ids_path}")
    
    # 파일 크기 확인
    index_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✅ IndexIVFPQ 인덱스 생성 완료!")
    logger.info(f"   인덱스 파일: {output_path}")
    logger.info(f"   파일 크기: {index_size_mb:.2f} MB")
    logger.info(f"   벡터 개수: {num_vectors:,}")
    logger.info(f"   PQ 파라미터: m={m}, nbits={nbits}")
    logger.info(f"   nlist: {nlist}, nprobe: {optimal_nprobe}")
    
    # 메모리 사용량 비교 (예상)
    original_size_mb = (num_vectors * dimension * 4) / (1024 * 1024)  # float32
    compressed_size_mb = (num_vectors * m * nbits / 8) / (1024 * 1024)  # PQ 압축
    compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
    logger.info(f"   예상 메모리 절약: {compression_ratio:.2f}x (원본: {original_size_mb:.2f} MB → 압축: {compressed_size_mb:.2f} MB)")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexIVFPQ 인덱스 생성")
    parser.add_argument("--db", default="data/lawfirm_v2.db", help="데이터베이스 경로")
    parser.add_argument("--vector-store", default="data/vector_store", help="벡터 스토어 경로")
    parser.add_argument("--version-id", type=int, required=True, help="임베딩 버전 ID")
    parser.add_argument("--output", type=str, default=None, help="출력 인덱스 파일 경로 (선택사항)")
    parser.add_argument("--m", type=int, default=64, help="Product Quantization 서브벡터 개수 (기본값: 64)")
    parser.add_argument("--nbits", type=int, default=8, help="각 서브벡터의 비트 수 (기본값: 8)")
    parser.add_argument("--nlist", type=int, default=None, help="클러스터 수 (선택사항, 자동 계산)")
    
    args = parser.parse_args()
    
    success = build_indexivfpq(
        db_path=args.db,
        vector_store_path=args.vector_store,
        embedding_version_id=args.version_id,
        output_path=args.output,
        m=args.m,
        nbits=args.nbits,
        nlist=args.nlist
    )
    
    sys.exit(0 if success else 1)

