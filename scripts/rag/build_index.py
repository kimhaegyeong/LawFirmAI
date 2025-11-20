"""
FAISS 인덱스 빌드 및 MLflow 저장

기존 SemanticSearchEngineV2를 활용하여 FAISS 인덱스를 빌드하고,
로컬 파일 저장 및 MLflow 저장을 수행합니다.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "utils"))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from faiss_version_manager import FAISSVersionManager
from mlflow_manager import MLflowFAISSManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_and_save_index(
    version_name: str,
    embedding_version_id: int,
    chunking_strategy: str,
    db_path: str = "data/lawfirm_v2.db",
    use_mlflow: bool = True,
    local_backup: bool = True,
    model_name: str = "jhgan/ko-sroberta-multitask"
) -> bool:
    """
    FAISS 인덱스 빌드 및 저장 (로컬 + MLflow)
    
    Args:
        version_name: 버전 이름
        embedding_version_id: 임베딩 버전 ID
        chunking_strategy: 청킹 전략
        db_path: 데이터베이스 경로
        use_mlflow: MLflow 사용 여부
        local_backup: 로컬 백업 저장 여부
        model_name: 임베딩 모델명
    
    Returns:
        bool: 성공 여부
    """
    try:
        logger.info(f"Building FAISS index for version: {version_name}")
        logger.info(f"Embedding version ID: {embedding_version_id}")
        logger.info(f"Chunking strategy: {chunking_strategy}")
        
        # 인덱스 빌드 시에는 MLflow 인덱스 로드를 건너뛰기 위해 use_mlflow_index=False 설정
        # 환경 변수가 설정되어 있어도 명시적으로 False로 설정
        import os
        original_use_mlflow = os.environ.get("USE_MLFLOW_INDEX")
        try:
            os.environ["USE_MLFLOW_INDEX"] = "false"
            search_engine = SemanticSearchEngineV2(
                db_path=db_path,
                model_name=model_name,
                use_mlflow_index=False
            )
        finally:
            # 환경 변수 복원
            if original_use_mlflow is not None:
                os.environ["USE_MLFLOW_INDEX"] = original_use_mlflow
            elif "USE_MLFLOW_INDEX" in os.environ:
                del os.environ["USE_MLFLOW_INDEX"]
        
        logger.info("Loading chunk vectors...")
        chunk_vectors = search_engine._load_chunk_vectors(embedding_version_id=embedding_version_id)
        if not chunk_vectors:
            logger.error("No chunk vectors found")
            return False
        
        chunk_ids_sorted = sorted(chunk_vectors.keys())
        
        # 개선 1: 인덱스 빌드 전 데이터베이스 일관성 검증
        logger.info("Validating database consistency...")
        import sqlite3
        missing_chunk_ids = []
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for chunk_id in chunk_ids_sorted:
                cursor = conn.execute("SELECT id FROM text_chunks WHERE id = ?", (chunk_id,))
                if not cursor.fetchone():
                    missing_chunk_ids.append(chunk_id)
        
        if missing_chunk_ids:
            logger.warning(f"⚠️  Found {len(missing_chunk_ids)} chunk_ids in vectors but not in database")
            logger.warning(f"   First 10 missing chunk_ids: {missing_chunk_ids[:10]}")
            logger.warning("   These chunks will be excluded from the index")
            # 누락된 chunk_id 제외
            chunk_ids_sorted = [cid for cid in chunk_ids_sorted if cid not in missing_chunk_ids]
            chunk_vectors = {cid: chunk_vectors[cid] for cid in chunk_ids_sorted}
            logger.info(f"   After filtering: {len(chunk_ids_sorted)} valid chunks")
        else:
            logger.info("✅ All chunk_ids found in database")
        import numpy as np
        vectors = np.array([
            chunk_vectors[chunk_id]
            for chunk_id in chunk_ids_sorted
        ]).astype('float32')
        
        if len(vectors) == 0:
            logger.error("No vectors to index")
            return False
        
        logger.info(f"Building index with {len(vectors)} vectors")
        
        import faiss
        dimension = vectors.shape[1]
        nlist = min(100, max(10, len(vectors) // 10))
        
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        logger.info(f"Training FAISS index with nlist={nlist}")
        index.train(vectors)
        index.add(vectors)
        index.nprobe = search_engine._calculate_optimal_nprobe(10, len(vectors))
        
        chunk_ids = chunk_ids_sorted
        id_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        
        chunk_metadata = {}
        if hasattr(search_engine, '_chunk_metadata'):
            chunk_metadata = search_engine._chunk_metadata
        metadata = [chunk_metadata.get(chunk_id, {}) for chunk_id in chunk_ids]
        
        logger.info(f"Index built successfully: {index.ntotal} vectors, dimension={dimension}")
        
        mlflow_run_id = None
        local_index_path = None
        
        if local_backup:
            logger.info("Saving to local file system...")
            local_manager = FAISSVersionManager(base_path="data/vector_store")
            
            version_path = local_manager.get_version_path(version_name)
            if version_path is None:
                try:
                    scripts_utils_path = project_root / "scripts" / "utils"
                    if scripts_utils_path.exists():
                        sys.path.insert(0, str(scripts_utils_path))
                    from embedding_version_manager import EmbeddingVersionManager
                    
                    evm = EmbeddingVersionManager(db_path)
                    version_info = evm.get_version_statistics(embedding_version_id)
                    
                    if version_info:
                        chunking_config = {
                            'chunk_size': version_info.get('chunk_size', 1000),
                            'chunk_overlap': version_info.get('chunk_overlap', 200)
                        }
                        embedding_config = {
                            'model': model_name,
                            'dimension': dimension
                        }
                        
                        version_path = local_manager.create_version(
                            version_name=version_name,
                            embedding_version_id=embedding_version_id,
                            chunking_strategy=chunking_strategy,
                            chunking_config=chunking_config,
                            embedding_config=embedding_config,
                            document_count=version_info.get('document_count', 0),
                            total_chunks=len(chunk_ids),
                            status='active'
                        )
                except Exception as e:
                    logger.warning(f"Failed to get version info: {e}")
                    chunking_config = {'chunk_size': 1000, 'chunk_overlap': 200}
                    embedding_config = {'model': model_name, 'dimension': dimension}
                    version_path = local_manager.create_version(
                        version_name=version_name,
                        embedding_version_id=embedding_version_id,
                        chunking_strategy=chunking_strategy,
                        chunking_config=chunking_config,
                        embedding_config=embedding_config,
                        document_count=0,
                        total_chunks=len(chunk_ids),
                        status='active'
                    )
            
            if version_path:
                local_index_path = version_path / "index.faiss"
                success = local_manager.save_index(
                    version_name=version_name,
                    index=index,
                    id_mapping=id_mapping,
                    metadata=metadata
                )
                if success:
                    logger.info(f"Saved to local: {local_index_path}")
                else:
                    logger.warning("Failed to save to local")
        
        if use_mlflow:
            try:
                logger.info("Saving to MLflow...")
                mlflow_manager = MLflowFAISSManager()
                
                try:
                    scripts_utils_path = project_root / "scripts" / "utils"
                    if scripts_utils_path.exists():
                        sys.path.insert(0, str(scripts_utils_path))
                    from embedding_version_manager import EmbeddingVersionManager
                    
                    evm = EmbeddingVersionManager(db_path)
                    version_info = evm.get_version_statistics(embedding_version_id)
                    
                    chunking_config = version_info.get('chunking_config', {}) if version_info else {'chunk_size': 1000, 'chunk_overlap': 200}
                    if not chunking_config:
                        chunking_config = {'chunk_size': 1000, 'chunk_overlap': 200}
                    
                    embedding_config = version_info.get('embedding_config', {}) if version_info else {}
                    if not embedding_config or not embedding_config.get('model'):
                        model_name_from_version = version_info.get('model_name', model_name) if version_info else model_name
                        embedding_config = {
                            'model': model_name_from_version,
                            'dimension': dimension
                        }
                        logger.info(f"Created embedding_config: {embedding_config}")
                    elif not embedding_config.get('dimension'):
                        embedding_config['dimension'] = dimension
                        logger.info(f"Updated embedding_config with dimension: {embedding_config}")
                    
                    document_count = version_info.get('document_count', 0) if version_info else 0
                except Exception as e:
                    logger.warning(f"Failed to get version info for MLflow: {e}")
                    chunking_config = {'chunk_size': 1000, 'chunk_overlap': 200}
                    embedding_config = {'model': model_name, 'dimension': dimension}
                    document_count = 0
                
                mlflow_run_id = mlflow_manager.create_run(
                    version_name=version_name,
                    embedding_version_id=embedding_version_id,
                    chunking_strategy=chunking_strategy,
                    chunking_config=chunking_config,
                    embedding_config=embedding_config,
                    document_count=document_count,
                    total_chunks=len(chunk_ids),
                    status='active'
                )
                
                success = mlflow_manager.save_index(
                    run_id=mlflow_run_id,
                    index=index,
                    id_mapping=id_mapping,
                    metadata=metadata,
                    index_path=local_index_path
                )
                
                if success:
                    logger.info(f"Saved to MLflow: run_id={mlflow_run_id}")
                else:
                    logger.warning("Failed to save to MLflow")
                    
            except Exception as e:
                logger.error(f"Failed to save to MLflow: {e}")
                if not local_backup:
                    raise
        
        logger.info("Index build and save completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build and save index: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and save to local/MLflow")
    parser.add_argument("--version-name", required=True, help="Version name")
    parser.add_argument("--embedding-version-id", type=int, required=True, help="Embedding version ID")
    parser.add_argument("--chunking-strategy", required=True, help="Chunking strategy")
    parser.add_argument("--db-path", default="data/lawfirm_v2.db", help="Database path")
    parser.add_argument("--use-mlflow", action="store_true", default=True, help="Use MLflow")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false", help="Don't use MLflow")
    parser.add_argument("--local-backup", action="store_true", default=True, help="Save to local file system")
    parser.add_argument("--no-local-backup", dest="local_backup", action="store_false", help="Don't save to local")
    parser.add_argument("--model-name", default="jhgan/ko-sroberta-multitask", help="Embedding model name")
    
    args = parser.parse_args()
    
    success = build_and_save_index(
        version_name=args.version_name,
        embedding_version_id=args.embedding_version_id,
        chunking_strategy=args.chunking_strategy,
        db_path=args.db_path,
        use_mlflow=args.use_mlflow,
        local_backup=args.local_backup,
        model_name=args.model_name
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

