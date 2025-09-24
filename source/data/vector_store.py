"""
Vector Store Management
벡터 저장소 관리 모듈
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, store_path: str, dimension: int = 768):
        """벡터 저장소 초기화"""
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.embeddings = []
        self.metadata = []
        self._load_store()
        logger.info(f"VectorStore initialized with dimension: {dimension}")
    
    def _load_store(self):
        """저장소 로드"""
        try:
            embeddings_file = self.store_path / "embeddings.pkl"
            metadata_file = self.store_path / "metadata.json"
            
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
                
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            self.embeddings = []
            self.metadata = []
    
    def _save_store(self):
        """저장소 저장"""
        try:
            embeddings_file = self.store_path / "embeddings.pkl"
            metadata_file = self.store_path / "metadata.json"
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """임베딩 추가"""
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
        
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # 자동 저장
        self._save_store()
        
        return len(self.embeddings) - 1
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """유사한 임베딩 검색"""
        if not self.embeddings:
            return []
        
        # 코사인 유사도 계산
        embeddings_array = np.array(self.embeddings)
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(embeddings_array, axis=1)
        
        similarities = np.dot(embeddings_array, query_embedding) / (query_norm * embeddings_norm)
        
        # 상위 k개 인덱스 반환
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            metadata = self.metadata[idx]
            results.append((idx, similarity, metadata))
        
        return results
    
    def get_embedding(self, index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """인덱스로 임베딩 조회"""
        if 0 <= index < len(self.embeddings):
            return self.embeddings[index], self.metadata[index]
        return None
    
    def get_all_embeddings(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """모든 임베딩 조회"""
        return list(zip(self.embeddings, self.metadata))
    
    def clear_store(self):
        """저장소 초기화"""
        self.embeddings = []
        self.metadata = []
        self._save_store()
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        return {
            "total_embeddings": len(self.embeddings),
            "dimension": self.dimension,
            "store_path": str(self.store_path)
        }
