"""
통합 버전 관리 인터페이스

SQLite 버전 관리와 FAISS 버전 관리를 통합하여 제공합니다.
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from embedding_version_manager import EmbeddingVersionManager
from faiss_version_manager import FAISSVersionManager

logger = logging.getLogger(__name__)


class IntegratedVersionManager:
    """통합 버전 관리 클래스"""
    
    def __init__(self, db_path: str, vector_store_base: str = "data/vector_store"):
        """
        초기화
        
        Args:
            db_path: 데이터베이스 경로
            vector_store_base: 벡터 스토어 기본 경로
        """
        self.embedding_version_manager = EmbeddingVersionManager(db_path)
        self.faiss_version_manager = FAISSVersionManager(vector_store_base)
        self.db_path = db_path
    
    def create_version(
        self,
        version_name: str,
        chunking_strategy: str,
        model_name: str,
        chunking_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        description: Optional[str] = None,
        set_active: bool = False
    ) -> Dict[str, Any]:
        """
        새 버전 생성 (SQLite + FAISS)
        
        Args:
            version_name: 버전 이름
            chunking_strategy: 청킹 전략
            model_name: 임베딩 모델명
            chunking_config: 청킹 설정
            embedding_config: 임베딩 설정
            description: 버전 설명
            set_active: 활성 버전으로 설정할지 여부
        
        Returns:
            Dict: 생성된 버전 정보 (embedding_version_id, faiss_version_name)
        """
        metadata = {
            'chunking_config': chunking_config,
            'embedding_config': embedding_config
        }
        
        embedding_version_id = self.embedding_version_manager.register_version(
            version_name=version_name,
            chunking_strategy=chunking_strategy,
            model_name=model_name,
            description=description,
            metadata=metadata,
            set_active=set_active,
            create_faiss_version=True,
            faiss_version_manager=self.faiss_version_manager
        )
        
        faiss_version_name = f"{version_name}-{chunking_strategy}"
        
        return {
            'embedding_version_id': embedding_version_id,
            'faiss_version_name': faiss_version_name,
            'version_name': version_name,
            'chunking_strategy': chunking_strategy
        }
    
    def switch_version(
        self,
        embedding_version_id: Optional[int] = None,
        faiss_version_name: Optional[str] = None
    ) -> bool:
        """
        버전 전환
        
        Args:
            embedding_version_id: 임베딩 버전 ID
            faiss_version_name: FAISS 버전 이름
        
        Returns:
            bool: 성공 여부
        """
        success = True
        
        if embedding_version_id is not None:
            if not self.embedding_version_manager.set_active_version(embedding_version_id):
                success = False
        
        if faiss_version_name is not None:
            if not self.faiss_version_manager.set_active_version(faiss_version_name):
                success = False
        
        return success
    
    def delete_version(
        self,
        embedding_version_id: Optional[int] = None,
        faiss_version_name: Optional[str] = None,
        delete_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        버전 삭제
        
        Args:
            embedding_version_id: 임베딩 버전 ID
            faiss_version_name: FAISS 버전 이름
            delete_chunks: 청크 및 임베딩도 함께 삭제할지 여부
        
        Returns:
            Dict: 삭제 결과
        """
        result = {
            'embedding_deleted': False,
            'faiss_deleted': False,
            'chunks_deleted': 0,
            'embeddings_deleted': 0
        }
        
        if embedding_version_id is not None:
            chunks, embeddings = self.embedding_version_manager.delete_version(
                embedding_version_id,
                delete_chunks=delete_chunks
            )
            result['embedding_deleted'] = True
            result['chunks_deleted'] = chunks
            result['embeddings_deleted'] = embeddings
        
        if faiss_version_name is not None:
            if self.faiss_version_manager.delete_version(faiss_version_name, force=True):
                result['faiss_deleted'] = True
        
        return result
    
    def list_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        모든 버전 목록 조회
        
        Returns:
            Dict: embedding_versions, faiss_versions
        """
        embedding_versions = self.embedding_version_manager.list_versions()
        faiss_versions = self.faiss_version_manager.list_versions()
        
        return {
            'embedding_versions': embedding_versions,
            'faiss_versions': faiss_versions
        }
    
    def get_version_info(
        self,
        embedding_version_id: Optional[int] = None,
        faiss_version_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        버전 정보 조회
        
        Args:
            embedding_version_id: 임베딩 버전 ID
            faiss_version_name: FAISS 버전 이름
        
        Returns:
            Dict: 버전 정보
        """
        result = {}
        
        if embedding_version_id is not None:
            stats = self.embedding_version_manager.get_version_statistics(embedding_version_id)
            result['embedding_version'] = stats
        
        if faiss_version_name is not None:
            info = self.faiss_version_manager.get_version_info(faiss_version_name)
            result['faiss_version'] = info
        
        return result

