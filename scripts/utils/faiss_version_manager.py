"""
FAISS 기반 벡터 임베딩 버전 관리 시스템

파일 기반으로 FAISS 인덱스를 버전별로 관리합니다.
"""
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class FAISSVersionManager:
    """FAISS 인덱스 버전 관리 클래스"""
    
    def __init__(self, base_path: str = "data/vector_store"):
        """
        버전 관리자 초기화
        
        Args:
            base_path: 벡터 스토어 기본 경로
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.active_version_file = self.base_path / "active_version.txt"
    
    def create_version(
        self,
        version_name: str,
        embedding_version_id: int,
        chunking_strategy: str,
        chunking_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        document_count: int = 0,
        total_chunks: int = 0,
        status: str = "active"
    ) -> Path:
        """
        새 버전 디렉토리 및 메타데이터 생성
        
        Args:
            version_name: 버전 이름 (예: "v1.0.0-standard")
            embedding_version_id: EmbeddingVersionManager의 버전 ID
            chunking_strategy: 청킹 전략
            chunking_config: 청킹 설정
            embedding_config: 임베딩 설정
            document_count: 문서 수
            total_chunks: 총 청크 수
            status: 버전 상태 (active, inactive, experimental)
        
        Returns:
            Path: 생성된 버전 디렉토리 경로
        """
        version_path = self.base_path / version_name
        version_path.mkdir(parents=True, exist_ok=True)
        
        version_info = {
            "version": version_name,
            "embedding_version_id": embedding_version_id,
            "chunking_strategy": chunking_strategy,
            "created_at": datetime.now().isoformat(),
            "chunking_config": chunking_config,
            "embedding_config": embedding_config,
            "document_count": document_count,
            "total_chunks": total_chunks,
            "status": status
        }
        
        version_info_path = version_path / "version_info.json"
        with open(version_info_path, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created FAISS version: {version_name} at {version_path}")
        return version_path
    
    def set_active_version(self, version_name: str) -> bool:
        """
        활성 버전 설정
        
        Args:
            version_name: 버전 이름
        
        Returns:
            bool: 성공 여부
        """
        version_path = self.base_path / version_name
        if not version_path.exists():
            logger.error(f"Version {version_name} does not exist")
            return False
        
        with open(self.active_version_file, 'w', encoding='utf-8') as f:
            f.write(version_name)
        
        logger.info(f"Set active version: {version_name}")
        return True
    
    def get_active_version(self) -> Optional[str]:
        """
        현재 활성 버전 조회
        
        Returns:
            Optional[str]: 활성 버전 이름, 없으면 None
        """
        if not self.active_version_file.exists():
            return None
        
        try:
            with open(self.active_version_file, 'r', encoding='utf-8') as f:
                version_name = f.read().strip()
            
            version_path = self.base_path / version_name
            if version_path.exists():
                return version_name
            else:
                logger.warning(f"Active version {version_name} directory not found")
                return None
        except Exception as e:
            logger.error(f"Failed to read active version: {e}")
            return None
    
    def get_version_path(self, version_name: Optional[str] = None) -> Optional[Path]:
        """
        버전별 경로 조회
        
        Args:
            version_name: 버전 이름 (None이면 활성 버전)
        
        Returns:
            Optional[Path]: 버전 디렉토리 경로
        """
        if version_name is None:
            version_name = self.get_active_version()
            if version_name is None:
                return None
        
        version_path = self.base_path / version_name
        if version_path.exists():
            return version_path
        else:
            logger.warning(f"Version {version_name} not found")
            return None
    
    def get_version_info(self, version_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        버전 정보 조회
        
        Args:
            version_name: 버전 이름 (None이면 활성 버전)
        
        Returns:
            Optional[Dict]: 버전 정보
        """
        version_path = self.get_version_path(version_name)
        if version_path is None:
            return None
        
        version_info_path = version_path / "version_info.json"
        if not version_info_path.exists():
            logger.warning(f"Version info not found for {version_name}")
            return None
        
        try:
            with open(version_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version info: {e}")
            return None
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        버전 목록 조회
        
        Returns:
            List[Dict]: 버전 정보 리스트
        """
        versions = []
        active_version = self.get_active_version()
        
        for version_dir in self.base_path.iterdir():
            if not version_dir.is_dir() or version_dir.name == "__pycache__":
                continue
            
            if version_dir.name == "active_version.txt":
                continue
            
            version_info = self.get_version_info(version_dir.name)
            if version_info:
                version_info['is_active'] = (version_dir.name == active_version)
                version_info['path'] = str(version_dir)
                versions.append(version_info)
        
        return sorted(versions, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def copy_version(
        self,
        source_version: str,
        target_version: str,
        update_status: Optional[str] = None
    ) -> Optional[Path]:
        """
        버전 복사 (실험용)
        
        Args:
            source_version: 원본 버전 이름
            target_version: 대상 버전 이름
            update_status: 상태 업데이트 (None이면 원본과 동일)
        
        Returns:
            Optional[Path]: 복사된 버전 경로
        """
        source_path = self.get_version_path(source_version)
        if source_path is None:
            logger.error(f"Source version {source_version} not found")
            return None
        
        target_path = self.base_path / target_version
        if target_path.exists():
            logger.error(f"Target version {target_version} already exists")
            return None
        
        try:
            shutil.copytree(source_path, target_path)
            
            version_info = self.get_version_info(target_version)
            if version_info:
                version_info['version'] = target_version
                version_info['created_at'] = datetime.now().isoformat()
                if update_status:
                    version_info['status'] = update_status
                
                version_info_path = target_path / "version_info.json"
                with open(version_info_path, 'w', encoding='utf-8') as f:
                    json.dump(version_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Copied version {source_version} to {target_version}")
            return target_path
            
        except Exception as e:
            logger.error(f"Failed to copy version: {e}")
            return None
    
    def delete_version(self, version_name: str, force: bool = False) -> bool:
        """
        버전 삭제
        
        Args:
            version_name: 버전 이름
            force: 활성 버전도 삭제할지 여부
        
        Returns:
            bool: 성공 여부
        """
        active_version = self.get_active_version()
        if version_name == active_version and not force:
            logger.error(f"Cannot delete active version {version_name} without force=True")
            return False
        
        version_path = self.get_version_path(version_name)
        if version_path is None:
            logger.error(f"Version {version_name} not found")
            return False
        
        try:
            shutil.rmtree(version_path)
            
            if version_name == active_version:
                self.active_version_file.unlink(missing_ok=True)
            
            logger.info(f"Deleted version: {version_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def save_index(
        self,
        version_name: str,
        index: Any,
        id_mapping: Dict[int, int],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        FAISS 인덱스 및 메타데이터 저장
        
        Args:
            version_name: 버전 이름
            index: FAISS 인덱스 객체
            id_mapping: FAISS 인덱스 ID → chunk_id 매핑
            metadata: 청크 메타데이터 리스트
        
        Returns:
            bool: 성공 여부
        """
        version_path = self.get_version_path(version_name)
        if version_path is None:
            logger.error(f"Version {version_name} not found")
            return False
        
        try:
            import faiss
            index_path = version_path / "index.faiss"
            faiss.write_index(index, str(index_path))
            
            id_mapping_path = version_path / "id_mapping.json"
            with open(id_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(id_mapping, f, indent=2)
            
            metadata_path = version_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index for version {version_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def load_index(
        self,
        version_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        FAISS 인덱스 및 메타데이터 로드
        
        Args:
            version_name: 버전 이름 (None이면 활성 버전)
        
        Returns:
            Optional[Dict]: 인덱스, id_mapping, metadata를 포함한 딕셔너리
        """
        version_path = self.get_version_path(version_name)
        if version_path is None:
            return None
        
        try:
            import faiss
            index_path = version_path / "index.faiss"
            if not index_path.exists():
                logger.warning(f"FAISS index not found for version {version_name}")
                return None
            
            index = faiss.read_index(str(index_path))
            
            id_mapping_path = version_path / "id_mapping.json"
            id_mapping = {}
            if id_mapping_path.exists():
                with open(id_mapping_path, 'r', encoding='utf-8') as f:
                    id_mapping = json.load(f)
            
            metadata_path = version_path / "metadata.pkl"
            metadata = []
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            return {
                'index': index,
                'id_mapping': id_mapping,
                'metadata': metadata,
                'version_info': self.get_version_info(version_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None

