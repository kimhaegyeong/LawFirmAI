"""
MLflow 통합 관리 모듈

FAISS 인덱스 버전 관리를 MLflow로 수행합니다.
"""
import logging
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import sys

project_root = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.tracking
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowFAISSManager:
    """MLflow를 사용한 FAISS 인덱스 버전 관리"""
    
    def __init__(
        self,
        experiment_name: str = "faiss_index_versions",
        tracking_uri: Optional[str] = None
    ):
        """
        MLflow FAISS 관리자 초기화
        
        Args:
            experiment_name: MLflow 실험 이름
            tracking_uri: MLflow tracking URI (None이면 환경변수 또는 기본값)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not available. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                default_path = project_root / "mlflow" / "mlruns"
                default_path.mkdir(parents=True, exist_ok=True)
                tracking_uri = f"file:///{str(default_path).replace(os.sep, '/')}"
            mlflow.set_tracking_uri(tracking_uri)
        
        self.tracking_uri = mlflow.get_tracking_uri()
        
        self.is_local_filesystem = self._is_local_filesystem()
        if self.is_local_filesystem:
            self.local_base_path = self._get_local_base_path()
            logger.info(f"✅ 로컬 파일 시스템 모드: {self.local_base_path}")
        else:
            logger.info(f"🌐 원격 서버 모드: {self.tracking_uri}")
        
        try:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
        except Exception as e:
            if "is not a valid remote uri" in str(e) or "지정된 경로를 찾을 수 없습니다" in str(e):
                tracking_uri_path = self.tracking_uri.replace("file://", "").replace("file:///", "")
                if not os.path.isabs(tracking_uri_path):
                    tracking_uri_path = os.path.abspath(tracking_uri_path)
                else:
                    tracking_uri_path = os.path.normpath(tracking_uri_path)
                
                os.makedirs(tracking_uri_path, exist_ok=True)
                
                if os.name == 'nt':
                    self.tracking_uri = f"file:///{tracking_uri_path.replace(os.sep, '/')}"
                else:
                    self.tracking_uri = f"file://{tracking_uri_path}"
                
                mlflow.set_tracking_uri(self.tracking_uri)
                self.client = MlflowClient(tracking_uri=self.tracking_uri)
            else:
                raise
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name} (id: {self.experiment_id})")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (id: {self.experiment_id})")
    
    def _is_local_filesystem(self) -> bool:
        """로컬 파일 시스템인지 확인"""
        return self.tracking_uri.startswith("file://")
    
    def _get_local_base_path(self) -> Path:
        """로컬 파일 시스템의 기본 경로 반환"""
        uri_path = self.tracking_uri.replace("file://", "").replace("file:///", "")
        
        if os.name == 'nt':
            if uri_path.startswith('/') and len(uri_path) > 1 and uri_path[1].isalpha() and uri_path[2:4] == ':/':
                uri_path = uri_path[1:]
        
        if not os.path.isabs(uri_path):
            uri_path = os.path.abspath(uri_path)
        else:
            uri_path = os.path.normpath(uri_path)
        
        return Path(uri_path)
    
    def _get_local_artifact_path(self, run_id: str, artifact_path: str = "faiss_index") -> Path:
        """로컬 파일 시스템에서 아티팩트 경로 계산"""
        return self.local_base_path / str(self.experiment_id) / run_id / "artifacts" / artifact_path
    
    def load_version_info_from_local(self, run_id: str) -> Optional[Dict[str, Any]]:
        """로컬 파일 시스템에서 version_info.json 직접 로드"""
        if not self.is_local_filesystem:
            logger.debug("원격 서버 모드: 로컬 파일 시스템 직접 접근 불가")
            return None
        
        try:
            version_info_path = self.local_base_path / str(self.experiment_id) / run_id / "artifacts" / "version_info.json"
            
            # 파일 존재 여부 사전 확인 및 경로 검증
            if not version_info_path.parent.exists():
                logger.debug(f"⚠️  아티팩트 디렉토리 없음: {version_info_path.parent}")
                return None
            
            if not version_info_path.exists():
                logger.debug(f"⚠️  로컬 경로에 version_info.json 없음: {version_info_path}")
                return None
            
            # 파일 크기 확인 (빈 파일 방지)
            file_size = version_info_path.stat().st_size
            if file_size == 0:
                logger.warning(f"⚠️  version_info.json이 비어있음: {version_info_path}")
                return None
            
            # 파일 읽기 시도
            logger.info(f"✅ 로컬 파일 시스템에서 version_info.json 직접 로드: {version_info_path} (크기: {file_size} bytes)")
            with open(version_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(f"⚠️  version_info.json이 딕셔너리가 아님: {type(data)}")
                    return None
                return data
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  로컬 version_info.json JSON 파싱 실패: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️  로컬 version_info.json 로드 실패: {e}")
            return None
    
    def create_run(
        self,
        version_name: str,
        embedding_version_id: int,
        chunking_strategy: str,
        chunking_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        document_count: int = 0,
        total_chunks: int = 0,
        status: str = "active"
    ) -> str:
        """
        MLflow run 생성
        
        Args:
            version_name: 버전 이름
            embedding_version_id: 임베딩 버전 ID
            chunking_strategy: 청킹 전략
            chunking_config: 청킹 설정
            embedding_config: 임베딩 설정
            document_count: 문서 수
            total_chunks: 총 청크 수
            status: 버전 상태
        
        Returns:
            str: run_id
        """
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=version_name) as run:
            run_id = run.info.run_id
            
            mlflow.set_tags({
                "version": version_name,
                "status": status,
                "embedding_version_id": str(embedding_version_id),
                "chunking_strategy": chunking_strategy
            })
            
            mlflow.log_params({
                "embedding_version_id": embedding_version_id,
                "chunking_strategy": chunking_strategy,
                "document_count": document_count,
                "total_chunks": total_chunks,
                "status": status
            })
            
            mlflow.log_metrics({
                "document_count": document_count,
                "total_chunks": total_chunks
            })
            
            version_info = {
                "version": version_name,
                "embedding_version_id": embedding_version_id,
                "chunking_strategy": chunking_strategy,
                "created_at": datetime.now().isoformat(),
                "chunking_config": chunking_config,
                "embedding_config": embedding_config,
                "document_count": document_count,
                "total_chunks": total_chunks,
                "status": status,
                "mlflow_run_id": run_id,
                "mlflow_tracking_uri": self.tracking_uri
            }
            
            mlflow.log_dict(version_info, "version_info.json")
            
            logger.info(f"Created MLflow run: {version_name} (run_id: {run_id})")
            return run_id
    
    def save_index(
        self,
        run_id: str,
        index: Any,
        id_mapping: Dict[int, int],
        metadata: List[Dict[str, Any]],
        index_path: Optional[Path] = None
    ) -> bool:
        """
        FAISS 인덱스를 MLflow 아티팩트로 저장
        
        Args:
            run_id: MLflow run ID
            index: FAISS 인덱스 객체
            id_mapping: ID 매핑
            metadata: 메타데이터
            index_path: 로컬 인덱스 파일 경로 (None이면 임시 파일 사용)
        
        Returns:
            bool: 성공 여부
        """
        try:
            import faiss
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "faiss_index"
                temp_path.mkdir()
                
                if index_path and index_path.exists():
                    shutil.copy(index_path, temp_path / "index.faiss")
                else:
                    faiss.write_index(index, str(temp_path / "index.faiss"))
                
                with open(temp_path / "id_mapping.json", 'w', encoding='utf-8') as f:
                    json.dump(id_mapping, f, indent=2)
                
                with open(temp_path / "metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                index_stats = {
                    "num_vectors": index.ntotal,
                    "dimension": index.d,
                    "index_type": type(index).__name__,
                    "id_mapping_size": len(id_mapping),
                    "metadata_size": len(metadata)
                }
                
                with open(temp_path / "index_stats.json", 'w', encoding='utf-8') as f:
                    json.dump(index_stats, f, indent=2)
                
                # 활성 run 컨텍스트 설정 후 아티팩트 저장
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifacts(str(temp_path), "faiss_index")
                    mlflow.log_metrics({
                        "num_vectors": index.ntotal,
                        "dimension": index.d,
                        "id_mapping_size": len(id_mapping),
                        "metadata_size": len(metadata)
                    })
                
                logger.info(f"Saved FAISS index to MLflow run: {run_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save index to MLflow: {e}")
            return False
    
    def load_index(
        self,
        run_id: str,
        output_dir: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        MLflow에서 FAISS 인덱스 로드 (로컬 파일 시스템 최적화)
        
        Args:
            run_id: MLflow run ID
            output_dir: 다운로드할 디렉토리 (None이면 로컬 직접 접근 또는 임시 디렉토리)
        
        Returns:
            Optional[Dict]: 인덱스, id_mapping, metadata를 포함한 딕셔너리
        """
        try:
            import faiss
            
            if self.is_local_filesystem and not output_dir:
                return self._load_index_from_local_path(run_id)
            
            return self._load_index_from_download(run_id, output_dir)
            
        except Exception as e:
            logger.error(f"Failed to load index from MLflow run {run_id}: {e}", exc_info=True)
            return None
    
    def _load_index_from_local_path(self, run_id: str) -> Optional[Dict[str, Any]]:
        """로컬 파일 시스템에서 직접 인덱스 로드 (복사 없음)"""
        try:
            import faiss
            
            run_info = self.client.get_run(run_id)
            run_tags = run_info.data.tags if hasattr(run_info.data, 'tags') else {}
            version_name = run_tags.get('version', None)
            
            vector_store_path = project_root / "data" / "vector_store"
            
            if version_name and vector_store_path.exists():
                version_path = vector_store_path / version_name
                index_path = version_path / "index.faiss"
                
                if index_path.exists():
                    logger.info(f"✅ data/vector_store에서 직접 로드: {index_path}")
                    logger.info(f"   📁 로컬 인덱스 경로: {version_path}")
                    
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
                    
                    version_info_path = version_path / "version_info.json"
                    stats = {}
                    if version_info_path.exists():
                        with open(version_info_path, 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                    
                    return {
                        'index': index,
                        'id_mapping': id_mapping,
                        'metadata': metadata,
                        'stats': stats,
                        'run_info': run_info.to_dictionary(),
                        'local_path': str(version_path)
                    }
            
            artifacts_path = self._get_local_artifact_path(run_id, "faiss_index")
            index_path = artifacts_path / "index.faiss"
            
            if not index_path.exists():
                logger.warning(f"Index file not found at local path: {index_path}")
                logger.info("Falling back to download method...")
                return self._load_index_from_download(run_id, None)
            
            logger.info(f"✅ MLflow 로컬 경로에서 직접 로드: {index_path}")
            logger.info(f"   📁 MLflow 아티팩트 경로: {artifacts_path}")
            
            index = faiss.read_index(str(index_path))
            
            id_mapping_path = artifacts_path / "id_mapping.json"
            id_mapping = {}
            if id_mapping_path.exists():
                with open(id_mapping_path, 'r', encoding='utf-8') as f:
                    id_mapping = json.load(f)
            
            metadata_path = artifacts_path / "metadata.pkl"
            metadata = []
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            stats_path = artifacts_path / "index_stats.json"
            stats = {}
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
            
            return {
                'index': index,
                'id_mapping': id_mapping,
                'metadata': metadata,
                'stats': stats,
                'run_info': run_info.to_dictionary(),
                'local_path': str(artifacts_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load index from local path: {e}", exc_info=True)
            logger.info("Falling back to download method...")
            return self._load_index_from_download(run_id, None)
    
    def _load_index_from_download(
        self,
        run_id: str,
        output_dir: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """기존 다운로드 방식으로 인덱스 로드 (원격 서버용)"""
        try:
            import faiss
            
            artifacts = self.client.list_artifacts(run_id, "faiss_index")
            if not artifacts:
                logger.warning(f"No FAISS artifacts found for run {run_id}")
                return None
            
            if output_dir:
                download_path = output_dir
                download_path.mkdir(parents=True, exist_ok=True)
                use_temp = False
            else:
                temp_dir = tempfile.mkdtemp()
                download_path = Path(temp_dir)
                use_temp = True
            
            try:
                logger.info(f"📥 MLflow에서 인덱스 다운로드: run_id={run_id}")
                
                for artifact in artifacts:
                    artifact_dest = download_path / artifact.path
                    artifact_dest.parent.mkdir(parents=True, exist_ok=True)
                    self.client.download_artifacts(run_id, artifact.path, str(artifact_dest.parent))
                
                index_path = download_path / "faiss_index" / "index.faiss"
                
                if not index_path.exists():
                    possible_paths = [
                        download_path / "index.faiss",
                        download_path / "faiss_index" / "index.faiss",
                    ]
                    
                    for candidate_path in possible_paths:
                        if candidate_path.exists():
                            index_path = candidate_path
                            logger.info(f"Found index file at: {index_path}")
                            break
                    
                    if not index_path.exists():
                        faiss_files = list(download_path.rglob("*.faiss"))
                        if faiss_files:
                            index_path = faiss_files[0]
                            logger.info(f"Found index file (auto-detected): {index_path}")
                        else:
                            logger.error(f"Index file not found. Searched in: {download_path}")
                            logger.error(f"Download path structure: {list(download_path.rglob('*')) if download_path.exists() else 'Directory does not exist'}")
                            return None
                
                index = faiss.read_index(str(index_path))
                
                id_mapping_path = download_path / "faiss_index" / "id_mapping.json"
                id_mapping = {}
                if id_mapping_path.exists():
                    with open(id_mapping_path, 'r', encoding='utf-8') as f:
                        id_mapping = json.load(f)
                
                metadata_path = download_path / "faiss_index" / "metadata.pkl"
                metadata = []
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                
                stats_path = download_path / "faiss_index" / "index_stats.json"
                stats = {}
                if stats_path.exists():
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                
                run_info = self.client.get_run(run_id)
                
                return {
                    'index': index,
                    'id_mapping': id_mapping,
                    'metadata': metadata,
                    'stats': stats,
                    'run_info': run_info.to_dictionary(),
                    'download_path': str(download_path) if not use_temp else None
                }
                
            finally:
                if use_temp:
                    shutil.rmtree(download_path, ignore_errors=True)
                    
        except Exception as e:
            logger.error(f"Failed to load index from download: {e}", exc_info=True)
            return None
    
    def list_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        MLflow runs 목록 조회
        
        Args:
            filter_string: MLflow filter string
            max_results: 최대 결과 수
        
        Returns:
            List[Dict]: run 정보 리스트
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            results = []
            for _, run in runs.iterrows():
                results.append({
                    "run_id": run["run_id"],
                    "version": run.get("tags.version", ""),
                    "status": run.get("tags.status", ""),
                    "start_time": run["start_time"],
                    "metrics": run.filter(regex="^metrics\.").to_dict(),
                    "params": run.filter(regex="^params\.").to_dict()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []
    
    def get_production_run(self) -> Optional[str]:
        """
        프로덕션 태그가 있는 run ID 조회
        
        Returns:
            Optional[str]: 프로덕션 run ID, 없으면 None
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="tags.status='production_ready'",
                max_results=1,
                order_by=["start_time DESC"]
            )
            
            if not runs.empty:
                return runs.iloc[0]["run_id"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get production run: {e}")
            return None
    
    def get_run_by_version(self, version_name: str) -> Optional[str]:
        """
        버전 이름으로 run ID 조회
        
        Args:
            version_name: 버전 이름
        
        Returns:
            Optional[str]: run ID, 없으면 None
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.version='{version_name}'",
                max_results=1,
                order_by=["start_time DESC"]
            )
            
            if not runs.empty:
                return runs.iloc[0]["run_id"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get run by version: {e}")
            return None

