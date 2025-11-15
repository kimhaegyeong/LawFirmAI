"""
FAISS 버전 백업 및 복원 시스템

FAISS 인덱스 버전을 압축 백업하고 복원합니다.
"""
import logging
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FAISSBackupManager:
    """FAISS 버전 백업 및 복원 클래스"""
    
    def __init__(self, backup_path: str = "data/backups/faiss_versions", faiss_version_manager=None):
        """
        초기화
        
        Args:
            backup_path: 백업 저장 경로
            faiss_version_manager: FAISSVersionManager 인스턴스
        """
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.faiss_version_manager = faiss_version_manager
    
    def backup_version(self, version_name: str, version_path: Optional[Path] = None) -> Optional[Path]:
        """
        버전 전체를 압축 백업
        
        Args:
            version_name: 버전 이름
            version_path: 버전 디렉토리 경로 (None이면 version_manager에서 조회)
        
        Returns:
            Optional[Path]: 백업 파일 경로
        """
        if version_path is None:
            if self.faiss_version_manager is None:
                logger.error("FAISSVersionManager not provided and version_path is None")
                return None
            version_path = self.faiss_version_manager.get_version_path(version_name)
            if version_path is None:
                logger.error(f"Version {version_name} not found")
                return None
        
        if not version_path.exists():
            logger.error(f"Version path does not exist: {version_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_path / f"{version_name}_{timestamp}.tar.gz"
        
        try:
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(version_path, arcname=version_name)
            
            logger.info(f"Backup completed: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to backup version: {e}")
            return None
    
    def restore_version(self, backup_file: Path, target_path: Optional[Path] = None) -> bool:
        """
        백업에서 버전 복원
        
        Args:
            backup_file: 백업 파일 경로
            target_path: 복원할 대상 경로 (None이면 version_manager의 base_path 사용)
        
        Returns:
            bool: 성공 여부
        """
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        if target_path is None:
            if self.faiss_version_manager is None:
                logger.error("FAISSVersionManager not provided and target_path is None")
                return False
            target_path = self.faiss_version_manager.base_path
        else:
            target_path = Path(target_path)
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(target_path)
            
            logger.info(f"Restore completed: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore version: {e}")
            return False
    
    def cleanup_old_backups(self, keep_recent: int = 5):
        """
        오래된 백업 삭제
        
        Args:
            keep_recent: 유지할 최근 백업 개수
        """
        backups = sorted(
            self.backup_path.glob("*.tar.gz"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        deleted_count = 0
        for backup in backups[keep_recent:]:
            try:
                backup.unlink()
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup.name}")
            except Exception as e:
                logger.warning(f"Failed to delete backup {backup.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backups")
    
    def list_backups(self, version_name: Optional[str] = None) -> list:
        """
        백업 목록 조회
        
        Args:
            version_name: 특정 버전의 백업만 조회 (None이면 전체)
        
        Returns:
            list: 백업 파일 리스트
        """
        if version_name:
            pattern = f"{version_name}_*.tar.gz"
            backups = list(self.backup_path.glob(pattern))
        else:
            backups = list(self.backup_path.glob("*.tar.gz"))
        
        return sorted(backups, key=lambda x: x.stat().st_mtime, reverse=True)

