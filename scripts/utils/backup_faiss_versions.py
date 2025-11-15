"""
FAISS 버전 백업 스크립트

주기적으로 FAISS 버전을 백업하고 오래된 백업을 정리합니다.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.faiss_version_manager import FAISSVersionManager
from utils.faiss_backup_manager import FAISSBackupManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backup FAISS versions")
    parser.add_argument("--version", help="Specific version to backup (None for all active)")
    parser.add_argument("--vector-store-base", default="data/vector_store", help="Vector store base path")
    parser.add_argument("--backup-path", default="data/backups/faiss_versions", help="Backup path")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old backups")
    parser.add_argument("--keep-recent", type=int, default=5, help="Number of recent backups to keep")
    
    args = parser.parse_args()
    
    vector_store_base = args.vector_store_base
    backup_path = args.backup_path
    version_name = args.version
    cleanup = args.cleanup
    keep_recent = args.keep_recent
    
    faiss_version_manager = FAISSVersionManager(vector_store_base)
    backup_manager = FAISSBackupManager(backup_path, faiss_version_manager)
    
    if version_name:
        versions_to_backup = [version_name]
    else:
        active_version = faiss_version_manager.get_active_version()
        if active_version:
            versions_to_backup = [active_version]
        else:
            logger.warning("No active version found")
            versions_to_backup = []
    
    for version in versions_to_backup:
        logger.info(f"Backing up version: {version}")
        backup_file = backup_manager.backup_version(version)
        if backup_file:
            logger.info(f"Backup created: {backup_file}")
        else:
            logger.error(f"Failed to backup version: {version}")
            return 1
    
    if cleanup:
        logger.info(f"Cleaning up old backups (keeping {keep_recent} most recent)")
        backup_manager.cleanup_old_backups(keep_recent)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

