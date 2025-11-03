#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?�이?�베?�스 백업 ?�크립트
마이그레?�션 ???�이?�베?�스 백업???�성?�니??

Usage:
  python scripts/backup_database.py --output data/backups/lawfirm_backup_20251013.db
  python scripts/backup_database.py --db-path data/lawfirm.db --output data/backups/
  python scripts/backup_database.py --help
"""

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/database_backup.log')
    ]
)
logger = logging.getLogger(__name__)


class DatabaseBackup:
    """?�이?�베?�스 백업 ?�래??""
    
    def __init__(self, db_path: str, output_path: Optional[str] = None):
        """
        백업 초기??
        
        Args:
            db_path: ?�본 ?�이?�베?�스 ?�일 경로
            output_path: 백업 ?�일 경로 (?�렉?�리 ?�는 ?�일)
        """
        self.db_path = Path(db_path)
        self.output_path = Path(output_path) if output_path else None
        
        # 백업 ?�계
        self.backup_stats = {
            'start_time': None,
            'end_time': None,
            'source_size': 0,
            'backup_size': 0,
            'backup_path': None,
            'success': False
        }
        
        logger.info(f"DatabaseBackup initialized for: {self.db_path}")
    
    def create_backup(self) -> bool:
        """?�이?�베?�스 백업 ?�성"""
        try:
            # ?�본 ?�일 존재 ?�인
            if not self.db_path.exists():
                logger.error(f"Source database not found: {self.db_path}")
                return False
            
            # 백업 경로 결정
            backup_path = self._determine_backup_path()
            
            # 백업 ?�렉?�리 ?�성
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating backup: {backup_path}")
            self.backup_stats['start_time'] = datetime.now()
            
            # ?�일 복사
            shutil.copy2(self.db_path, backup_path)
            
            # ?�계 ?�데?�트
            self.backup_stats['end_time'] = datetime.now()
            self.backup_stats['source_size'] = self.db_path.stat().st_size
            self.backup_stats['backup_size'] = backup_path.stat().st_size
            self.backup_stats['backup_path'] = str(backup_path)
            self.backup_stats['success'] = True
            
            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Source size: {self.backup_stats['source_size']:,} bytes")
            logger.info(f"Backup size: {self.backup_stats['backup_size']:,} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            self.backup_stats['end_time'] = datetime.now()
            return False
    
    def _determine_backup_path(self) -> Path:
        """백업 ?�일 경로 결정"""
        if self.output_path is None:
            # 기본 백업 ?�렉?�리 ?�용
            backup_dir = self.db_path.parent / "backups"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{self.db_path.stem}_backup_{timestamp}.db"
            return backup_dir / backup_filename
        
        elif self.output_path.is_dir():
            # ?�렉?�리가 지?�된 경우
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{self.db_path.stem}_backup_{timestamp}.db"
            return self.output_path / backup_filename
        
        else:
            # ?�일 경로가 지?�된 경우
            return self.output_path
    
    def verify_backup(self) -> bool:
        """백업 ?�일 검�?""
        try:
            backup_path = Path(self.backup_stats['backup_path'])
            
            if not backup_path.exists():
                logger.error("Backup file not found")
                return False
            
            # ?�일 ?�기 비교
            if self.backup_stats['source_size'] != self.backup_stats['backup_size']:
                logger.warning("Backup size differs from source size")
                return False
            
            # SQLite ?�일 무결??검??
            import sqlite3
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()
            
            if result != "ok":
                logger.error(f"Backup integrity check failed: {result}")
                return False
            
            logger.info("Backup verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def print_backup_summary(self):
        """백업 결과 ?�약 출력"""
        print("\n" + "="*60)
        print("DATABASE BACKUP SUMMARY")
        print("="*60)
        
        print(f"Source: {self.db_path}")
        print(f"Backup: {self.backup_stats['backup_path']}")
        
        if self.backup_stats['start_time'] and self.backup_stats['end_time']:
            duration = self.backup_stats['end_time'] - self.backup_stats['start_time']
            print(f"Duration: {duration}")
        
        print(f"Source Size: {self.backup_stats['source_size']:,} bytes")
        print(f"Backup Size: {self.backup_stats['backup_size']:,} bytes")
        print(f"Success: {'Yes' if self.backup_stats['success'] else 'No'}")
        
        print("="*60)


def main():
    """메인 ?�수"""
    parser = argparse.ArgumentParser(description="Database Backup Utility")
    parser.add_argument("--db-path", default="data/lawfirm.db",
                       help="Source database file path (default: data/lawfirm.db)")
    parser.add_argument("--output", "-o", required=True,
                       help="Backup output path (file or directory)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify backup after creation")
    
    args = parser.parse_args()
    
    # ?�본 ?�이?�베?�스 ?�일 존재 ?�인
    if not Path(args.db_path).exists():
        logger.error(f"Source database not found: {args.db_path}")
        return 1
    
    # 백업 ?�성
    backup = DatabaseBackup(args.db_path, args.output)
    success = backup.create_backup()
    
    if success:
        # 백업 검�?(?�션)
        if args.verify:
            verify_success = backup.verify_backup()
            if not verify_success:
                logger.error("Backup verification failed")
                return 1
        
        # 결과 출력
        backup.print_backup_summary()
        return 0
    else:
        logger.error("Backup creation failed")
        return 1


if __name__ == "__main__":
    exit(main())


