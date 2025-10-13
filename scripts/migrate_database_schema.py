#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 스키마 마이그레이션 스크립트
ML 강화 파싱 결과를 지원하는 새로운 필드를 추가합니다.

Usage:
  python scripts/migrate_database_schema.py --db-path data/lawfirm.db
  python scripts/migrate_database_schema.py --db-path data/lawfirm.db --backup-dir data/backups
  python scripts/migrate_database_schema.py --help
"""

import argparse
import logging
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/schema_migration.log')
    ]
)
logger = logging.getLogger(__name__)


class DatabaseSchemaMigrator:
    """데이터베이스 스키마 마이그레이션 클래스"""
    
    def __init__(self, db_path: str, backup_dir: Optional[str] = None):
        """
        마이그레이션 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            backup_dir: 백업 디렉토리 경로
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir) if backup_dir else self.db_path.parent / "backups"
        if self.backup_dir.is_file():
            # 파일인 경우 부모 디렉토리 사용
            self.backup_dir = self.backup_dir.parent
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 마이그레이션 통계
        self.migration_stats = {
            'start_time': None,
            'end_time': None,
            'tables_modified': 0,
            'columns_added': 0,
            'backup_created': False,
            'errors': []
        }
        
        logger.info(f"DatabaseSchemaMigrator initialized for: {self.db_path}")
        logger.info(f"Backup directory: {self.backup_dir}")
    
    def create_backup(self) -> bool:
        """데이터베이스 백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"lawfirm_backup_{timestamp}.db"
            
            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(self.db_path, backup_path)
            
            self.migration_stats['backup_created'] = True
            logger.info(f"Backup created successfully: {backup_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create backup: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def check_column_exists(self, conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        """컬럼 존재 여부 확인"""
        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            return column_name in columns
        except Exception as e:
            logger.error(f"Error checking column {column_name} in {table_name}: {e}")
            return False
    
    def add_column_if_not_exists(self, conn: sqlite3.Connection, table_name: str, 
                                column_name: str, column_definition: str) -> bool:
        """컬럼이 존재하지 않으면 추가"""
        try:
            if not self.check_column_exists(conn, table_name, column_name):
                cursor = conn.cursor()
                sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                cursor.execute(sql)
                conn.commit()
                logger.info(f"Added column {column_name} to {table_name}")
                self.migration_stats['columns_added'] += 1
                return True
            else:
                logger.info(f"Column {column_name} already exists in {table_name}")
                return False
        except Exception as e:
            error_msg = f"Failed to add column {column_name} to {table_name}: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def migrate_assembly_laws_table(self, conn: sqlite3.Connection) -> bool:
        """assembly_laws 테이블 마이그레이션"""
        logger.info("Migrating assembly_laws table...")
        
        # ML 강화 관련 컬럼들
        columns_to_add = [
            ("ml_enhanced", "BOOLEAN DEFAULT 0"),
            ("parsing_quality_score", "REAL"),
            ("article_count", "INTEGER"),
            ("supplementary_count", "INTEGER"),
            ("control_characters_removed", "BOOLEAN DEFAULT 1")
        ]
        
        success_count = 0
        for column_name, column_definition in columns_to_add:
            if self.add_column_if_not_exists(conn, "assembly_laws", column_name, column_definition):
                success_count += 1
        
        if success_count > 0:
            self.migration_stats['tables_modified'] += 1
            logger.info(f"Successfully migrated assembly_laws table: {success_count} columns added")
            return True
        else:
            logger.info("No new columns added to assembly_laws table")
            return True
    
    def migrate_assembly_articles_table(self, conn: sqlite3.Connection) -> bool:
        """assembly_articles 테이블 마이그레이션"""
        logger.info("Migrating assembly_articles table...")
        
        # ML 강화 관련 컬럼들
        columns_to_add = [
            ("is_supplementary", "BOOLEAN DEFAULT 0"),
            ("ml_confidence_score", "REAL"),
            ("parsing_method", "TEXT DEFAULT 'rule_based'"),
            ("article_type", "TEXT")
        ]
        
        success_count = 0
        for column_name, column_definition in columns_to_add:
            if self.add_column_if_not_exists(conn, "assembly_articles", column_name, column_definition):
                success_count += 1
        
        if success_count > 0:
            self.migration_stats['tables_modified'] += 1
            logger.info(f"Successfully migrated assembly_articles table: {success_count} columns added")
            return True
        else:
            logger.info("No new columns added to assembly_articles table")
            return True
    
    def create_ml_enhanced_indices(self, conn: sqlite3.Connection) -> bool:
        """ML 강화 관련 인덱스 생성"""
        logger.info("Creating ML enhanced indices...")
        
        try:
            cursor = conn.cursor()
            
            # ML 강화 관련 인덱스들
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_assembly_laws_ml_enhanced ON assembly_laws(ml_enhanced)",
                "CREATE INDEX IF NOT EXISTS idx_assembly_laws_quality_score ON assembly_laws(parsing_quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_assembly_articles_supplementary ON assembly_articles(is_supplementary)",
                "CREATE INDEX IF NOT EXISTS idx_assembly_articles_ml_confidence ON assembly_articles(ml_confidence_score)",
                "CREATE INDEX IF NOT EXISTS idx_assembly_articles_parsing_method ON assembly_articles(parsing_method)",
                "CREATE INDEX IF NOT EXISTS idx_assembly_articles_article_type ON assembly_articles(article_type)"
            ]
            
            for index_sql in indices:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("ML enhanced indices created successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create ML enhanced indices: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def verify_migration(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """마이그레이션 결과 검증"""
        logger.info("Verifying migration results...")
        
        verification_results = {
            'assembly_laws_columns': [],
            'assembly_articles_columns': [],
            'indices_created': [],
            'total_records': {}
        }
        
        try:
            cursor = conn.cursor()
            
            # assembly_laws 테이블 컬럼 확인
            cursor.execute("PRAGMA table_info(assembly_laws)")
            assembly_laws_columns = [row[1] for row in cursor.fetchall()]
            verification_results['assembly_laws_columns'] = assembly_laws_columns
            
            # assembly_articles 테이블 컬럼 확인
            cursor.execute("PRAGMA table_info(assembly_articles)")
            assembly_articles_columns = [row[1] for row in cursor.fetchall()]
            verification_results['assembly_articles_columns'] = assembly_articles_columns
            
            # 인덱스 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%ml%'")
            ml_indices = [row[0] for row in cursor.fetchall()]
            verification_results['indices_created'] = ml_indices
            
            # 레코드 수 확인
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            verification_results['total_records']['assembly_laws'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            verification_results['total_records']['assembly_articles'] = cursor.fetchone()[0]
            
            logger.info("Migration verification completed")
            return verification_results
            
        except Exception as e:
            error_msg = f"Failed to verify migration: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return verification_results
    
    def migrate(self) -> bool:
        """전체 마이그레이션 실행"""
        logger.info("Starting database schema migration...")
        self.migration_stats['start_time'] = datetime.now()
        
        try:
            # 1. 백업 생성
            if not self.create_backup():
                logger.error("Failed to create backup. Migration aborted.")
                return False
            
            # 2. 데이터베이스 연결
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            
            # 3. 테이블 마이그레이션
            self.migrate_assembly_laws_table(conn)
            self.migrate_assembly_articles_table(conn)
            
            # 4. 인덱스 생성
            self.create_ml_enhanced_indices(conn)
            
            # 5. 검증
            verification_results = self.verify_migration(conn)
            
            # 6. 마이그레이션 완료
            conn.close()
            self.migration_stats['end_time'] = datetime.now()
            
            # 7. 결과 출력
            self.print_migration_summary(verification_results)
            
            logger.info("Database schema migration completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            self.migration_stats['end_time'] = datetime.now()
            return False
    
    def print_migration_summary(self, verification_results: Dict[str, Any]):
        """마이그레이션 결과 요약 출력"""
        print("\n" + "="*80)
        print("DATABASE SCHEMA MIGRATION SUMMARY")
        print("="*80)
        
        print(f"Start Time: {self.migration_stats['start_time']}")
        print(f"End Time: {self.migration_stats['end_time']}")
        print(f"Duration: {self.migration_stats['end_time'] - self.migration_stats['start_time']}")
        
        print(f"\nTables Modified: {self.migration_stats['tables_modified']}")
        print(f"Columns Added: {self.migration_stats['columns_added']}")
        print(f"Backup Created: {'Yes' if self.migration_stats['backup_created'] else 'No'}")
        
        print(f"\nTotal Records:")
        for table, count in verification_results['total_records'].items():
            print(f"  {table}: {count:,}")
        
        print(f"\nNew Columns in assembly_laws:")
        ml_columns = [col for col in verification_results['assembly_laws_columns'] 
                     if col in ['ml_enhanced', 'parsing_quality_score', 'article_count', 
                               'supplementary_count', 'control_characters_removed']]
        for col in ml_columns:
            print(f"  [OK] {col}")
        
        print(f"\nNew Columns in assembly_articles:")
        ml_columns = [col for col in verification_results['assembly_articles_columns'] 
                     if col in ['is_supplementary', 'ml_confidence_score', 'parsing_method', 'article_type']]
        for col in ml_columns:
            print(f"  [OK] {col}")
        
        print(f"\nML Enhanced Indices Created: {len(verification_results['indices_created'])}")
        for idx in verification_results['indices_created']:
            print(f"  [OK] {idx}")
        
        if self.migration_stats['errors']:
            print(f"\nErrors ({len(self.migration_stats['errors'])}):")
            for error in self.migration_stats['errors']:
                print(f"  [ERROR] {error}")
        
        print("="*80)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Database Schema Migration for ML Enhanced Law Data")
    parser.add_argument("--db-path", default="data/lawfirm.db", 
                       help="Database file path (default: data/lawfirm.db)")
    parser.add_argument("--backup-dir", 
                       help="Backup directory path (default: data/backups)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # 데이터베이스 파일 존재 확인
    if not Path(args.db_path).exists():
        logger.error(f"Database file not found: {args.db_path}")
        return 1
    
    # 마이그레이션 실행
    migrator = DatabaseSchemaMigrator(args.db_path, args.backup_dir)
    
    if args.dry_run:
        logger.info("Dry run mode - no changes will be made")
        # TODO: 드라이 런 로직 구현
        return 0
    
    success = migrator.migrate()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
