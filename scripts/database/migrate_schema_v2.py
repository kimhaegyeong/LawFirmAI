#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Schema Migration Script v2

This script migrates the database schema to support enhanced data quality features,
including duplicate detection, quality scoring, and version tracking.

Usage:
    python migrate_schema_v2.py --db-path data/lawfirm.db
    python migrate_schema_v2.py --db-path data/lawfirm.db --backup
    python migrate_schema_v2.py --help
"""

import argparse
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import database manager
try:
    from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False
    logging.warning("DatabaseManager not available. Using direct SQLite operations.")

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


class SchemaMigrationV2:
    """Database schema migration to version 2 with quality features"""
    
    def __init__(self, db_path: str):
        """
        Initialize schema migration
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = Path(db_path)
        self.backup_path = None
        
        # Migration version info
        self.migration_version = "2.0"
        self.migration_timestamp = datetime.now().isoformat()
        
        # New columns to add
        self.new_columns = {
            'assembly_laws': [
                ('law_name_hash', 'TEXT UNIQUE'),
                ('content_hash', 'TEXT UNIQUE'),
                ('quality_score', 'REAL DEFAULT 0.0'),
                ('duplicate_group_id', 'TEXT'),
                ('is_primary_version', 'BOOLEAN DEFAULT TRUE'),
                ('version_number', 'INTEGER DEFAULT 1'),
                ('parsing_method', 'TEXT DEFAULT "legacy"'),
                ('auto_corrected', 'BOOLEAN DEFAULT FALSE'),
                ('manual_review_required', 'BOOLEAN DEFAULT FALSE'),
                ('processing_version', 'TEXT DEFAULT "1.0"'),
                ('migration_timestamp', 'TEXT')
            ]
        }
        
        # New tables to create
        self.new_tables = {
            'duplicate_groups': '''
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    group_id TEXT PRIMARY KEY,
                    group_type TEXT NOT NULL,
                    primary_law_id TEXT NOT NULL,
                    duplicate_law_ids TEXT NOT NULL,
                    resolution_strategy TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (primary_law_id) REFERENCES assembly_laws (law_id)
                )
            ''',
            'quality_reports': '''
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id TEXT PRIMARY KEY,
                    law_id TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    article_count_score REAL NOT NULL,
                    title_extraction_score REAL NOT NULL,
                    article_sequence_score REAL NOT NULL,
                    structure_completeness_score REAL NOT NULL,
                    issues TEXT,
                    suggestions TEXT,
                    validation_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (law_id) REFERENCES assembly_laws (law_id)
                )
            ''',
            'migration_history': '''
                CREATE TABLE IF NOT EXISTS migration_history (
                    migration_id TEXT PRIMARY KEY,
                    migration_version TEXT NOT NULL,
                    migration_timestamp TIMESTAMP NOT NULL,
                    description TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    records_affected INTEGER DEFAULT 0
                )
            '''
        }
        
        # New indices to create
        self.new_indices = [
            'CREATE INDEX IF NOT EXISTS idx_assembly_laws_law_name_hash ON assembly_laws (law_name_hash)',
            'CREATE INDEX IF NOT EXISTS idx_assembly_laws_content_hash ON assembly_laws (content_hash)',
            'CREATE INDEX IF NOT EXISTS idx_assembly_laws_quality_score ON assembly_laws (quality_score)',
            'CREATE INDEX IF NOT EXISTS idx_assembly_laws_duplicate_group_id ON assembly_laws (duplicate_group_id)',
            'CREATE INDEX IF NOT EXISTS idx_assembly_laws_is_primary_version ON assembly_laws (is_primary_version)',
            'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_group_type ON duplicate_groups (group_type)',
            'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_primary_law_id ON duplicate_groups (primary_law_id)',
            'CREATE INDEX IF NOT EXISTS idx_quality_reports_law_id ON quality_reports (law_id)',
            'CREATE INDEX IF NOT EXISTS idx_quality_reports_overall_score ON quality_reports (overall_score)'
        ]
    
    def migrate(self, create_backup: bool = True) -> bool:
        """
        Perform the complete schema migration
        
        Args:
            create_backup: Whether to create a backup before migration
            
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            logger.info(f"Starting schema migration to version {self.migration_version}")
            
            # Check if database exists
            if not self.db_path.exists():
                logger.error(f"Database file not found: {self.db_path}")
                return False
            
            # Create backup if requested
            if create_backup:
                if not self._create_backup():
                    logger.error("Failed to create backup")
                    return False
            
            # Connect to database
            conn = sqlite3.connect(str(self.db_path))
            conn.execute('PRAGMA foreign_keys = ON')
            
            try:
                # Check current schema version
                current_version = self._get_current_schema_version(conn)
                logger.info(f"Current schema version: {current_version}")
                
                # Perform migration steps
                self._create_migration_record(conn)
                self._add_new_columns(conn)
                self._create_new_tables(conn)
                self._create_new_indices(conn)
                self._update_schema_version(conn)
                
                # Commit changes
                conn.commit()
                logger.info("Schema migration completed successfully")
                
                # Log migration statistics
                self._log_migration_statistics(conn)
                
                return True
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Migration failed: {e}")
                self._record_migration_error(conn, str(e))
                return False
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """Create a backup of the database before migration"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_v1_{timestamp}.db"
            
            shutil.copy2(self.db_path, self.backup_path)
            logger.info(f"Database backup created: {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _get_current_schema_version(self, conn: sqlite3.Connection) -> str:
        """Get current schema version from database"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            
            if cursor.fetchone():
                cursor.execute("SELECT version FROM schema_version ORDER BY updated_at DESC LIMIT 1")
                result = cursor.fetchone()
                return result[0] if result else "1.0"
            else:
                return "1.0"
                
        except Exception as e:
            logger.warning(f"Could not determine schema version: {e}")
            return "1.0"
    
    def _create_migration_record(self, conn: sqlite3.Connection):
        """Create migration history record"""
        try:
            cursor = conn.cursor()
            
            # Create migration_history table if it doesn't exist
            cursor.execute(self.new_tables['migration_history'])
            
            # Insert migration record
            migration_id = f"migration_{self.migration_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cursor.execute('''
                INSERT INTO migration_history 
                (migration_id, migration_version, migration_timestamp, description, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (migration_id, self.migration_version, self.migration_timestamp, 
                  f"Schema migration to version {self.migration_version}", True))
            
            logger.info(f"Created migration record: {migration_id}")
            
        except Exception as e:
            logger.error(f"Error creating migration record: {e}")
            raise
    
    def _add_new_columns(self, conn: sqlite3.Connection):
        """Add new columns to existing tables"""
        try:
            cursor = conn.cursor()
            
            for table_name, columns in self.new_columns.items():
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cursor.fetchone():
                    logger.warning(f"Table {table_name} does not exist, skipping column addition")
                    continue
                
                # Add each column
                for column_name, column_definition in columns:
                    try:
                        # Check if column already exists
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        existing_columns = [row[1] for row in cursor.fetchall()]
                        
                        if column_name not in existing_columns:
                            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                            cursor.execute(alter_sql)
                            logger.info(f"Added column {column_name} to table {table_name}")
                        else:
                            logger.info(f"Column {column_name} already exists in table {table_name}")
                            
                    except sqlite3.Error as e:
                        logger.error(f"Error adding column {column_name} to {table_name}: {e}")
                        raise
            
        except Exception as e:
            logger.error(f"Error adding new columns: {e}")
            raise
    
    def _create_new_tables(self, conn: sqlite3.Connection):
        """Create new tables"""
        try:
            cursor = conn.cursor()
            
            for table_name, create_sql in self.new_tables.items():
                try:
                    cursor.execute(create_sql)
                    logger.info(f"Created table: {table_name}")
                except sqlite3.Error as e:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
            
        except Exception as e:
            logger.error(f"Error creating new tables: {e}")
            raise
    
    def _create_new_indices(self, conn: sqlite3.Connection):
        """Create new indices"""
        try:
            cursor = conn.cursor()
            
            for index_sql in self.new_indices:
                try:
                    cursor.execute(index_sql)
                    logger.info(f"Created index: {index_sql.split()[-1]}")
                except sqlite3.Error as e:
                    logger.error(f"Error creating index: {e}")
                    raise
            
        except Exception as e:
            logger.error(f"Error creating indices: {e}")
            raise
    
    def _update_schema_version(self, conn: sqlite3.Connection):
        """Update schema version"""
        try:
            cursor = conn.cursor()
            
            # Create schema_version table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            
            # Insert new version
            cursor.execute('''
                INSERT OR REPLACE INTO schema_version (version, description)
                VALUES (?, ?)
            ''', (self.migration_version, f"Schema migration completed at {self.migration_timestamp}"))
            
            logger.info(f"Updated schema version to {self.migration_version}")
            
        except Exception as e:
            logger.error(f"Error updating schema version: {e}")
            raise
    
    def _log_migration_statistics(self, conn: sqlite3.Connection):
        """Log migration statistics"""
        try:
            cursor = conn.cursor()
            
            # Count records in each table
            tables_to_check = ['assembly_laws', 'duplicate_groups', 'quality_reports']
            stats = {}
            
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[table] = count
                except sqlite3.Error:
                    stats[table] = 0
            
            logger.info(f"Migration statistics: {stats}")
            
            # Update migration record with statistics
            migration_id = f"migration_{self.migration_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            total_records = sum(stats.values())
            cursor.execute('''
                UPDATE migration_history 
                SET records_affected = ?
                WHERE migration_id = ?
            ''', (total_records, migration_id))
            
        except Exception as e:
            logger.error(f"Error logging migration statistics: {e}")
    
    def _record_migration_error(self, conn: sqlite3.Connection, error_message: str):
        """Record migration error"""
        try:
            cursor = conn.cursor()
            migration_id = f"migration_{self.migration_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cursor.execute('''
                UPDATE migration_history 
                SET success = ?, error_message = ?
                WHERE migration_id = ?
            ''', (False, error_message, migration_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error recording migration error: {e}")
    
    def verify_migration(self) -> bool:
        """Verify that migration was successful"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check schema version
            cursor.execute("SELECT version FROM schema_version ORDER BY updated_at DESC LIMIT 1")
            result = cursor.fetchone()
            if not result or result[0] != self.migration_version:
                logger.error(f"Schema version mismatch. Expected: {self.migration_version}, Got: {result[0] if result else 'None'}")
                return False
            
            # Check new columns exist
            cursor.execute("PRAGMA table_info(assembly_laws)")
            columns = [row[1] for row in cursor.fetchall()]
            required_columns = ['law_name_hash', 'content_hash', 'quality_score', 'duplicate_group_id']
            
            for column in required_columns:
                if column not in columns:
                    logger.error(f"Required column {column} not found in assembly_laws table")
                    return False
            
            # Check new tables exist
            required_tables = ['duplicate_groups', 'quality_reports', 'migration_history']
            for table in required_tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    logger.error(f"Required table {table} not found")
                    return False
            
            logger.info("Migration verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False
        finally:
            conn.close()
    
    def rollback_migration(self) -> bool:
        """Rollback migration using backup"""
        try:
            if not self.backup_path or not self.backup_path.exists():
                logger.error("No backup file available for rollback")
                return False
            
            # Replace current database with backup
            shutil.copy2(self.backup_path, self.db_path)
            logger.info(f"Migration rolled back using backup: {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Database Schema Migration v2')
    
    parser.add_argument('--db-path', required=True, help='Path to database file')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    parser.add_argument('--verify', action='store_true', help='Verify migration after completion')
    parser.add_argument('--rollback', action='store_true', help='Rollback migration using backup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run migration
    try:
        migration = SchemaMigrationV2(args.db_path)
        
        if args.rollback:
            success = migration.rollback_migration()
        else:
            success = migration.migrate(args.backup)
            
            if success and args.verify:
                success = migration.verify_migration()
        
        if success:
            print("Migration completed successfully!")
            if args.backup and migration.backup_path:
                print(f"Backup created at: {migration.backup_path}")
        else:
            print("Migration failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        print(f"Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

