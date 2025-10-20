#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Migration Script

This script migrates existing data to the new schema by calculating hashes,
computing quality scores, detecting duplicates, and updating records.

Usage:
    python migrate_existing_data.py --db-path data/lawfirm.db
    python migrate_existing_data.py --db-path data/lawfirm.db --batch-size 100
    python migrate_existing_data.py --help
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    sys.path.append(str(project_root / 'scripts' / 'data_processing' / 'quality'))
    from data_quality_validator import DataQualityValidator
    from duplicate_detector import AdvancedDuplicateDetector
    from duplicate_resolver import IntelligentDuplicateResolver
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing quality modules: {e}")
    QUALITY_MODULES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_migration.log')
    ]
)
logger = logging.getLogger(__name__)


class DataMigration:
    """Data migration for existing records"""
    
    def __init__(self, db_path: str, batch_size: int = 50):
        """
        Initialize data migration
        
        Args:
            db_path: Path to the database file
            batch_size: Number of records to process in each batch
        """
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        
        # Migration statistics
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'updated_records': 0,
            'duplicates_found': 0,
            'quality_scores_calculated': 0,
            'hashes_calculated': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Initialize quality components
        if QUALITY_MODULES_AVAILABLE:
            self.quality_validator = DataQualityValidator()
            self.duplicate_detector = AdvancedDuplicateDetector()
            self.duplicate_resolver = IntelligentDuplicateResolver()
        else:
            logger.error("Quality modules not available. Data migration cannot run.")
            sys.exit(1)
    
    def migrate_data(self) -> bool:
        """
        Perform complete data migration
        
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            self.stats['start_time'] = datetime.now().isoformat()
            logger.info("Starting data migration")
            
            # Connect to database
            conn = sqlite3.connect(str(self.db_path))
            conn.execute('PRAGMA foreign_keys = ON')
            
            try:
                # Step 1: Count total records
                self._count_total_records(conn)
                
                # Step 2: Calculate hashes for all records
                self._calculate_hashes(conn)
                
                # Step 3: Calculate quality scores
                self._calculate_quality_scores(conn)
                
                # Step 4: Detect and resolve duplicates
                self._detect_and_resolve_duplicates(conn)
                
                # Step 5: Update migration metadata
                self._update_migration_metadata(conn)
                
                # Commit changes
                conn.commit()
                self.stats['end_time'] = datetime.now().isoformat()
                
                logger.info("Data migration completed successfully")
                self._log_migration_summary()
                
                return True
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Data migration failed: {e}")
                return False
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Data migration error: {e}")
            return False
    
    def _count_total_records(self, conn: sqlite3.Connection):
        """Count total records to migrate"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            self.stats['total_records'] = cursor.fetchone()[0]
            logger.info(f"Total records to migrate: {self.stats['total_records']}")
            
        except Exception as e:
            logger.error(f"Error counting records: {e}")
            raise
    
    def _calculate_hashes(self, conn: sqlite3.Connection):
        """Calculate hashes for all existing records"""
        try:
            logger.info("Calculating hashes for existing records")
            
            cursor = conn.cursor()
            
            # Get all records that don't have hashes yet
            cursor.execute('''
                SELECT law_id, law_name, full_text 
                FROM assembly_laws 
                WHERE law_name_hash IS NULL OR content_hash IS NULL
            ''')
            
            records = cursor.fetchall()
            logger.info(f"Found {len(records)} records needing hash calculation")
            
            # Process in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                self._process_hash_batch(conn, batch)
                
                # Log progress
                processed = min(i + self.batch_size, len(records))
                logger.info(f"Processed {processed}/{len(records)} records for hash calculation")
            
            logger.info("Hash calculation completed")
            
        except Exception as e:
            logger.error(f"Error calculating hashes: {e}")
            raise
    
    def _process_hash_batch(self, conn: sqlite3.Connection, batch: List[Tuple]):
        """Process a batch of records for hash calculation"""
        try:
            cursor = conn.cursor()
            
            for law_id, law_name, full_text in batch:
                # Calculate law name hash
                law_name_hash = self._calculate_law_name_hash(law_name)
                
                # Calculate content hash
                content_hash = self._calculate_content_hash(full_text)
                
                # Update record
                cursor.execute('''
                    UPDATE assembly_laws 
                    SET law_name_hash = ?, content_hash = ?
                    WHERE law_id = ?
                ''', (law_name_hash, content_hash, law_id))
                
                self.stats['hashes_calculated'] += 1
            
        except Exception as e:
            logger.error(f"Error processing hash batch: {e}")
            raise
    
    def _calculate_law_name_hash(self, law_name: str) -> str:
        """Calculate normalized hash for law name"""
        if not law_name:
            return ""
        
        # Normalize law name
        normalized_name = self._normalize_law_name(law_name)
        
        # Calculate MD5 hash
        return hashlib.md5(normalized_name.encode('utf-8')).hexdigest()
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash for content"""
        if not content:
            return ""
        
        # Normalize content
        normalized_content = self._normalize_content(content)
        
        # Calculate SHA256 hash
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    
    def _normalize_law_name(self, law_name: str) -> str:
        """Normalize law name for consistent hashing"""
        if not law_name:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = law_name.lower().strip()
        
        # Remove common variations
        variations = {
            '법률': '법',
            '규칙': '규칙',
            '령': '령',
            '규정': '규정'
        }
        
        for old, new in variations.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing"""
        if not content:
            return ""
        
        # Remove extra whitespace
        normalized = ' '.join(content.split())
        
        # Remove common formatting variations
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
        
        return normalized.strip()
    
    def _calculate_quality_scores(self, conn: sqlite3.Connection):
        """Calculate quality scores for all existing records"""
        try:
            logger.info("Calculating quality scores for existing records")
            
            cursor = conn.cursor()
            
            # Get all records that don't have quality scores yet
            cursor.execute('''
                SELECT law_id, law_name, full_text, articles
                FROM assembly_laws 
                WHERE quality_score IS NULL OR quality_score = 0.0
            ''')
            
            records = cursor.fetchall()
            logger.info(f"Found {len(records)} records needing quality score calculation")
            
            # Process in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                self._process_quality_batch(conn, batch)
                
                # Log progress
                processed = min(i + self.batch_size, len(records))
                logger.info(f"Processed {processed}/{len(records)} records for quality calculation")
            
            logger.info("Quality score calculation completed")
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            raise
    
    def _process_quality_batch(self, conn: sqlite3.Connection, batch: List[Tuple]):
        """Process a batch of records for quality score calculation"""
        try:
            cursor = conn.cursor()
            
            for law_id, law_name, full_text, articles_json in batch:
                # Parse articles if available
                articles = []
                if articles_json:
                    try:
                        articles = json.loads(articles_json)
                    except (json.JSONDecodeError, TypeError):
                        articles = []
                
                # Create law data structure for quality validation
                law_data = {
                    'law_name': law_name,
                    'articles': articles,
                    'full_text': full_text
                }
                
                # Calculate quality score
                quality_score = self.quality_validator.calculate_quality_score(law_data)
                
                # Store quality report
                quality_report = self.quality_validator.validate_parsing_quality(law_data)
                
                # Update record with quality score
                cursor.execute('''
                    UPDATE assembly_laws 
                    SET quality_score = ?
                    WHERE law_id = ?
                ''', (quality_score, law_id))
                
                # Store detailed quality report
                if quality_report:
                    report_id = f"qr_{law_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    cursor.execute('''
                        INSERT OR REPLACE INTO quality_reports 
                        (report_id, law_id, overall_score, article_count_score, 
                         title_extraction_score, article_sequence_score, 
                         structure_completeness_score, issues, suggestions, validation_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        report_id, law_id, quality_report.overall_score,
                        quality_report.article_count_score, quality_report.title_extraction_score,
                        quality_report.article_sequence_score, quality_report.structure_completeness_score,
                        json.dumps(quality_report.issues, ensure_ascii=False),
                        json.dumps(quality_report.suggestions, ensure_ascii=False),
                        quality_report.validation_timestamp
                    ))
                
                self.stats['quality_scores_calculated'] += 1
            
        except Exception as e:
            logger.error(f"Error processing quality batch: {e}")
            raise
    
    def _detect_and_resolve_duplicates(self, conn: sqlite3.Connection):
        """Detect and resolve duplicates in existing data"""
        try:
            logger.info("Detecting and resolving duplicates")
            
            cursor = conn.cursor()
            
            # Get all laws for duplicate detection
            cursor.execute('''
                SELECT law_id, law_name, full_text, articles, quality_score
                FROM assembly_laws
                ORDER BY quality_score DESC
            ''')
            
            laws = []
            for row in cursor.fetchall():
                law_id, law_name, full_text, articles_json, quality_score = row
                
                # Parse articles
                articles = []
                if articles_json:
                    try:
                        articles = json.loads(articles_json)
                    except (json.JSONDecodeError, TypeError):
                        articles = []
                
                law_data = {
                    'law_id': law_id,
                    'law_name': law_name,
                    'full_text': full_text,
                    'articles': articles,
                    'quality_score': quality_score or 0.0
                }
                laws.append(law_data)
            
            logger.info(f"Analyzing {len(laws)} laws for duplicates")
            
            # Detect content-level duplicates
            content_duplicates = self.duplicate_detector.detect_content_level_duplicates(laws)
            
            # Detect semantic duplicates
            semantic_duplicates = self.duplicate_detector.detect_semantic_duplicates(laws)
            
            # Combine all duplicate groups
            all_duplicates = content_duplicates + semantic_duplicates
            
            if all_duplicates:
                logger.info(f"Found {len(all_duplicates)} duplicate groups")
                
                # Resolve duplicates
                resolution_results = self.duplicate_resolver.resolve_duplicates(
                    all_duplicates, 'quality_based'
                )
                
                # Update database with resolution results
                self._update_duplicate_resolution(conn, resolution_results)
                
                self.stats['duplicates_found'] = len(all_duplicates)
            else:
                logger.info("No duplicates found")
            
        except Exception as e:
            logger.error(f"Error detecting and resolving duplicates: {e}")
            raise
    
    def _update_duplicate_resolution(self, conn: sqlite3.Connection, resolution_results: List):
        """Update database with duplicate resolution results"""
        try:
            cursor = conn.cursor()
            
            for result in resolution_results:
                # Create duplicate group record
                group_id = f"dg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(result.primary_item))}"
                
                duplicate_law_ids = [item['law_id'] for item in result.duplicate_items]
                
                cursor.execute('''
                    INSERT INTO duplicate_groups 
                    (group_id, group_type, primary_law_id, duplicate_law_ids, 
                     resolution_strategy, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    group_id, 'content_semantic', result.primary_item['law_id'],
                    json.dumps(duplicate_law_ids, ensure_ascii=False),
                    result.resolution_strategy, result.confidence_score
                ))
                
                # Update primary item
                cursor.execute('''
                    UPDATE assembly_laws 
                    SET is_primary_version = TRUE, duplicate_group_id = ?
                    WHERE law_id = ?
                ''', (group_id, result.primary_item['law_id']))
                
                # Update duplicate items
                for i, duplicate_item in enumerate(result.duplicate_items):
                    cursor.execute('''
                        UPDATE assembly_laws 
                        SET is_primary_version = FALSE, duplicate_group_id = ?, version_number = ?
                        WHERE law_id = ?
                    ''', (group_id, i + 2, duplicate_item['law_id']))
            
            logger.info(f"Updated {len(resolution_results)} duplicate groups in database")
            
        except Exception as e:
            logger.error(f"Error updating duplicate resolution: {e}")
            raise
    
    def _update_migration_metadata(self, conn: sqlite3.Connection):
        """Update migration metadata for all records"""
        try:
            cursor = conn.cursor()
            
            migration_timestamp = datetime.now().isoformat()
            
            # Update all records with migration metadata
            cursor.execute('''
                UPDATE assembly_laws 
                SET migration_timestamp = ?, processing_version = '2.0'
                WHERE migration_timestamp IS NULL
            ''', (migration_timestamp,))
            
            updated_count = cursor.rowcount
            logger.info(f"Updated migration metadata for {updated_count} records")
            
        except Exception as e:
            logger.error(f"Error updating migration metadata: {e}")
            raise
    
    def _log_migration_summary(self):
        """Log migration summary"""
        duration = 0
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = (end - start).total_seconds() / 60.0
        
        logger.info("=== Data Migration Summary ===")
        logger.info(f"Total records: {self.stats['total_records']}")
        logger.info(f"Processed records: {self.stats['processed_records']}")
        logger.info(f"Updated records: {self.stats['updated_records']}")
        logger.info(f"Hashes calculated: {self.stats['hashes_calculated']}")
        logger.info(f"Quality scores calculated: {self.stats['quality_scores_calculated']}")
        logger.info(f"Duplicates found: {self.stats['duplicates_found']}")
        logger.info(f"Migration duration: {duration:.2f} minutes")


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Data Migration Script')
    
    parser.add_argument('--db-path', required=True, help='Path to database file')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if quality modules are available
    if not QUALITY_MODULES_AVAILABLE:
        print("Error: Quality modules not available. Please check installation.")
        sys.exit(1)
    
    # Run migration
    try:
        migration = DataMigration(args.db_path, args.batch_size)
        success = migration.migrate_data()
        
        if success:
            print("Data migration completed successfully!")
        else:
            print("Data migration failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        print(f"Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

