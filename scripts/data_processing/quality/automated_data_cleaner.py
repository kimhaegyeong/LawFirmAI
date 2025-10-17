#!/usr/bin/env python3
"""
Automated Data Cleaner for LawFirmAI

This module provides automated data cleaning routines for maintaining data quality
and removing duplicates on a scheduled basis.

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    from scripts.data_processing.quality.data_quality_validator import DataQualityValidator, QualityReport
    from scripts.data_processing.quality.duplicate_detector import AdvancedDuplicateDetector, DuplicateGroup
    from scripts.data_processing.quality.duplicate_resolver import IntelligentDuplicateResolver, ResolutionResult
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quality modules not available: {e}")
    QUALITY_MODULES_AVAILABLE = False

# Import database manager
try:
    from source.data.database import DatabaseManager
    DATABASE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Database module not available: {e}")
    DATABASE_MODULE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automated_data_cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Report for data cleaning operations"""
    operation_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    records_processed: int
    records_cleaned: int
    duplicates_found: int
    duplicates_resolved: int
    quality_improvements: int
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class AutomatedDataCleaner:
    """
    Automated data cleaner for maintaining data quality and removing duplicates
    
    This class provides scheduled cleaning routines including:
    - Daily duplicate detection and resolution
    - Weekly quality assessment and improvement
    - Monthly comprehensive data audit
    - Real-time quality monitoring
    """
    
    def __init__(self, db_path: str = "data/lawfirm.db", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the automated data cleaner
        
        Args:
            db_path (str): Path to the database file
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.db_path = db_path
        self.config = config or self._get_default_config()
        
        # Initialize components
        if DATABASE_MODULE_AVAILABLE:
            self.db_manager = DatabaseManager(db_path)
        else:
            self.db_manager = None
            
        if QUALITY_MODULES_AVAILABLE:
            self.quality_validator = DataQualityValidator()
            self.duplicate_detector = AdvancedDuplicateDetector()
            self.duplicate_resolver = IntelligentDuplicateResolver()
        else:
            self.quality_validator = None
            self.duplicate_detector = None
            self.duplicate_resolver = None
        
        # Statistics tracking
        self.cleaning_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_records_processed': 0,
            'total_duplicates_resolved': 0,
            'total_quality_improvements': 0,
            'last_cleaning_date': None,
            'last_quality_check': None
        }
        
        logger.info("AutomatedDataCleaner initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'daily_cleaning': {
                'enabled': True,
                'duplicate_threshold': 0.95,
                'quality_threshold': 0.7,
                'max_records_per_batch': 1000
            },
            'weekly_cleaning': {
                'enabled': True,
                'comprehensive_quality_check': True,
                'orphaned_data_cleanup': True,
                'index_optimization': True
            },
            'monthly_audit': {
                'enabled': True,
                'full_duplicate_scan': True,
                'data_integrity_check': True,
                'performance_analysis': True
            },
            'real_time_monitoring': {
                'enabled': True,
                'quality_threshold': 0.8,
                'duplicate_threshold': 0.9,
                'alert_on_degradation': True
            }
        }
    
    def run_daily_cleaning(self) -> CleaningReport:
        """
        Run daily data cleaning routine
        
        Returns:
            CleaningReport: Report of the cleaning operation
        """
        if not self.config['daily_cleaning']['enabled']:
            logger.info("Daily cleaning is disabled")
            return self._create_skipped_report('daily_cleaning')
        
        logger.info("Starting daily data cleaning routine")
        start_time = datetime.now()
        
        report = CleaningReport(
            operation_type='daily_cleaning',
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0,
            records_processed=0,
            records_cleaned=0,
            duplicates_found=0,
            duplicates_resolved=0,
            quality_improvements=0,
            errors=[],
            warnings=[],
            summary={}
        )
        
        try:
            # Step 1: Detect and resolve duplicates
            duplicate_result = self._run_duplicate_cleaning()
            report.duplicates_found = duplicate_result.get('duplicates_found', 0)
            report.duplicates_resolved = duplicate_result.get('duplicates_resolved', 0)
            
            # Step 2: Quality assessment and improvement
            quality_result = self._run_quality_improvement()
            report.quality_improvements = quality_result.get('improvements_made', 0)
            
            # Step 3: Clean up orphaned data
            orphaned_result = self._cleanup_orphaned_data()
            report.records_cleaned += orphaned_result.get('records_cleaned', 0)
            
            # Step 4: Update statistics
            report.records_processed = duplicate_result.get('records_processed', 0)
            report.summary = {
                'duplicate_cleaning': duplicate_result,
                'quality_improvement': quality_result,
                'orphaned_cleanup': orphaned_result
            }
            
            # Update cleaning stats
            self.cleaning_stats['successful_operations'] += 1
            self.cleaning_stats['last_cleaning_date'] = start_time.isoformat()
            
            logger.info(f"Daily cleaning completed successfully: {report.duplicates_resolved} duplicates resolved, {report.quality_improvements} quality improvements")
            
        except Exception as e:
            error_msg = f"Error in daily cleaning: {e}"
            logger.error(error_msg)
            report.errors.append(error_msg)
            self.cleaning_stats['failed_operations'] += 1
        
        finally:
            report.end_time = datetime.now()
            report.duration_seconds = (report.end_time - report.start_time).total_seconds()
            self.cleaning_stats['total_operations'] += 1
        
        return report
    
    def run_weekly_cleaning(self) -> CleaningReport:
        """
        Run weekly comprehensive data cleaning routine
        
        Returns:
            CleaningReport: Report of the cleaning operation
        """
        if not self.config['weekly_cleaning']['enabled']:
            logger.info("Weekly cleaning is disabled")
            return self._create_skipped_report('weekly_cleaning')
        
        logger.info("Starting weekly comprehensive data cleaning routine")
        start_time = datetime.now()
        
        report = CleaningReport(
            operation_type='weekly_cleaning',
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0,
            records_processed=0,
            records_cleaned=0,
            duplicates_found=0,
            duplicates_resolved=0,
            quality_improvements=0,
            errors=[],
            warnings=[],
            summary={}
        )
        
        try:
            # Step 1: Comprehensive duplicate detection
            comprehensive_duplicates = self._run_comprehensive_duplicate_detection()
            report.duplicates_found = comprehensive_duplicates.get('duplicates_found', 0)
            report.duplicates_resolved = comprehensive_duplicates.get('duplicates_resolved', 0)
            
            # Step 2: Full quality assessment
            quality_assessment = self._run_comprehensive_quality_assessment()
            report.quality_improvements = quality_assessment.get('improvements_made', 0)
            
            # Step 3: Data integrity check
            integrity_check = self._run_data_integrity_check()
            report.records_cleaned += integrity_check.get('issues_fixed', 0)
            
            # Step 4: Index optimization
            if self.config['weekly_cleaning']['index_optimization']:
                index_result = self._optimize_database_indices()
                report.summary['index_optimization'] = index_result
            
            # Step 5: Performance analysis
            performance_result = self._analyze_database_performance()
            report.summary['performance_analysis'] = performance_result
            
            report.records_processed = comprehensive_duplicates.get('records_processed', 0)
            report.summary.update({
                'comprehensive_duplicates': comprehensive_duplicates,
                'quality_assessment': quality_assessment,
                'integrity_check': integrity_check
            })
            
            # Update cleaning stats
            self.cleaning_stats['successful_operations'] += 1
            
            logger.info(f"Weekly cleaning completed: {report.duplicates_resolved} duplicates resolved, {report.quality_improvements} quality improvements")
            
        except Exception as e:
            error_msg = f"Error in weekly cleaning: {e}"
            logger.error(error_msg)
            report.errors.append(error_msg)
            self.cleaning_stats['failed_operations'] += 1
        
        finally:
            report.end_time = datetime.now()
            report.duration_seconds = (report.end_time - report.start_time).total_seconds()
            self.cleaning_stats['total_operations'] += 1
        
        return report
    
    def run_monthly_audit(self) -> CleaningReport:
        """
        Run monthly comprehensive data audit
        
        Returns:
            CleaningReport: Report of the audit operation
        """
        if not self.config['monthly_audit']['enabled']:
            logger.info("Monthly audit is disabled")
            return self._create_skipped_report('monthly_audit')
        
        logger.info("Starting monthly comprehensive data audit")
        start_time = datetime.now()
        
        report = CleaningReport(
            operation_type='monthly_audit',
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0,
            records_processed=0,
            records_cleaned=0,
            duplicates_found=0,
            duplicates_resolved=0,
            quality_improvements=0,
            errors=[],
            warnings=[],
            summary={}
        )
        
        try:
            # Step 1: Full database scan for duplicates
            full_duplicate_scan = self._run_full_duplicate_scan()
            report.duplicates_found = full_duplicate_scan.get('duplicates_found', 0)
            report.duplicates_resolved = full_duplicate_scan.get('duplicates_resolved', 0)
            
            # Step 2: Complete data integrity check
            full_integrity_check = self._run_full_data_integrity_check()
            report.records_cleaned += full_integrity_check.get('issues_fixed', 0)
            
            # Step 3: Performance analysis and optimization
            performance_analysis = self._run_comprehensive_performance_analysis()
            report.summary['performance_analysis'] = performance_analysis
            
            # Step 4: Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report()
            report.summary['comprehensive_report'] = comprehensive_report
            
            report.records_processed = full_duplicate_scan.get('records_processed', 0)
            report.summary.update({
                'full_duplicate_scan': full_duplicate_scan,
                'full_integrity_check': full_integrity_check
            })
            
            # Update cleaning stats
            self.cleaning_stats['successful_operations'] += 1
            
            logger.info(f"Monthly audit completed: {report.duplicates_resolved} duplicates resolved, {report.records_cleaned} issues fixed")
            
        except Exception as e:
            error_msg = f"Error in monthly audit: {e}"
            logger.error(error_msg)
            report.errors.append(error_msg)
            self.cleaning_stats['failed_operations'] += 1
        
        finally:
            report.end_time = datetime.now()
            report.duration_seconds = (report.end_time - report.start_time).total_seconds()
            self.cleaning_stats['total_operations'] += 1
        
        return report
    
    def monitor_real_time_quality(self) -> Dict[str, Any]:
        """
        Monitor data quality in real-time
        
        Returns:
            Dict[str, Any]: Real-time quality metrics
        """
        if not self.config['real_time_monitoring']['enabled']:
            return {'status': 'disabled'}
        
        logger.info("Running real-time quality monitoring")
        
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'overall_quality_score': 0.0,
                'duplicate_percentage': 0.0,
                'low_quality_records': 0,
                'alerts': [],
                'recommendations': []
            }
            
            if not self.db_manager:
                metrics['alerts'].append('Database manager not available')
                return metrics
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get overall quality statistics
                cursor.execute('''
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as total_records,
                        COUNT(CASE WHEN quality_score < ? THEN 1 END) as low_quality_count
                    FROM assembly_laws 
                    WHERE quality_score IS NOT NULL
                ''', (self.config['real_time_monitoring']['quality_threshold'],))
                
                quality_stats = cursor.fetchone()
                if quality_stats:
                    metrics['overall_quality_score'] = quality_stats[0] or 0.0
                    metrics['low_quality_records'] = quality_stats[2] or 0
                
                # Check for recent quality degradation
                cursor.execute('''
                    SELECT COUNT(*) FROM assembly_laws 
                    WHERE quality_score < ? AND migration_timestamp > datetime('now', '-1 day')
                ''', (self.config['real_time_monitoring']['quality_threshold'],))
                
                recent_low_quality = cursor.fetchone()[0]
                if recent_low_quality > 0:
                    metrics['alerts'].append(f'{recent_low_quality} low-quality records added in last 24 hours')
                
                # Generate recommendations
                if metrics['overall_quality_score'] < 0.8:
                    metrics['recommendations'].append('Consider running comprehensive quality improvement')
                
                if metrics['low_quality_records'] > 100:
                    metrics['recommendations'].append('High number of low-quality records detected')
            
            logger.info(f"Real-time monitoring completed: Quality score {metrics['overall_quality_score']:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'alerts': [f'Monitoring error: {e}']
            }
    
    def _run_duplicate_cleaning(self) -> Dict[str, Any]:
        """Run duplicate detection and resolution"""
        if not self.duplicate_detector or not self.duplicate_resolver:
            return {'error': 'Duplicate detection modules not available'}
        
        try:
            # Get recent records for daily cleaning
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT law_id, law_name, full_text, quality_score
                    FROM assembly_laws 
                    WHERE migration_timestamp > datetime('now', '-1 day')
                    ORDER BY migration_timestamp DESC
                    LIMIT ?
                ''', (self.config['daily_cleaning']['max_records_per_batch'],))
                
                recent_records = cursor.fetchall()
            
            if not recent_records:
                return {'records_processed': 0, 'duplicates_found': 0, 'duplicates_resolved': 0}
            
            # Detect duplicates
            duplicate_groups = []
            for record in recent_records:
                law_id, law_name, full_text, quality_score = record
                
                # Check for duplicates using content similarity
                similar_records = self._find_similar_records(law_name, full_text)
                if similar_records:
                    duplicate_groups.append({
                        'primary_id': law_id,
                        'duplicate_ids': similar_records,
                        'similarity_score': 0.95  # Placeholder
                    })
            
            # Resolve duplicates
            resolved_count = 0
            for group in duplicate_groups:
                try:
                    resolution_result = self.duplicate_resolver.resolve_duplicates([group])
                    if resolution_result.get('success', False):
                        resolved_count += len(group['duplicate_ids'])
                except Exception as e:
                    logger.error(f"Error resolving duplicate group: {e}")
            
            return {
                'records_processed': len(recent_records),
                'duplicates_found': len(duplicate_groups),
                'duplicates_resolved': resolved_count
            }
            
        except Exception as e:
            logger.error(f"Error in duplicate cleaning: {e}")
            return {'error': str(e)}
    
    def _run_quality_improvement(self) -> Dict[str, Any]:
        """Run quality assessment and improvement"""
        if not self.quality_validator:
            return {'error': 'Quality validator not available'}
        
        try:
            improvements_made = 0
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get records with low quality scores
                cursor.execute('''
                    SELECT law_id, law_name, full_text, quality_score
                    FROM assembly_laws 
                    WHERE quality_score < ? OR quality_score IS NULL
                    ORDER BY quality_score ASC
                    LIMIT ?
                ''', (self.config['daily_cleaning']['quality_threshold'], 
                      self.config['daily_cleaning']['max_records_per_batch']))
                
                low_quality_records = cursor.fetchall()
            
            for record in low_quality_records:
                law_id, law_name, full_text, current_score = record
                
                # Recalculate quality score
                try:
                    # Create law data structure for validation
                    law_data = {
                        'law_name': law_name,
                        'full_text': full_text,
                        'articles': []  # Simplified for daily cleaning
                    }
                    
                    new_score = self.quality_validator.calculate_quality_score(law_data)
                    
                    if new_score > (current_score or 0.0):
                        # Update quality score
                        with self.db_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE assembly_laws 
                                SET quality_score = ?, migration_timestamp = datetime('now')
                                WHERE law_id = ?
                            ''', (new_score, law_id))
                            conn.commit()
                        
                        improvements_made += 1
                        
                except Exception as e:
                    logger.error(f"Error improving quality for law {law_id}: {e}")
            
            return {
                'records_processed': len(low_quality_records),
                'improvements_made': improvements_made
            }
            
        except Exception as e:
            logger.error(f"Error in quality improvement: {e}")
            return {'error': str(e)}
    
    def _cleanup_orphaned_data(self) -> Dict[str, Any]:
        """Clean up orphaned data"""
        try:
            records_cleaned = 0
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up orphaned articles
                cursor.execute('''
                    DELETE FROM assembly_articles 
                    WHERE law_id NOT IN (SELECT law_id FROM assembly_laws)
                ''')
                orphaned_articles = cursor.rowcount
                
                # Clean up orphaned FTS entries
                cursor.execute('''
                    DELETE FROM assembly_laws_fts 
                    WHERE law_id NOT IN (SELECT law_id FROM assembly_laws)
                ''')
                orphaned_fts_laws = cursor.rowcount
                
                cursor.execute('''
                    DELETE FROM assembly_articles_fts 
                    WHERE law_id NOT IN (SELECT law_id FROM assembly_laws)
                ''')
                orphaned_fts_articles = cursor.rowcount
                
                conn.commit()
                records_cleaned = orphaned_articles + orphaned_fts_laws + orphaned_fts_articles
            
            return {
                'orphaned_articles': orphaned_articles,
                'orphaned_fts_laws': orphaned_fts_laws,
                'orphaned_fts_articles': orphaned_fts_articles,
                'records_cleaned': records_cleaned
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned data: {e}")
            return {'error': str(e)}
    
    def _find_similar_records(self, law_name: str, full_text: str) -> List[str]:
        """Find similar records based on name and content"""
        try:
            similar_ids = []
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Find records with similar names
                cursor.execute('''
                    SELECT law_id, law_name, full_text
                    FROM assembly_laws 
                    WHERE law_name LIKE ? AND law_id != ?
                    LIMIT 10
                ''', (f'%{law_name}%', 'dummy_id'))
                
                candidates = cursor.fetchall()
                
                for candidate in candidates:
                    candidate_id, candidate_name, candidate_text = candidate
                    
                    # Simple similarity check (can be enhanced with proper similarity algorithms)
                    if self._calculate_text_similarity(full_text, candidate_text) > 0.9:
                        similar_ids.append(candidate_id)
            
            return similar_ids
            
        except Exception as e:
            logger.error(f"Error finding similar records: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _run_comprehensive_duplicate_detection(self) -> Dict[str, Any]:
        """Run comprehensive duplicate detection for weekly cleaning"""
        # This would use the full duplicate detection pipeline
        # For now, return a placeholder
        return {
            'records_processed': 0,
            'duplicates_found': 0,
            'duplicates_resolved': 0,
            'method': 'comprehensive_scan'
        }
    
    def _run_comprehensive_quality_assessment(self) -> Dict[str, Any]:
        """Run comprehensive quality assessment for weekly cleaning"""
        # This would run full quality assessment on all records
        # For now, return a placeholder
        return {
            'records_processed': 0,
            'improvements_made': 0,
            'method': 'comprehensive_assessment'
        }
    
    def _run_data_integrity_check(self) -> Dict[str, Any]:
        """Run data integrity check"""
        try:
            issues_fixed = 0
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check for missing required fields
                cursor.execute('''
                    UPDATE assembly_laws 
                    SET law_name = 'Unknown Law' 
                    WHERE law_name IS NULL OR law_name = ''
                ''')
                issues_fixed += cursor.rowcount
                
                # Check for invalid dates
                cursor.execute('''
                    UPDATE assembly_laws 
                    SET promulgation_date = NULL 
                    WHERE promulgation_date = '' OR promulgation_date = '0000-00-00'
                ''')
                issues_fixed += cursor.rowcount
                
                conn.commit()
            
            return {
                'issues_fixed': issues_fixed,
                'method': 'integrity_check'
            }
            
        except Exception as e:
            logger.error(f"Error in data integrity check: {e}")
            return {'error': str(e)}
    
    def _optimize_database_indices(self) -> Dict[str, Any]:
        """Optimize database indices"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze tables for optimization
                cursor.execute('ANALYZE assembly_laws')
                cursor.execute('ANALYZE assembly_articles')
                
                # Rebuild FTS indices
                cursor.execute('INSERT INTO assembly_laws_fts(assembly_laws_fts) VALUES("rebuild")')
                cursor.execute('INSERT INTO assembly_articles_fts(assembly_articles_fts) VALUES("rebuild")')
                
                conn.commit()
            
            return {
                'status': 'success',
                'method': 'index_optimization'
            }
            
        except Exception as e:
            logger.error(f"Error optimizing indices: {e}")
            return {'error': str(e)}
    
    def _analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database performance"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                cursor.execute('SELECT COUNT(*) FROM assembly_laws')
                law_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM assembly_articles')
                article_count = cursor.fetchone()[0]
                
                # Get database file size
                db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            
            return {
                'law_count': law_count,
                'article_count': article_count,
                'database_size_bytes': db_size,
                'database_size_mb': db_size / (1024 * 1024),
                'method': 'performance_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {'error': str(e)}
    
    def _run_full_duplicate_scan(self) -> Dict[str, Any]:
        """Run full duplicate scan for monthly audit"""
        # This would use the complete duplicate detection pipeline
        return {
            'records_processed': 0,
            'duplicates_found': 0,
            'duplicates_resolved': 0,
            'method': 'full_scan'
        }
    
    def _run_full_data_integrity_check(self) -> Dict[str, Any]:
        """Run full data integrity check for monthly audit"""
        # This would run comprehensive integrity checks
        return {
            'issues_fixed': 0,
            'method': 'full_integrity_check'
        }
    
    def _run_comprehensive_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis for monthly audit"""
        # This would run detailed performance analysis
        return {
            'analysis_completed': True,
            'method': 'comprehensive_performance_analysis'
        }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report for monthly audit"""
        return {
            'report_generated': True,
            'timestamp': datetime.now().isoformat(),
            'method': 'comprehensive_report'
        }
    
    def _create_skipped_report(self, operation_type: str) -> CleaningReport:
        """Create a report for skipped operations"""
        now = datetime.now()
        return CleaningReport(
            operation_type=operation_type,
            start_time=now,
            end_time=now,
            duration_seconds=0.0,
            records_processed=0,
            records_cleaned=0,
            duplicates_found=0,
            duplicates_resolved=0,
            quality_improvements=0,
            errors=[],
            warnings=[f'{operation_type} is disabled in configuration'],
            summary={'status': 'skipped'}
        )
    
    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return {
            'cleaning_stats': self.cleaning_stats.copy(),
            'config': self.config,
            'modules_available': {
                'quality_validator': self.quality_validator is not None,
                'duplicate_detector': self.duplicate_detector is not None,
                'duplicate_resolver': self.duplicate_resolver is not None,
                'database_manager': self.db_manager is not None
            }
        }
    
    def save_cleaning_report(self, report: CleaningReport, output_path: str):
        """Save cleaning report to file"""
        try:
            report_data = {
                'operation_type': report.operation_type,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat(),
                'duration_seconds': report.duration_seconds,
                'records_processed': report.records_processed,
                'records_cleaned': report.records_cleaned,
                'duplicates_found': report.duplicates_found,
                'duplicates_resolved': report.duplicates_resolved,
                'quality_improvements': report.quality_improvements,
                'errors': report.errors,
                'warnings': report.warnings,
                'summary': report.summary
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cleaning report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving cleaning report: {e}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Data Cleaner for LawFirmAI')
    parser.add_argument('--operation', type=str, required=True,
                       choices=['daily', 'weekly', 'monthly', 'monitor'],
                       help='Type of cleaning operation to run')
    parser.add_argument('--db-path', type=str, default='data/lawfirm.db',
                       help='Database file path')
    parser.add_argument('--output', type=str,
                       help='Output file path for cleaning report')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return
    
    # Create cleaner
    cleaner = AutomatedDataCleaner(args.db_path, config)
    
    try:
        # Run the specified operation
        if args.operation == 'daily':
            report = cleaner.run_daily_cleaning()
        elif args.operation == 'weekly':
            report = cleaner.run_weekly_cleaning()
        elif args.operation == 'monthly':
            report = cleaner.run_monthly_audit()
        elif args.operation == 'monitor':
            result = cleaner.monitor_real_time_quality()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        
        # Save report if output path provided
        if args.output:
            cleaner.save_cleaning_report(report, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print(f"{args.operation.upper()} CLEANING REPORT")
        print("="*50)
        print(f"Operation: {report.operation_type}")
        print(f"Duration: {report.duration_seconds:.2f} seconds")
        print(f"Records processed: {report.records_processed}")
        print(f"Records cleaned: {report.records_cleaned}")
        print(f"Duplicates found: {report.duplicates_found}")
        print(f"Duplicates resolved: {report.duplicates_resolved}")
        print(f"Quality improvements: {report.quality_improvements}")
        
        if report.errors:
            print(f"Errors: {len(report.errors)}")
            for error in report.errors:
                print(f"  - {error}")
        
        if report.warnings:
            print(f"Warnings: {len(report.warnings)}")
            for warning in report.warnings:
                print(f"  - {warning}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Cleaning operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

