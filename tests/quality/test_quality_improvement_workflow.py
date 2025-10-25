#!/usr/bin/env python3
"""
End-to-End Tests for Complete Quality Improvement Workflow

This module provides comprehensive tests for the entire data quality improvement
system including all phases and components.

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    from scripts.data_processing.quality.data_quality_validator import DataQualityValidator, QualityReport
    from scripts.data_processing.quality.duplicate_detector import AdvancedDuplicateDetector, DuplicateGroup
    from scripts.data_processing.quality.duplicate_resolver import IntelligentDuplicateResolver, ResolutionResult
    from scripts.data_processing.quality.automated_data_cleaner import AutomatedDataCleaner, CleaningReport
    from scripts.data_processing.quality.real_time_quality_monitor import RealTimeQualityMonitor, QualityMetrics, QualityAlert
    from scripts.data_processing.quality.quality_reporting_dashboard import QualityReportingDashboard, QualityDashboardData, QualityReportData
    from scripts.data_processing.quality.scheduled_task_manager import ScheduledTaskManager
    from scripts.data_processing.auto_pipeline_orchestrator import AutoPipelineOrchestrator, PipelineResult
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDataQualityValidator(unittest.TestCase):
    """Test DataQualityValidator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        self.validator = DataQualityValidator()
        self.sample_law_data = {
            'law_name': '테스트 법률',
            'full_text': '제1조 (목적) 이 법은 테스트를 위한 법률이다.\n제2조 (정의) 이 법에서 사용하는 용어의 정의는 다음과 같다.',
            'articles': [
                {
                    'article_number': '제1조',
                    'article_title': '목적',
                    'article_content': '이 법은 테스트를 위한 법률이다.'
                },
                {
                    'article_number': '제2조',
                    'article_title': '정의',
                    'article_content': '이 법에서 사용하는 용어의 정의는 다음과 같다.'
                }
            ]
        }
    
    def test_validate_parsing_quality(self):
        """Test parsing quality validation"""
        report = self.validator.validate_parsing_quality(self.sample_law_data)
        
        self.assertIsInstance(report, QualityReport)
        self.assertIsInstance(report.overall_score, float)
        self.assertGreaterEqual(report.overall_score, 0.0)
        self.assertLessEqual(report.overall_score, 1.0)
        self.assertIsInstance(report.issues, list)
        self.assertIsInstance(report.suggestions, list)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        score = self.validator.calculate_quality_score(self.sample_law_data)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_suggest_improvements(self):
        """Test improvement suggestions"""
        suggestions = self.validator.suggest_improvements(self.sample_law_data)
        
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 0)


class TestAdvancedDuplicateDetector(unittest.TestCase):
    """Test AdvancedDuplicateDetector functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        self.detector = AdvancedDuplicateDetector()
        self.sample_files = [
            {'path': '/test/file1.json', 'size': 1000, 'hash': 'hash1'},
            {'path': '/test/file2.json', 'size': 1000, 'hash': 'hash1'},  # Duplicate
            {'path': '/test/file3.json', 'size': 2000, 'hash': 'hash2'}
        ]
    
    def test_detect_file_level_duplicates(self):
        """Test file-level duplicate detection"""
        duplicates = self.detector.detect_file_level_duplicates(self.sample_files)
        
        self.assertIsInstance(duplicates, list)
        for duplicate in duplicates:
            self.assertIsInstance(duplicate, DuplicateGroup)
            self.assertIsInstance(duplicate.group_id, str)
            self.assertIsInstance(duplicate.items, list)
            self.assertGreater(len(duplicate.items), 1)
    
    def test_detect_content_level_duplicates(self):
        """Test content-level duplicate detection"""
        sample_content = [
            {'id': '1', 'content': 'This is test content'},
            {'id': '2', 'content': 'This is test content'},  # Duplicate
            {'id': '3', 'content': 'This is different content'}
        ]
        
        duplicates = self.detector.detect_content_level_duplicates(sample_content)
        
        self.assertIsInstance(duplicates, list)
        for duplicate in duplicates:
            self.assertIsInstance(duplicate, DuplicateGroup)
    
    def test_get_detection_statistics(self):
        """Test detection statistics"""
        stats = self.detector.get_detection_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('similarity_threshold', stats)
        self.assertIn('detection_methods', stats)


class TestIntelligentDuplicateResolver(unittest.TestCase):
    """Test IntelligentDuplicateResolver functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        self.resolver = IntelligentDuplicateResolver()
        self.sample_duplicate_groups = [
            DuplicateGroup(
                group_id='test_group_1',
                items=[
                    {'id': '1', 'quality_score': 0.9, 'content': 'High quality content'},
                    {'id': '2', 'quality_score': 0.7, 'content': 'Lower quality content'}
                ],
                similarity_score=0.95,
                detection_method='content_similarity'
            )
        ]
    
    def test_resolve_duplicates(self):
        """Test duplicate resolution"""
        results = self.resolver.resolve_duplicates(self.sample_duplicate_groups, 'quality_based')
        
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, ResolutionResult)
            self.assertIsInstance(result.success, bool)
            self.assertIsInstance(result.primary_item, dict)
            self.assertIsInstance(result.resolved_items, list)
    
    def test_add_resolution_strategy(self):
        """Test adding custom resolution strategy"""
        custom_strategy = {
            'name': 'custom_strategy',
            'description': 'Custom resolution strategy',
            'scoring_function': lambda item: item.get('custom_score', 0.5)
        }
        
        self.resolver.add_resolution_strategy(custom_strategy)
        strategies = self.resolver.get_resolution_strategies()
        
        self.assertIn('custom_strategy', strategies)
    
    def test_export_resolution_report(self):
        """Test resolution report export"""
        results = self.resolver.resolve_duplicates(self.sample_duplicate_groups)
        report = self.resolver.export_resolution_report(results)
        
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('resolutions', report)


class TestAutomatedDataCleaner(unittest.TestCase):
    """Test AutomatedDataCleaner functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.cleaner = AutomatedDataCleaner(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_run_daily_cleaning(self):
        """Test daily cleaning routine"""
        report = self.cleaner.run_daily_cleaning()
        
        self.assertIsInstance(report, CleaningReport)
        self.assertIsInstance(report.operation_type, str)
        self.assertIsInstance(report.start_time, datetime)
        self.assertIsInstance(report.end_time, datetime)
        self.assertIsInstance(report.duration_seconds, float)
        self.assertIsInstance(report.records_processed, int)
        self.assertIsInstance(report.records_cleaned, int)
        self.assertIsInstance(report.duplicates_found, int)
        self.assertIsInstance(report.duplicates_resolved, int)
        self.assertIsInstance(report.quality_improvements, int)
        self.assertIsInstance(report.errors, list)
        self.assertIsInstance(report.warnings, list)
        self.assertIsInstance(report.summary, dict)
    
    def test_run_weekly_cleaning(self):
        """Test weekly cleaning routine"""
        report = self.cleaner.run_weekly_cleaning()
        
        self.assertIsInstance(report, CleaningReport)
        self.assertEqual(report.operation_type, 'weekly_cleaning')
    
    def test_run_monthly_audit(self):
        """Test monthly audit routine"""
        report = self.cleaner.run_monthly_audit()
        
        self.assertIsInstance(report, CleaningReport)
        self.assertEqual(report.operation_type, 'monthly_audit')
    
    def test_monitor_real_time_quality(self):
        """Test real-time quality monitoring"""
        metrics = self.cleaner.monitor_real_time_quality()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('timestamp', metrics)
    
    def test_get_cleaning_statistics(self):
        """Test cleaning statistics retrieval"""
        stats = self.cleaner.get_cleaning_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('cleaning_stats', stats)
        self.assertIn('config', stats)
        self.assertIn('modules_available', stats)


class TestRealTimeQualityMonitor(unittest.TestCase):
    """Test RealTimeQualityMonitor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.monitor = RealTimeQualityMonitor(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        # Test starting monitoring
        success = self.monitor.start_monitoring()
        self.assertIsInstance(success, bool)
        
        if success:
            # Test stopping monitoring
            stop_success = self.monitor.stop_monitoring()
            self.assertIsInstance(stop_success, bool)
    
    def test_get_current_metrics(self):
        """Test getting current metrics"""
        metrics = self.monitor.get_current_metrics()
        
        # Metrics might be None if no data is available
        if metrics is not None:
            self.assertIsInstance(metrics, QualityMetrics)
            self.assertIsInstance(metrics.timestamp, datetime)
            self.assertIsInstance(metrics.overall_quality_score, float)
    
    def test_get_metrics_history(self):
        """Test getting metrics history"""
        history = self.monitor.get_metrics_history(hours=24)
        
        self.assertIsInstance(history, list)
        for metric in history:
            self.assertIsInstance(metric, QualityMetrics)
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        alerts = self.monitor.get_active_alerts()
        
        self.assertIsInstance(alerts, list)
        for alert in alerts:
            self.assertIsInstance(alert, QualityAlert)
            self.assertIsInstance(alert.alert_id, str)
            self.assertIsInstance(alert.alert_type, str)
            self.assertIsInstance(alert.severity, str)
            self.assertIsInstance(alert.message, str)
    
    def test_get_monitoring_status(self):
        """Test getting monitoring status"""
        status = self.monitor.get_monitoring_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('is_monitoring', status)
        self.assertIn('monitoring_stats', status)
        self.assertIn('active_alerts_count', status)
        self.assertIn('metrics_history_size', status)


class TestQualityReportingDashboard(unittest.TestCase):
    """Test QualityReportingDashboard functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.dashboard = QualityReportingDashboard(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_get_dashboard_data(self):
        """Test getting dashboard data"""
        dashboard_data = self.dashboard.get_dashboard_data(force_refresh=True)
        
        self.assertIsInstance(dashboard_data, QualityDashboardData)
        self.assertIsInstance(dashboard_data.timestamp, datetime)
        self.assertIsInstance(dashboard_data.overall_quality_score, float)
        self.assertIsInstance(dashboard_data.quality_distribution, dict)
        self.assertIsInstance(dashboard_data.duplicate_statistics, dict)
        self.assertIsInstance(dashboard_data.recent_quality_trends, list)
        self.assertIsInstance(dashboard_data.active_alerts, list)
        self.assertIsInstance(dashboard_data.performance_metrics, dict)
        self.assertIsInstance(dashboard_data.recommendations, list)
    
    def test_generate_quality_report(self):
        """Test generating quality report"""
        report = self.dashboard.generate_quality_report('summary', period_days=7)
        
        self.assertIsInstance(report, QualityReportData)
        self.assertIsInstance(report.report_id, str)
        self.assertIsInstance(report.report_type, str)
        self.assertIsInstance(report.generation_time, datetime)
        self.assertIsInstance(report.period_start, datetime)
        self.assertIsInstance(report.period_end, datetime)
        self.assertIsInstance(report.summary, dict)
        self.assertIsInstance(report.detailed_metrics, dict)
        self.assertIsInstance(report.quality_analysis, dict)
        self.assertIsInstance(report.duplicate_analysis, dict)
        self.assertIsInstance(report.performance_analysis, dict)
        self.assertIsInstance(report.recommendations, list)
        self.assertIsInstance(report.charts_data, dict)
    
    def test_export_report_json(self):
        """Test exporting report as JSON"""
        report = self.dashboard.generate_quality_report('summary', period_days=7)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            output_path = self.dashboard.export_report(report, 'json', temp_path)
            
            self.assertIsInstance(output_path, str)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify JSON content
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            self.assertIn('report_id', exported_data)
            self.assertIn('report_type', exported_data)
            self.assertIn('summary', exported_data)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_csv(self):
        """Test exporting report as CSV"""
        report = self.dashboard.generate_quality_report('summary', period_days=7)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            output_path = self.dashboard.export_report(report, 'csv', temp_path)
            
            self.assertIsInstance(output_path, str)
            self.assertTrue(os.path.exists(output_path))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_report_html(self):
        """Test exporting report as HTML"""
        report = self.dashboard.generate_quality_report('summary', period_days=7)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            output_path = self.dashboard.export_report(report, 'html', temp_path)
            
            self.assertIsInstance(output_path, str)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.assertIn('<html>', html_content)
            self.assertIn('<head>', html_content)
            self.assertIn('<body>', html_content)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestScheduledTaskManager(unittest.TestCase):
    """Test ScheduledTaskManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.task_manager = ScheduledTaskManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_setup_scheduled_tasks(self):
        """Test setting up scheduled tasks"""
        self.task_manager.setup_scheduled_tasks()
        
        # Verify that tasks were set up (this is hard to test without running the scheduler)
        self.assertIsNotNone(self.task_manager.data_cleaner)
        self.assertIsNotNone(self.task_manager.quality_monitor)
    
    def test_execute_task_manually(self):
        """Test manual task execution"""
        result = self.task_manager.execute_task_manually('daily_cleaning')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('task', result)
    
    def test_get_task_status(self):
        """Test getting task status"""
        status = self.task_manager.get_task_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('running_tasks', status)
        self.assertIn('execution_stats', status)
        self.assertIn('recent_tasks', status)
        self.assertIn('scheduled_jobs', status)
        self.assertIn('next_run_times', status)


class TestAutoPipelineOrchestratorIntegration(unittest.TestCase):
    """Test AutoPipelineOrchestrator with quality integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary directories and database
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, 'test.db')
        
        # Create sample data structure
        self.raw_dir = os.path.join(self.temp_dir, 'raw')
        self.processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create sample law data file
        self.sample_law_file = os.path.join(self.processed_dir, 'sample_law.json')
        sample_data = {
            'law_name': '테스트 법률',
            'full_text': '제1조 (목적) 이 법은 테스트를 위한 법률이다.',
            'articles': [
                {
                    'article_number': '제1조',
                    'article_title': '목적',
                    'article_content': '이 법은 테스트를 위한 법률이다.'
                }
            ]
        }
        
        with open(self.sample_law_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        # Initialize orchestrator
        config = {
            'quality': {
                'enabled': True,
                'validation_threshold': 0.7
            }
        }
        
        self.orchestrator = AutoPipelineOrchestrator(
            config=config,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
            db_path=self.temp_db
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quality_validation_stage(self):
        """Test quality validation stage in pipeline"""
        # Test the quality validation stage directly
        processed_files = [Path(self.sample_law_file)]
        
        result = self.orchestrator._run_quality_validation_stage(processed_files)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('metrics', result)
        self.assertIn('improvements_made', result)
        self.assertIn('duplicates_resolved', result)
        self.assertIn('errors', result)
        
        # Verify that quality score was added to the file
        with open(self.sample_law_file, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
        
        self.assertIn('quality_score', updated_data)
        self.assertIn('quality_validated', updated_data)
        self.assertIn('quality_validation_timestamp', updated_data)
    
    def test_pipeline_result_with_quality(self):
        """Test pipeline result includes quality metrics"""
        # Create a mock pipeline result with quality data
        result = PipelineResult(
            success=True,
            total_files_detected=1,
            files_processed=1,
            vectors_added=0,
            laws_imported=0,
            processing_time=1.0,
            stage_results={'quality_validation': {'success': True}},
            error_messages=[],
            quality_metrics={'average_quality_score': 0.8},
            quality_improvements=1,
            duplicates_resolved=0
        )
        
        self.assertIsInstance(result.quality_metrics, dict)
        self.assertIsInstance(result.quality_improvements, int)
        self.assertIsInstance(result.duplicates_resolved, int)
        self.assertEqual(result.quality_improvements, 1)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end quality improvement workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not QUALITY_MODULES_AVAILABLE:
            self.skipTest("Quality modules not available")
        
        # Create temporary environment
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, 'test.db')
        
        # Create necessary directories
        os.makedirs(os.path.join(self.temp_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'reports'), exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_quality_workflow(self):
        """Test complete quality improvement workflow"""
        # Step 1: Initialize quality validator
        validator = DataQualityValidator()
        
        # Step 2: Create sample law data
        sample_law_data = {
            'law_name': '테스트 법률',
            'full_text': '제1조 (목적) 이 법은 테스트를 위한 법률이다.\n제2조 (정의) 이 법에서 사용하는 용어의 정의는 다음과 같다.',
            'articles': [
                {
                    'article_number': '제1조',
                    'article_title': '목적',
                    'article_content': '이 법은 테스트를 위한 법률이다.'
                },
                {
                    'article_number': '제2조',
                    'article_title': '정의',
                    'article_content': '이 법에서 사용하는 용어의 정의는 다음과 같다.'
                }
            ]
        }
        
        # Step 3: Validate quality
        quality_report = validator.validate_parsing_quality(sample_law_data)
        self.assertIsInstance(quality_report, QualityReport)
        
        quality_score = validator.calculate_quality_score(sample_law_data)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        
        # Step 4: Initialize automated data cleaner
        cleaner = AutomatedDataCleaner(self.temp_db)
        
        # Step 5: Run daily cleaning
        cleaning_report = cleaner.run_daily_cleaning()
        self.assertIsInstance(cleaning_report, CleaningReport)
        
        # Step 6: Initialize real-time quality monitor
        monitor = RealTimeQualityMonitor(self.temp_db)
        
        # Step 7: Get current metrics
        metrics = monitor.get_current_metrics()
        # Metrics might be None if no data is available
        
        # Step 8: Initialize quality reporting dashboard
        dashboard = QualityReportingDashboard(self.temp_db)
        
        # Step 9: Generate dashboard data
        dashboard_data = dashboard.get_dashboard_data(force_refresh=True)
        self.assertIsInstance(dashboard_data, QualityDashboardData)
        
        # Step 10: Generate quality report
        quality_report_data = dashboard.generate_quality_report('summary', period_days=7)
        self.assertIsInstance(quality_report_data, QualityReportData)
        
        # Step 11: Export report
        report_path = os.path.join(self.temp_dir, 'reports', 'test_report.json')
        exported_path = dashboard.export_report(quality_report_data, 'json', report_path)
        self.assertTrue(os.path.exists(exported_path))
        
        # Step 12: Initialize scheduled task manager
        task_manager = ScheduledTaskManager(self.temp_db)
        
        # Step 13: Setup scheduled tasks
        task_manager.setup_scheduled_tasks()
        
        # Step 14: Get task status
        status = task_manager.get_task_status()
        self.assertIsInstance(status, dict)
        
        # Verify all components are working together
        self.assertIsNotNone(validator)
        self.assertIsNotNone(cleaner)
        self.assertIsNotNone(monitor)
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(task_manager)


def run_all_tests():
    """Run all end-to-end tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataQualityValidator,
        TestAdvancedDuplicateDetector,
        TestIntelligentDuplicateResolver,
        TestAutomatedDataCleaner,
        TestRealTimeQualityMonitor,
        TestQualityReportingDashboard,
        TestScheduledTaskManager,
        TestAutoPipelineOrchestratorIntegration,
        TestEndToEndWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='End-to-End Tests for Quality Improvement Workflow')
    parser.add_argument('--test-class', type=str,
                       choices=['validator', 'detector', 'resolver', 'cleaner', 'monitor', 
                               'dashboard', 'task_manager', 'orchestrator', 'workflow', 'all'],
                       default='all',
                       help='Specific test class to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        if args.test_class == 'all':
            result = run_all_tests()
        else:
            # Run specific test class
            test_class_map = {
                'validator': TestDataQualityValidator,
                'detector': TestAdvancedDuplicateDetector,
                'resolver': TestIntelligentDuplicateResolver,
                'cleaner': TestAutomatedDataCleaner,
                'monitor': TestRealTimeQualityMonitor,
                'dashboard': TestQualityReportingDashboard,
                'task_manager': TestScheduledTaskManager,
                'orchestrator': TestAutoPipelineOrchestratorIntegration,
                'workflow': TestEndToEndWorkflow
            }
            
            test_class = test_class_map[args.test_class]
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
            result = runner.run(suite)
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        print("="*50)
        
        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

