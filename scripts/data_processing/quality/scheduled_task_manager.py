#!/usr/bin/env python3
"""
Scheduled Task Scripts for LawFirmAI Data Quality Management

This module provides scheduled task execution for automated data cleaning
and quality monitoring operations.

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Try to import schedule module
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logging.warning("schedule module not available. Install with: pip install schedule")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    from scripts.data_processing.quality.automated_data_cleaner import AutomatedDataCleaner
    from scripts.data_processing.quality.real_time_quality_monitor import RealTimeQualityMonitor
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quality modules not available: {e}")
    QUALITY_MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_tasks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScheduledTaskManager:
    """
    Manager for scheduled data quality tasks
    
    This class provides:
    - Daily data cleaning tasks
    - Weekly comprehensive cleaning
    - Monthly audits
    - Real-time monitoring
    - Task execution logging
    - Error handling and recovery
    """
    
    def __init__(self, db_path: str = "data/lawfirm.db", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scheduled task manager
        
        Args:
            db_path (str): Path to the database file
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.db_path = db_path
        self.config = config or self._get_default_config()
        
        # Initialize components
        if QUALITY_MODULES_AVAILABLE:
            self.data_cleaner = AutomatedDataCleaner(db_path, self.config.get('data_cleaner', {}))
            self.quality_monitor = RealTimeQualityMonitor(db_path, self.config.get('quality_monitor', {}))
        else:
            self.data_cleaner = None
            self.quality_monitor = None
        
        # Task execution tracking
        self.task_history: List[Dict[str, Any]] = []
        self.running_tasks: Dict[str, bool] = {}
        
        # Statistics
        self.execution_stats = {
            'total_tasks_executed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'last_execution': None,
            'start_time': datetime.now()
        }
        
        logger.info("ScheduledTaskManager initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'scheduling': {
                'daily_cleaning_time': '02:00',  # 2 AM
                'weekly_cleaning_day': 'sunday',
                'weekly_cleaning_time': '03:00',  # 3 AM
                'monthly_audit_day': 1,  # 1st of month
                'monthly_audit_time': '04:00',  # 4 AM
                'real_time_monitoring': True
            },
            'data_cleaner': {
                'daily_cleaning': {'enabled': True},
                'weekly_cleaning': {'enabled': True},
                'monthly_audit': {'enabled': True}
            },
            'quality_monitor': {
                'monitoring': {'enabled': True, 'check_interval_seconds': 300}
            },
            'logging': {
                'enable_task_logging': True,
                'log_retention_days': 30,
                'enable_performance_logging': True
            },
            'notifications': {
                'enable_email_notifications': False,
                'enable_webhook_notifications': False,
                'notification_webhook_url': None
            }
        }
    
    def setup_scheduled_tasks(self):
        """Setup all scheduled tasks"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("Schedule module not available. Tasks cannot be scheduled automatically.")
            logger.info("Install schedule module with: pip install schedule")
            return False
            
        try:
            # Clear existing schedules
            schedule.clear()
            
            # Daily cleaning task
            if self.config['scheduling']['daily_cleaning_time']:
                schedule.every().day.at(self.config['scheduling']['daily_cleaning_time']).do(
                    self._execute_daily_cleaning
                )
                logger.info(f"Daily cleaning scheduled for {self.config['scheduling']['daily_cleaning_time']}")
            
            # Weekly cleaning task
            if self.config['scheduling']['weekly_cleaning_day'] and self.config['scheduling']['weekly_cleaning_time']:
                getattr(schedule.every(), self.config['scheduling']['weekly_cleaning_day']).at(
                    self.config['scheduling']['weekly_cleaning_time']
                ).do(self._execute_weekly_cleaning)
                logger.info(f"Weekly cleaning scheduled for {self.config['scheduling']['weekly_cleaning_day']} at {self.config['scheduling']['weekly_cleaning_time']}")
            
            # Monthly audit task
            if self.config['scheduling']['monthly_audit_day'] and self.config['scheduling']['monthly_audit_time']:
                schedule.every().month.do(self._execute_monthly_audit)
                logger.info(f"Monthly audit scheduled for day {self.config['scheduling']['monthly_audit_day']} at {self.config['scheduling']['monthly_audit_time']}")
            
            # Start real-time monitoring if enabled
            if self.config['scheduling']['real_time_monitoring'] and self.quality_monitor:
                self.quality_monitor.start_monitoring()
                logger.info("Real-time quality monitoring started")
            
            logger.info("All scheduled tasks setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up scheduled tasks: {e}")
            raise
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        if not SCHEDULE_AVAILABLE:
            logger.warning("Schedule module not available. Cannot run scheduler.")
            logger.info("Install schedule module with: pip install schedule")
            return False
            
        logger.info("Starting scheduled task scheduler")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self._cleanup()
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            self._cleanup()
            raise
    
    def _execute_daily_cleaning(self):
        """Execute daily cleaning task"""
        task_name = 'daily_cleaning'
        
        if self.running_tasks.get(task_name, False):
            logger.warning(f"Task {task_name} is already running, skipping")
            return
        
        self.running_tasks[task_name] = True
        start_time = datetime.now()
        
        try:
            logger.info("Starting daily cleaning task")
            
            if not self.data_cleaner:
                raise Exception("Data cleaner not available")
            
            # Execute daily cleaning
            report = self.data_cleaner.run_daily_cleaning()
            
            # Log task execution
            self._log_task_execution(task_name, start_time, True, report)
            
            # Send notifications if enabled
            self._send_task_notification(task_name, report, success=True)
            
            logger.info(f"Daily cleaning completed: {report.duplicates_resolved} duplicates resolved, {report.quality_improvements} quality improvements")
            
        except Exception as e:
            error_msg = f"Error in daily cleaning: {e}"
            logger.error(error_msg)
            self._log_task_execution(task_name, start_time, False, {'error': error_msg})
            self._send_task_notification(task_name, {'error': error_msg}, success=False)
            
        finally:
            self.running_tasks[task_name] = False
    
    def _execute_weekly_cleaning(self):
        """Execute weekly cleaning task"""
        task_name = 'weekly_cleaning'
        
        if self.running_tasks.get(task_name, False):
            logger.warning(f"Task {task_name} is already running, skipping")
            return
        
        self.running_tasks[task_name] = True
        start_time = datetime.now()
        
        try:
            logger.info("Starting weekly cleaning task")
            
            if not self.data_cleaner:
                raise Exception("Data cleaner not available")
            
            # Execute weekly cleaning
            report = self.data_cleaner.run_weekly_cleaning()
            
            # Log task execution
            self._log_task_execution(task_name, start_time, True, report)
            
            # Send notifications if enabled
            self._send_task_notification(task_name, report, success=True)
            
            logger.info(f"Weekly cleaning completed: {report.duplicates_resolved} duplicates resolved, {report.quality_improvements} quality improvements")
            
        except Exception as e:
            error_msg = f"Error in weekly cleaning: {e}"
            logger.error(error_msg)
            self._log_task_execution(task_name, start_time, False, {'error': error_msg})
            self._send_task_notification(task_name, {'error': error_msg}, success=False)
            
        finally:
            self.running_tasks[task_name] = False
    
    def _execute_monthly_audit(self):
        """Execute monthly audit task"""
        task_name = 'monthly_audit'
        
        if self.running_tasks.get(task_name, False):
            logger.warning(f"Task {task_name} is already running, skipping")
            return
        
        self.running_tasks[task_name] = True
        start_time = datetime.now()
        
        try:
            logger.info("Starting monthly audit task")
            
            if not self.data_cleaner:
                raise Exception("Data cleaner not available")
            
            # Execute monthly audit
            report = self.data_cleaner.run_monthly_audit()
            
            # Log task execution
            self._log_task_execution(task_name, start_time, True, report)
            
            # Send notifications if enabled
            self._send_task_notification(task_name, report, success=True)
            
            logger.info(f"Monthly audit completed: {report.duplicates_resolved} duplicates resolved, {report.records_cleaned} issues fixed")
            
        except Exception as e:
            error_msg = f"Error in monthly audit: {e}"
            logger.error(error_msg)
            self._log_task_execution(task_name, start_time, False, {'error': error_msg})
            self._send_task_notification(task_name, {'error': error_msg}, success=False)
            
        finally:
            self.running_tasks[task_name] = False
    
    def _log_task_execution(self, task_name: str, start_time: datetime, success: bool, result: Any):
        """Log task execution"""
        try:
            execution_log = {
                'task_name': task_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'success': success,
                'result': result if isinstance(result, dict) else str(result)
            }
            
            self.task_history.append(execution_log)
            
            # Limit history size
            max_history = 1000
            if len(self.task_history) > max_history:
                self.task_history = self.task_history[-max_history:]
            
            # Update statistics
            self.execution_stats['total_tasks_executed'] += 1
            if success:
                self.execution_stats['successful_tasks'] += 1
            else:
                self.execution_stats['failed_tasks'] += 1
            self.execution_stats['last_execution'] = datetime.now().isoformat()
            
            # Save to file if enabled
            if self.config['logging']['enable_task_logging']:
                self._save_task_log(execution_log)
                
        except Exception as e:
            logger.error(f"Error logging task execution: {e}")
    
    def _save_task_log(self, execution_log: Dict[str, Any]):
        """Save task log to file"""
        try:
            log_dir = Path('logs/scheduled_tasks')
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"task_execution_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing logs
            existing_logs = []
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            # Add new log
            existing_logs.append(execution_log)
            
            # Save updated logs
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving task log: {e}")
    
    def _send_task_notification(self, task_name: str, result: Any, success: bool):
        """Send task notification"""
        try:
            if not self.config['notifications']['enable_email_notifications'] and \
               not self.config['notifications']['enable_webhook_notifications']:
                return
            
            notification = {
                'task_name': task_name,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'result': result if isinstance(result, dict) else str(result)
            }
            
            # Email notification (placeholder)
            if self.config['notifications']['enable_email_notifications']:
                logger.info(f"Email notification would be sent for task {task_name}")
            
            # Webhook notification (placeholder)
            if self.config['notifications']['enable_webhook_notifications']:
                webhook_url = self.config['notifications']['notification_webhook_url']
                if webhook_url:
                    logger.info(f"Webhook notification would be sent to {webhook_url} for task {task_name}")
                    
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            # Stop real-time monitoring
            if self.quality_monitor and self.quality_monitor.is_monitoring:
                self.quality_monitor.stop_monitoring()
                logger.info("Real-time monitoring stopped")
            
            # Save final statistics
            self._save_execution_statistics()
            
            logger.info("Scheduled task manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _save_execution_statistics(self):
        """Save execution statistics"""
        try:
            stats_file = Path('logs/scheduled_tasks/execution_statistics.json')
            stats_file.parent.mkdir(exist_ok=True)
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.execution_stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving execution statistics: {e}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task status"""
        return {
            'running_tasks': self.running_tasks.copy(),
            'execution_stats': self.execution_stats.copy(),
            'recent_tasks': self.task_history[-10:] if self.task_history else [],
            'scheduled_jobs': len(schedule.jobs),
            'next_run_times': [str(job.next_run) for job in schedule.jobs]
        }
    
    def execute_task_manually(self, task_name: str) -> Dict[str, Any]:
        """Execute a task manually"""
        try:
            if task_name == 'daily_cleaning':
                self._execute_daily_cleaning()
            elif task_name == 'weekly_cleaning':
                self._execute_weekly_cleaning()
            elif task_name == 'monthly_audit':
                self._execute_monthly_audit()
            else:
                return {'error': f'Unknown task: {task_name}'}
            
            return {'success': True, 'task': task_name}
            
        except Exception as e:
            logger.error(f"Error executing manual task {task_name}: {e}")
            return {'error': str(e)}


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scheduled Task Manager for LawFirmAI')
    parser.add_argument('--action', type=str, required=True,
                       choices=['start', 'status', 'execute', 'setup'],
                       help='Action to perform')
    parser.add_argument('--task', type=str,
                       choices=['daily_cleaning', 'weekly_cleaning', 'monthly_audit'],
                       help='Task to execute manually')
    parser.add_argument('--db-path', type=str, default='data/lawfirm.db',
                       help='Database file path')
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
    
    # Create task manager
    task_manager = ScheduledTaskManager(args.db_path, config)
    
    try:
        if args.action == 'setup':
            task_manager.setup_scheduled_tasks()
            print("Scheduled tasks setup completed")
        
        elif args.action == 'start':
            task_manager.setup_scheduled_tasks()
            print("Starting scheduled task scheduler...")
            print("Press Ctrl+C to stop")
            task_manager.run_scheduler()
        
        elif args.action == 'status':
            status = task_manager.get_task_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
        
        elif args.action == 'execute':
            if not args.task:
                print("Task name is required for execute action")
                sys.exit(1)
            
            result = task_manager.execute_task_manually(args.task)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        logger.error(f"Scheduled task operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
