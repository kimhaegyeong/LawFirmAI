#!/usr/bin/env python3
"""
Real-Time Quality Monitor for LawFirmAI

This module provides real-time monitoring of data quality metrics and alerts
when quality thresholds are exceeded.

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    from scripts.data_processing.quality.data_quality_validator import DataQualityValidator, QualityReport
    QUALITY_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quality module not available: {e}")
    QUALITY_MODULE_AVAILABLE = False

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
        logging.FileHandler('logs/real_time_quality_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """Quality alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    threshold_value: float
    actual_value: float
    law_id: Optional[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class QualityMetrics:
    """Quality metrics data structure"""
    timestamp: datetime
    overall_quality_score: float
    average_quality_score: float
    low_quality_percentage: float
    duplicate_percentage: float
    total_records: int
    low_quality_records: int
    duplicate_records: int
    recent_additions: int
    quality_trend: str  # 'improving', 'stable', 'degrading'
    alerts_count: int
    critical_alerts: int


class RealTimeQualityMonitor:
    """
    Real-time quality monitor for continuous data quality tracking
    
    This class provides:
    - Continuous quality monitoring
    - Alert generation when thresholds are exceeded
    - Quality trend analysis
    - Performance metrics tracking
    - Automated recommendations
    """
    
    def __init__(self, db_path: str = "data/lawfirm.db", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time quality monitor
        
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
            
        if QUALITY_MODULE_AVAILABLE:
            self.quality_validator = DataQualityValidator()
        else:
            self.quality_validator = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Metrics storage
        self.metrics_history: List[QualityMetrics] = []
        self.active_alerts: List[QualityAlert] = []
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        # Statistics
        self.monitoring_stats = {
            'start_time': None,
            'total_checks': 0,
            'alerts_generated': 0,
            'quality_checks': 0,
            'last_check_time': None
        }
        
        logger.info("RealTimeQualityMonitor initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'monitoring': {
                'enabled': True,
                'check_interval_seconds': 300,  # 5 minutes
                'max_history_size': 1000,
                'alert_cooldown_seconds': 3600  # 1 hour
            },
            'thresholds': {
                'overall_quality_min': 0.8,
                'low_quality_max_percentage': 10.0,
                'duplicate_max_percentage': 5.0,
                'quality_degradation_threshold': 0.05,
                'recent_low_quality_max': 50
            },
            'alerts': {
                'enable_email': False,
                'enable_logging': True,
                'enable_callbacks': True,
                'severity_levels': {
                    'low': 0.7,
                    'medium': 0.5,
                    'high': 0.3,
                    'critical': 0.1
                }
            },
            'performance': {
                'enable_performance_monitoring': True,
                'slow_query_threshold_ms': 1000,
                'memory_usage_threshold_mb': 1000
            }
        }
    
    def start_monitoring(self) -> bool:
        """
        Start real-time quality monitoring
        
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return False
        
        if not self.config['monitoring']['enabled']:
            logger.info("Monitoring is disabled in configuration")
            return False
        
        try:
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.is_monitoring = True
            self.monitoring_stats['start_time'] = datetime.now()
            
            logger.info("Real-time quality monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop real-time quality monitoring
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return False
        
        try:
            self.stop_event.set()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.is_monitoring = False
            
            logger.info("Real-time quality monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Perform quality check
                self._perform_quality_check()
                
                # Wait for next check
                check_interval = self.config['monitoring']['check_interval_seconds']
                self.stop_event.wait(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Wait a bit before retrying
                self.stop_event.wait(60)
        
        logger.info("Monitoring loop stopped")
    
    def _perform_quality_check(self):
        """Perform a single quality check"""
        try:
            start_time = datetime.now()
            
            # Collect current metrics
            metrics = self._collect_quality_metrics()
            
            # Analyze trends
            self._analyze_quality_trends(metrics)
            
            # Check thresholds and generate alerts
            self._check_thresholds_and_alerts(metrics)
            
            # Store metrics
            self._store_metrics(metrics)
            
            # Update statistics
            self.monitoring_stats['total_checks'] += 1
            self.monitoring_stats['last_check_time'] = start_time
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Quality check completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
    
    def _collect_quality_metrics(self) -> QualityMetrics:
        """Collect current quality metrics"""
        if not self.db_manager:
            return self._create_empty_metrics()
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get overall quality statistics
                cursor.execute('''
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as total_records,
                        COUNT(CASE WHEN quality_score < ? THEN 1 END) as low_quality_count,
                        COUNT(CASE WHEN duplicate_group_id IS NOT NULL THEN 1 END) as duplicate_count,
                        COUNT(CASE WHEN migration_timestamp > datetime('now', '-1 hour') THEN 1 END) as recent_additions
                    FROM assembly_laws 
                    WHERE quality_score IS NOT NULL
                ''', (self.config['thresholds']['overall_quality_min'],))
                
                stats = cursor.fetchone()
                
                if not stats or stats[0] is None:
                    return self._create_empty_metrics()
                
                avg_quality, total_records, low_quality_count, duplicate_count, recent_additions = stats
                
                # Calculate percentages
                low_quality_percentage = (low_quality_count / total_records * 100) if total_records > 0 else 0
                duplicate_percentage = (duplicate_count / total_records * 100) if total_records > 0 else 0
                
                # Determine quality trend
                quality_trend = self._determine_quality_trend(avg_quality)
                
                # Count active alerts
                alerts_count = len(self.active_alerts)
                critical_alerts = len([a for a in self.active_alerts if a.severity == 'critical'])
                
                return QualityMetrics(
                    timestamp=datetime.now(),
                    overall_quality_score=avg_quality,
                    average_quality_score=avg_quality,
                    low_quality_percentage=low_quality_percentage,
                    duplicate_percentage=duplicate_percentage,
                    total_records=total_records,
                    low_quality_records=low_quality_count,
                    duplicate_records=duplicate_count,
                    recent_additions=recent_additions,
                    quality_trend=quality_trend,
                    alerts_count=alerts_count,
                    critical_alerts=critical_alerts
                )
                
        except Exception as e:
            logger.error(f"Error collecting quality metrics: {e}")
            return self._create_empty_metrics()
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty metrics when database is not available"""
        return QualityMetrics(
            timestamp=datetime.now(),
            overall_quality_score=0.0,
            average_quality_score=0.0,
            low_quality_percentage=0.0,
            duplicate_percentage=0.0,
            total_records=0,
            low_quality_records=0,
            duplicate_records=0,
            recent_additions=0,
            quality_trend='stable',
            alerts_count=0,
            critical_alerts=0
        )
    
    def _determine_quality_trend(self, current_quality: float) -> str:
        """Determine quality trend based on recent metrics"""
        if len(self.metrics_history) < 2:
            return 'stable'
        
        # Get recent metrics (last 5 checks)
        recent_metrics = self.metrics_history[-5:]
        
        if len(recent_metrics) < 2:
            return 'stable'
        
        # Calculate trend
        quality_values = [m.overall_quality_score for m in recent_metrics]
        trend_threshold = self.config['thresholds']['quality_degradation_threshold']
        
        # Simple trend analysis
        if len(quality_values) >= 3:
            recent_avg = sum(quality_values[-3:]) / 3
            older_avg = sum(quality_values[:-3]) / len(quality_values[:-3])
            
            if recent_avg > older_avg + trend_threshold:
                return 'improving'
            elif recent_avg < older_avg - trend_threshold:
                return 'degrading'
        
        return 'stable'
    
    def _analyze_quality_trends(self, current_metrics: QualityMetrics):
        """Analyze quality trends and patterns"""
        try:
            # Check for rapid quality degradation
            if len(self.metrics_history) >= 3:
                recent_metrics = self.metrics_history[-3:]
                quality_values = [m.overall_quality_score for m in recent_metrics]
                
                if all(quality_values[i] > quality_values[i+1] for i in range(len(quality_values)-1)):
                    # Quality is consistently decreasing
                    self._generate_alert(
                        alert_type='quality_degradation',
                        severity='high',
                        message=f'Quality score has decreased from {quality_values[0]:.3f} to {quality_values[-1]:.3f} over last 3 checks',
                        threshold_value=quality_values[0],
                        actual_value=quality_values[-1]
                    )
            
            # Check for high duplicate rate
            if current_metrics.duplicate_percentage > self.config['thresholds']['duplicate_max_percentage']:
                self._generate_alert(
                    alert_type='high_duplicate_rate',
                    severity='medium',
                    message=f'Duplicate rate is {current_metrics.duplicate_percentage:.1f}%, exceeding threshold of {self.config["thresholds"]["duplicate_max_percentage"]}%',
                    threshold_value=self.config['thresholds']['duplicate_max_percentage'],
                    actual_value=current_metrics.duplicate_percentage
                )
            
            # Check for recent low-quality additions
            if current_metrics.recent_additions > self.config['thresholds']['recent_low_quality_max']:
                self._generate_alert(
                    alert_type='recent_low_quality',
                    severity='medium',
                    message=f'{current_metrics.recent_additions} low-quality records added in the last hour',
                    threshold_value=self.config['thresholds']['recent_low_quality_max'],
                    actual_value=current_metrics.recent_additions
                )
                
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {e}")
    
    def _check_thresholds_and_alerts(self, metrics: QualityMetrics):
        """Check thresholds and generate alerts"""
        try:
            # Overall quality threshold
            if metrics.overall_quality_score < self.config['thresholds']['overall_quality_min']:
                severity = self._determine_severity(metrics.overall_quality_score)
                self._generate_alert(
                    alert_type='low_overall_quality',
                    severity=severity,
                    message=f'Overall quality score {metrics.overall_quality_score:.3f} is below threshold {self.config["thresholds"]["overall_quality_min"]}',
                    threshold_value=self.config['thresholds']['overall_quality_min'],
                    actual_value=metrics.overall_quality_score
                )
            
            # Low quality percentage threshold
            if metrics.low_quality_percentage > self.config['thresholds']['low_quality_max_percentage']:
                severity = self._determine_severity_by_percentage(metrics.low_quality_percentage)
                self._generate_alert(
                    alert_type='high_low_quality_percentage',
                    severity=severity,
                    message=f'{metrics.low_quality_percentage:.1f}% of records have low quality, exceeding threshold of {self.config["thresholds"]["low_quality_max_percentage"]}%',
                    threshold_value=self.config['thresholds']['low_quality_max_percentage'],
                    actual_value=metrics.low_quality_percentage
                )
            
            # Duplicate percentage threshold
            if metrics.duplicate_percentage > self.config['thresholds']['duplicate_max_percentage']:
                severity = self._determine_severity_by_percentage(metrics.duplicate_percentage)
                self._generate_alert(
                    alert_type='high_duplicate_percentage',
                    severity=severity,
                    message=f'{metrics.duplicate_percentage:.1f}% of records are duplicates, exceeding threshold of {self.config["thresholds"]["duplicate_max_percentage"]}%',
                    threshold_value=self.config['thresholds']['duplicate_max_percentage'],
                    actual_value=metrics.duplicate_percentage
                )
                
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    def _determine_severity(self, quality_score: float) -> str:
        """Determine alert severity based on quality score"""
        severity_levels = self.config['alerts']['severity_levels']
        
        if quality_score <= severity_levels['critical']:
            return 'critical'
        elif quality_score <= severity_levels['high']:
            return 'high'
        elif quality_score <= severity_levels['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _determine_severity_by_percentage(self, percentage: float) -> str:
        """Determine alert severity based on percentage"""
        if percentage > 50:
            return 'critical'
        elif percentage > 25:
            return 'high'
        elif percentage > 15:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alert(self, alert_type: str, severity: str, message: str, 
                       threshold_value: float, actual_value: float, law_id: Optional[str] = None):
        """Generate a quality alert"""
        try:
            # Check alert cooldown
            cooldown_seconds = self.config['monitoring']['alert_cooldown_seconds']
            recent_alerts = [
                a for a in self.active_alerts 
                if a.alert_type == alert_type and 
                (datetime.now() - a.timestamp).total_seconds() < cooldown_seconds
            ]
            
            if recent_alerts:
                logger.debug(f"Alert {alert_type} suppressed due to cooldown")
                return
            
            # Create alert
            alert_id = f"{alert_type}_{int(time.time())}"
            alert = QualityAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                threshold_value=threshold_value,
                actual_value=actual_value,
                law_id=law_id,
                recommendations=self._generate_recommendations(alert_type, severity)
            )
            
            # Add to active alerts
            self.active_alerts.append(alert)
            
            # Limit active alerts
            max_alerts = 100
            if len(self.active_alerts) > max_alerts:
                self.active_alerts = self.active_alerts[-max_alerts:]
            
            # Log alert
            if self.config['alerts']['enable_logging']:
                logger.warning(f"Quality Alert [{severity.upper()}]: {message}")
            
            # Call callbacks
            if self.config['alerts']['enable_callbacks']:
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
            
            # Update statistics
            self.monitoring_stats['alerts_generated'] += 1
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    def _generate_recommendations(self, alert_type: str, severity: str) -> List[str]:
        """Generate recommendations based on alert type"""
        recommendations = []
        
        if alert_type == 'low_overall_quality':
            recommendations.extend([
                'Run comprehensive quality assessment',
                'Review data preprocessing pipeline',
                'Consider manual quality review for low-scoring records'
            ])
        
        elif alert_type == 'high_duplicate_percentage':
            recommendations.extend([
                'Run duplicate detection and resolution',
                'Review data import processes',
                'Implement stricter duplicate prevention'
            ])
        
        elif alert_type == 'quality_degradation':
            recommendations.extend([
                'Investigate recent data changes',
                'Review quality validation rules',
                'Consider rolling back recent changes'
            ])
        
        elif alert_type == 'recent_low_quality':
            recommendations.extend([
                'Review recent data sources',
                'Check preprocessing pipeline for errors',
                'Implement stricter quality gates'
            ])
        
        # Add severity-based recommendations
        if severity in ['high', 'critical']:
            recommendations.append('Consider immediate manual intervention')
        
        return recommendations
    
    def _store_metrics(self, metrics: QualityMetrics):
        """Store metrics in history"""
        try:
            self.metrics_history.append(metrics)
            
            # Limit history size
            max_history = self.config['monitoring']['max_history_size']
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add an alert callback function"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Remove an alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_current_metrics(self) -> Optional[QualityMetrics]:
        """Get current quality metrics"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_history(self, hours: int = 24) -> List[QualityMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[QualityAlert]:
        """Get active alerts, optionally filtered by severity"""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts.copy()
    
    def clear_alerts(self, alert_type: Optional[str] = None):
        """Clear alerts, optionally filtered by type"""
        if alert_type:
            self.active_alerts = [a for a in self.active_alerts if a.alert_type != alert_type]
        else:
            self.active_alerts.clear()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status (alias for get_monitoring_status)"""
        return self.get_monitoring_status()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status and statistics"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_stats': self.monitoring_stats.copy(),
            'active_alerts_count': len(self.active_alerts),
            'metrics_history_size': len(self.metrics_history),
            'config': self.config,
            'modules_available': {
                'quality_validator': self.quality_validator is not None,
                'database_manager': self.db_manager is not None
            }
        }
    
    def export_metrics_report(self, output_path: str, hours: int = 24):
        """Export metrics report to file"""
        try:
            metrics_history = self.get_metrics_history(hours)
            active_alerts = self.get_active_alerts()
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_period_hours': hours,
                'monitoring_status': self.get_monitoring_status(),
                'metrics_history': [asdict(m) for m in metrics_history],
                'active_alerts': [asdict(a) for a in active_alerts],
                'summary': {
                    'total_metrics_points': len(metrics_history),
                    'total_alerts': len(active_alerts),
                    'critical_alerts': len([a for a in active_alerts if a.severity == 'critical']),
                    'average_quality_score': sum(m.overall_quality_score for m in metrics_history) / len(metrics_history) if metrics_history else 0
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Metrics report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics report: {e}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Quality Monitor for LawFirmAI')
    parser.add_argument('--action', type=str, required=True,
                       choices=['start', 'stop', 'status', 'metrics', 'alerts', 'export'],
                       help='Action to perform')
    parser.add_argument('--db-path', type=str, default='data/lawfirm.db',
                       help='Database file path')
    parser.add_argument('--output', type=str,
                       help='Output file path for reports')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours of history to include in reports')
    parser.add_argument('--severity', type=str,
                       choices=['low', 'medium', 'high', 'critical'],
                       help='Filter alerts by severity')
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
    
    # Create monitor
    monitor = RealTimeQualityMonitor(args.db_path, config)
    
    try:
        if args.action == 'start':
            success = monitor.start_monitoring()
            if success:
                print("Real-time quality monitoring started successfully")
                print("Press Ctrl+C to stop monitoring")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    monitor.stop_monitoring()
                    print("\nMonitoring stopped")
            else:
                print("Failed to start monitoring")
                sys.exit(1)
        
        elif args.action == 'stop':
            success = monitor.stop_monitoring()
            if success:
                print("Real-time quality monitoring stopped")
            else:
                print("Failed to stop monitoring")
                sys.exit(1)
        
        elif args.action == 'status':
            status = monitor.get_monitoring_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
        
        elif args.action == 'metrics':
            metrics = monitor.get_current_metrics()
            if metrics:
                print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2, default=str))
            else:
                print("No metrics available")
        
        elif args.action == 'alerts':
            alerts = monitor.get_active_alerts(args.severity)
            if alerts:
                print(json.dumps([asdict(a) for a in alerts], ensure_ascii=False, indent=2, default=str))
            else:
                print("No active alerts")
        
        elif args.action == 'export':
            if not args.output:
                print("Output path is required for export action")
                sys.exit(1)
            
            monitor.export_metrics_report(args.output, args.hours)
            print(f"Metrics report exported to {args.output}")
        
    except Exception as e:
        logger.error(f"Monitor operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
