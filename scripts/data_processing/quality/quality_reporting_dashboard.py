#!/usr/bin/env python3
"""
Quality Reporting Dashboard and Report Generator for LawFirmAI

This module provides comprehensive quality reporting capabilities including:
- Interactive dashboard for quality metrics
- Automated report generation
- Quality trend analysis
- Performance metrics visualization
- Export capabilities for various formats

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    from scripts.data_processing.quality.data_quality_validator import DataQualityValidator, QualityReport
    from scripts.data_processing.quality.automated_data_cleaner import AutomatedDataCleaner, CleaningReport
    from scripts.data_processing.quality.real_time_quality_monitor import RealTimeQualityMonitor, QualityMetrics, QualityAlert
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
        logging.FileHandler('logs/quality_reporting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QualityDashboardData:
    """Data structure for quality dashboard"""
    timestamp: datetime
    overall_quality_score: float
    quality_distribution: Dict[str, int]
    duplicate_statistics: Dict[str, Any]
    recent_quality_trends: List[Dict[str, Any]]
    active_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class QualityReportData:
    """Data structure for quality reports"""
    report_id: str
    report_type: str
    generation_time: datetime
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    detailed_metrics: Dict[str, Any]
    quality_analysis: Dict[str, Any]
    duplicate_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    charts_data: Dict[str, Any]


class QualityReportingDashboard:
    """
    Quality reporting dashboard for comprehensive data quality analysis
    
    This class provides:
    - Real-time quality metrics dashboard
    - Historical quality trend analysis
    - Automated report generation
    - Export capabilities (JSON, CSV, HTML)
    - Performance monitoring
    - Quality recommendations
    """
    
    def __init__(self, db_path: str = "data/lawfirm.db", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quality reporting dashboard
        
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
            self.data_cleaner = AutomatedDataCleaner(db_path, self.config.get('data_cleaner', {}))
            self.quality_monitor = RealTimeQualityMonitor(db_path, self.config.get('quality_monitor', {}))
        else:
            self.quality_validator = None
            self.data_cleaner = None
            self.quality_monitor = None
        
        # Report storage
        self.report_cache: Dict[str, QualityReportData] = {}
        self.dashboard_cache: Optional[QualityDashboardData] = None
        
        logger.info("QualityReportingDashboard initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'dashboard': {
                'cache_duration_minutes': 5,
                'max_history_days': 30,
                'auto_refresh_interval_seconds': 300
            },
            'reporting': {
                'default_report_types': ['summary', 'detailed', 'trends', 'performance'],
                'export_formats': ['json', 'csv', 'html'],
                'include_charts': True,
                'include_recommendations': True
            },
            'quality_thresholds': {
                'excellent': 0.9,
                'good': 0.8,
                'fair': 0.7,
                'poor': 0.6
            },
            'performance_metrics': {
                'enable_query_performance': True,
                'enable_memory_usage': True,
                'enable_response_time': True
            }
        }
    
    def get_dashboard_data(self, force_refresh: bool = False) -> QualityDashboardData:
        """
        Get current dashboard data
        
        Args:
            force_refresh (bool): Force refresh of cached data
            
        Returns:
            QualityDashboardData: Current dashboard data
        """
        # Check cache validity
        if not force_refresh and self.dashboard_cache:
            cache_age = datetime.now() - self.dashboard_cache.timestamp
            if cache_age.total_seconds() < self.config['dashboard']['cache_duration_minutes'] * 60:
                return self.dashboard_cache
        
        try:
            logger.info("Generating dashboard data")
            
            # Collect overall quality metrics
            overall_quality = self._get_overall_quality_metrics()
            
            # Get quality distribution
            quality_distribution = self._get_quality_distribution()
            
            # Get duplicate statistics
            duplicate_stats = self._get_duplicate_statistics()
            
            # Get recent quality trends
            recent_trends = self._get_recent_quality_trends()
            
            # Get active alerts
            active_alerts = self._get_active_alerts()
            
            # Get performance metrics
            performance_metrics = self._get_performance_metrics()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_quality, duplicate_stats, active_alerts)
            
            # Create dashboard data
            dashboard_data = QualityDashboardData(
                timestamp=datetime.now(),
                overall_quality_score=overall_quality.get('average_quality', 0.0),
                quality_distribution=quality_distribution,
                duplicate_statistics=duplicate_stats,
                recent_quality_trends=recent_trends,
                active_alerts=active_alerts,
                performance_metrics=performance_metrics,
                recommendations=recommendations
            )
            
            # Cache the data
            self.dashboard_cache = dashboard_data
            
            logger.info("Dashboard data generated successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return self._create_empty_dashboard_data()
    
    def _get_overall_quality_metrics(self) -> Dict[str, Any]:
        """Get overall quality metrics"""
        if not self.db_manager:
            return {'average_quality': 0.0, 'total_records': 0}
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        COUNT(*) as total_records,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality,
                        COUNT(CASE WHEN quality_score >= ? THEN 1 END) as excellent_count,
                        COUNT(CASE WHEN quality_score >= ? AND quality_score < ? THEN 1 END) as good_count,
                        COUNT(CASE WHEN quality_score >= ? AND quality_score < ? THEN 1 END) as fair_count,
                        COUNT(CASE WHEN quality_score < ? THEN 1 END) as poor_count
                    FROM assembly_laws 
                    WHERE quality_score IS NOT NULL
                ''', (
                    self.config['quality_thresholds']['excellent'],
                    self.config['quality_thresholds']['good'],
                    self.config['quality_thresholds']['excellent'],
                    self.config['quality_thresholds']['fair'],
                    self.config['quality_thresholds']['good'],
                    self.config['quality_thresholds']['fair']
                ))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'average_quality': result[0] or 0.0,
                        'total_records': result[1] or 0,
                        'min_quality': result[2] or 0.0,
                        'max_quality': result[3] or 0.0,
                        'excellent_count': result[4] or 0,
                        'good_count': result[5] or 0,
                        'fair_count': result[6] or 0,
                        'poor_count': result[7] or 0
                    }
                
                return {'average_quality': 0.0, 'total_records': 0}
                
        except Exception as e:
            logger.error(f"Error getting overall quality metrics: {e}")
            return {'average_quality': 0.0, 'total_records': 0}
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get quality distribution by categories"""
        if not self.db_manager:
            return {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN quality_score >= ? THEN 1 END) as excellent,
                        COUNT(CASE WHEN quality_score >= ? AND quality_score < ? THEN 1 END) as good,
                        COUNT(CASE WHEN quality_score >= ? AND quality_score < ? THEN 1 END) as fair,
                        COUNT(CASE WHEN quality_score < ? THEN 1 END) as poor
                    FROM assembly_laws 
                    WHERE quality_score IS NOT NULL
                ''', (
                    self.config['quality_thresholds']['excellent'],
                    self.config['quality_thresholds']['good'],
                    self.config['quality_thresholds']['excellent'],
                    self.config['quality_thresholds']['fair'],
                    self.config['quality_thresholds']['good'],
                    self.config['quality_thresholds']['fair']
                ))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'excellent': result[0] or 0,
                        'good': result[1] or 0,
                        'fair': result[2] or 0,
                        'poor': result[3] or 0
                    }
                
                return {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
                
        except Exception as e:
            logger.error(f"Error getting quality distribution: {e}")
            return {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
    
    def _get_duplicate_statistics(self) -> Dict[str, Any]:
        """Get duplicate statistics"""
        if not self.db_manager:
            return {'total_duplicates': 0, 'duplicate_groups': 0, 'duplicate_percentage': 0.0}
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get duplicate statistics
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN duplicate_group_id IS NOT NULL THEN 1 END) as duplicate_count,
                        COUNT(DISTINCT duplicate_group_id) as duplicate_groups,
                        COUNT(*) as total_records
                    FROM assembly_laws
                ''')
                
                result = cursor.fetchone()
                
                if result:
                    duplicate_count, duplicate_groups, total_records = result
                    duplicate_percentage = (duplicate_count / total_records * 100) if total_records > 0 else 0
                    
                    return {
                        'total_duplicates': duplicate_count or 0,
                        'duplicate_groups': duplicate_groups or 0,
                        'duplicate_percentage': duplicate_percentage,
                        'total_records': total_records or 0
                    }
                
                return {'total_duplicates': 0, 'duplicate_groups': 0, 'duplicate_percentage': 0.0}
                
        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {'total_duplicates': 0, 'duplicate_groups': 0, 'duplicate_percentage': 0.0}
    
    def _get_recent_quality_trends(self) -> List[Dict[str, Any]]:
        """Get recent quality trends"""
        if not self.db_manager:
            return []
        
        try:
            trends = []
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get quality trends for last 7 days
                for i in range(7):
                    date = datetime.now() - timedelta(days=i)
                    date_str = date.strftime('%Y-%m-%d')
                    
                    cursor.execute('''
                        SELECT 
                            AVG(quality_score) as avg_quality,
                            COUNT(*) as record_count
                        FROM assembly_laws 
                        WHERE DATE(migration_timestamp) = ? AND quality_score IS NOT NULL
                    ''', (date_str,))
                    
                    result = cursor.fetchone()
                    
                    if result:
                        trends.append({
                            'date': date_str,
                            'average_quality': result[0] or 0.0,
                            'record_count': result[1] or 0
                        })
            
            return sorted(trends, key=lambda x: x['date'])
            
        except Exception as e:
            logger.error(f"Error getting recent quality trends: {e}")
            return []
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active quality alerts"""
        if not self.quality_monitor:
            return []
        
        try:
            alerts = self.quality_monitor.get_active_alerts()
            return [asdict(alert) for alert in alerts]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = {}
            
            if self.config['performance_metrics']['enable_query_performance']:
                metrics['query_performance'] = self._get_query_performance_metrics()
            
            if self.config['performance_metrics']['enable_memory_usage']:
                metrics['memory_usage'] = self._get_memory_usage_metrics()
            
            if self.config['performance_metrics']['enable_response_time']:
                metrics['response_time'] = self._get_response_time_metrics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_query_performance_metrics(self) -> Dict[str, Any]:
        """Get query performance metrics"""
        if not self.db_manager:
            return {'average_query_time': 0.0, 'slow_queries': 0}
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get database file size
                db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                
                # Get table statistics
                cursor.execute('SELECT COUNT(*) FROM assembly_laws')
                law_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM assembly_articles')
                article_count = cursor.fetchone()[0]
                
                return {
                    'database_size_mb': db_size / (1024 * 1024),
                    'law_count': law_count,
                    'article_count': article_count,
                    'records_per_mb': law_count / (db_size / (1024 * 1024)) if db_size > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting query performance metrics: {e}")
            return {'average_query_time': 0.0, 'slow_queries': 0}
    
    def _get_memory_usage_metrics(self) -> Dict[str, Any]:
        """Get memory usage metrics"""
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
            
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
        except Exception as e:
            logger.error(f"Error getting memory usage metrics: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def _get_response_time_metrics(self) -> Dict[str, Any]:
        """Get response time metrics"""
        # Placeholder for response time metrics
        return {
            'average_response_time_ms': 0.0,
            'p95_response_time_ms': 0.0,
            'p99_response_time_ms': 0.0
        }
    
    def _generate_recommendations(self, overall_quality: Dict[str, Any], 
                                 duplicate_stats: Dict[str, Any], 
                                 active_alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate quality recommendations"""
        recommendations = []
        
        try:
            # Quality-based recommendations
            avg_quality = overall_quality.get('average_quality', 0.0)
            if avg_quality < self.config['quality_thresholds']['good']:
                recommendations.append(f"Overall quality score ({avg_quality:.2f}) is below good threshold. Consider running comprehensive quality improvement.")
            
            # Duplicate-based recommendations
            duplicate_percentage = duplicate_stats.get('duplicate_percentage', 0.0)
            if duplicate_percentage > 5.0:
                recommendations.append(f"Duplicate rate ({duplicate_percentage:.1f}%) is high. Run duplicate detection and resolution.")
            
            # Alert-based recommendations
            critical_alerts = [a for a in active_alerts if a.get('severity') == 'critical']
            if critical_alerts:
                recommendations.append(f"{len(critical_alerts)} critical alerts active. Immediate attention required.")
            
            # Performance-based recommendations
            if overall_quality.get('total_records', 0) > 100000:
                recommendations.append("Large dataset detected. Consider implementing data archiving strategy.")
            
            # Default recommendations if no specific issues
            if not recommendations:
                recommendations.append("Data quality is within acceptable ranges. Continue regular monitoring.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _create_empty_dashboard_data(self) -> QualityDashboardData:
        """Create empty dashboard data when errors occur"""
        return QualityDashboardData(
            timestamp=datetime.now(),
            overall_quality_score=0.0,
            quality_distribution={'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            duplicate_statistics={'total_duplicates': 0, 'duplicate_groups': 0, 'duplicate_percentage': 0.0},
            recent_quality_trends=[],
            active_alerts=[],
            performance_metrics={},
            recommendations=["Unable to generate recommendations due to system error"]
        )
    
    def generate_quality_report(self, report_type: str = 'summary', 
                              period_days: int = 7) -> QualityReportData:
        """
        Generate a comprehensive quality report
        
        Args:
            report_type (str): Type of report to generate
            period_days (int): Number of days to include in the report
            
        Returns:
            QualityReportData: Generated report data
        """
        try:
            logger.info(f"Generating {report_type} quality report for {period_days} days")
            
            report_id = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)
            
            # Generate report sections
            summary = self._generate_report_summary(period_start, period_end)
            detailed_metrics = self._generate_detailed_metrics(period_start, period_end)
            quality_analysis = self._generate_quality_analysis(period_start, period_end)
            duplicate_analysis = self._generate_duplicate_analysis(period_start, period_end)
            performance_analysis = self._generate_performance_analysis(period_start, period_end)
            recommendations = self._generate_report_recommendations(summary, quality_analysis, duplicate_analysis)
            charts_data = self._generate_charts_data(period_start, period_end)
            
            # Create report data
            report_data = QualityReportData(
                report_id=report_id,
                report_type=report_type,
                generation_time=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                summary=summary,
                detailed_metrics=detailed_metrics,
                quality_analysis=quality_analysis,
                duplicate_analysis=duplicate_analysis,
                performance_analysis=performance_analysis,
                recommendations=recommendations,
                charts_data=charts_data
            )
            
            # Cache the report
            self.report_cache[report_id] = report_data
            
            logger.info(f"Quality report {report_id} generated successfully")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return self._create_empty_report_data(report_type, period_days)
    
    def _generate_report_summary(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate report summary"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'overall_quality_score': dashboard_data.overall_quality_score,
                'total_records': dashboard_data.duplicate_statistics.get('total_records', 0),
                'duplicate_percentage': dashboard_data.duplicate_statistics.get('duplicate_percentage', 0.0),
                'active_alerts': len(dashboard_data.active_alerts),
                'quality_trend': 'stable',  # Placeholder
                'period_days': (period_end - period_start).days
            }
            
        except Exception as e:
            logger.error(f"Error generating report summary: {e}")
            return {'error': str(e)}
    
    def _generate_detailed_metrics(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate detailed metrics"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'quality_distribution': dashboard_data.quality_distribution,
                'duplicate_statistics': dashboard_data.duplicate_statistics,
                'recent_trends': dashboard_data.recent_quality_trends,
                'performance_metrics': dashboard_data.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed metrics: {e}")
            return {'error': str(e)}
    
    def _generate_quality_analysis(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate quality analysis"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'quality_score_analysis': {
                    'average': dashboard_data.overall_quality_score,
                    'distribution': dashboard_data.quality_distribution,
                    'trends': dashboard_data.recent_quality_trends
                },
                'quality_issues': [
                    alert for alert in dashboard_data.active_alerts 
                    if alert.get('alert_type', '').startswith('quality')
                ],
                'improvement_opportunities': self._identify_improvement_opportunities(dashboard_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating quality analysis: {e}")
            return {'error': str(e)}
    
    def _generate_duplicate_analysis(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate duplicate analysis"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'duplicate_statistics': dashboard_data.duplicate_statistics,
                'duplicate_trends': self._get_duplicate_trends(period_start, period_end),
                'duplicate_impact': self._assess_duplicate_impact(dashboard_data.duplicate_statistics)
            }
            
        except Exception as e:
            logger.error(f"Error generating duplicate analysis: {e}")
            return {'error': str(e)}
    
    def _generate_performance_analysis(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate performance analysis"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'performance_metrics': dashboard_data.performance_metrics,
                'performance_trends': self._get_performance_trends(period_start, period_end),
                'performance_recommendations': self._generate_performance_recommendations(dashboard_data.performance_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance analysis: {e}")
            return {'error': str(e)}
    
    def _generate_report_recommendations(self, summary: Dict[str, Any], 
                                       quality_analysis: Dict[str, Any], 
                                       duplicate_analysis: Dict[str, Any]) -> List[str]:
        """Generate report recommendations"""
        recommendations = []
        
        try:
            # Quality-based recommendations
            overall_quality = summary.get('overall_quality_score', 0.0)
            if overall_quality < 0.8:
                recommendations.append("Implement comprehensive quality improvement program")
            
            # Duplicate-based recommendations
            duplicate_percentage = summary.get('duplicate_percentage', 0.0)
            if duplicate_percentage > 5.0:
                recommendations.append("Execute duplicate detection and resolution process")
            
            # Alert-based recommendations
            active_alerts = summary.get('active_alerts', 0)
            if active_alerts > 0:
                recommendations.append("Address active quality alerts")
            
            # Performance-based recommendations
            if 'performance_metrics' in quality_analysis:
                recommendations.append("Monitor and optimize system performance")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating report recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _generate_charts_data(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Generate data for charts"""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            return {
                'quality_trend_chart': {
                    'labels': [trend['date'] for trend in dashboard_data.recent_quality_trends],
                    'data': [trend['average_quality'] for trend in dashboard_data.recent_quality_trends]
                },
                'quality_distribution_chart': {
                    'labels': list(dashboard_data.quality_distribution.keys()),
                    'data': list(dashboard_data.quality_distribution.values())
                },
                'duplicate_trend_chart': {
                    'labels': ['Current'],
                    'data': [dashboard_data.duplicate_statistics.get('duplicate_percentage', 0.0)]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating charts data: {e}")
            return {}
    
    def _identify_improvement_opportunities(self, dashboard_data: QualityDashboardData) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []
        
        try:
            # Quality score opportunities
            if dashboard_data.overall_quality_score < 0.9:
                opportunities.append("Improve overall data quality through enhanced preprocessing")
            
            # Duplicate opportunities
            if dashboard_data.duplicate_statistics.get('duplicate_percentage', 0) > 2.0:
                opportunities.append("Implement stricter duplicate prevention measures")
            
            # Alert opportunities
            if dashboard_data.active_alerts:
                opportunities.append("Address active quality alerts to improve system stability")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying improvement opportunities: {e}")
            return []
    
    def _get_duplicate_trends(self, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
        """Get duplicate trends for the period"""
        # Placeholder implementation
        return [{'date': period_start.strftime('%Y-%m-%d'), 'duplicate_percentage': 0.0}]
    
    def _assess_duplicate_impact(self, duplicate_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of duplicates"""
        duplicate_percentage = duplicate_stats.get('duplicate_percentage', 0.0)
        
        if duplicate_percentage < 1.0:
            impact_level = 'low'
        elif duplicate_percentage < 5.0:
            impact_level = 'medium'
        else:
            impact_level = 'high'
        
        return {
            'impact_level': impact_level,
            'duplicate_percentage': duplicate_percentage,
            'recommendation': f"Duplicate impact is {impact_level}"
        }
    
    def _get_performance_trends(self, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
        """Get performance trends for the period"""
        # Placeholder implementation
        return [{'date': period_start.strftime('%Y-%m-%d'), 'performance_score': 1.0}]
    
    def _generate_performance_recommendations(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            if 'memory_usage' in performance_metrics:
                memory_percent = performance_metrics['memory_usage'].get('percent', 0)
                if memory_percent > 80:
                    recommendations.append("High memory usage detected. Consider memory optimization")
            
            if 'query_performance' in performance_metrics:
                db_size = performance_metrics['query_performance'].get('database_size_mb', 0)
                if db_size > 1000:
                    recommendations.append("Large database size. Consider archiving old data")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return []
    
    def _create_empty_report_data(self, report_type: str, period_days: int) -> QualityReportData:
        """Create empty report data when errors occur"""
        return QualityReportData(
            report_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=report_type,
            generation_time=datetime.now(),
            period_start=datetime.now() - timedelta(days=period_days),
            period_end=datetime.now(),
            summary={'error': 'Report generation failed'},
            detailed_metrics={'error': 'Report generation failed'},
            quality_analysis={'error': 'Report generation failed'},
            duplicate_analysis={'error': 'Report generation failed'},
            performance_analysis={'error': 'Report generation failed'},
            recommendations=['Unable to generate report due to system error'],
            charts_data={}
        )
    
    def export_report(self, report_data: QualityReportData, format: str = 'json', 
                     output_path: Optional[str] = None) -> str:
        """
        Export report to specified format
        
        Args:
            report_data (QualityReportData): Report data to export
            format (str): Export format ('json', 'csv', 'html')
            output_path (str, optional): Output file path
            
        Returns:
            str: Path to exported file
        """
        try:
            if not output_path:
                output_path = f"reports/quality_report_{report_data.report_id}.{format}"
            
            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)
            
            if format == 'json':
                return self._export_json_report(report_data, output_file)
            elif format == 'csv':
                return self._export_csv_report(report_data, output_file)
            elif format == 'html':
                return self._export_html_report(report_data, output_file)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
    
    def _export_json_report(self, report_data: QualityReportData, output_file: Path) -> str:
        """Export report as JSON"""
        try:
            report_dict = asdict(report_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Report exported to JSON: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
            raise
    
    def _export_csv_report(self, report_data: QualityReportData, output_file: Path) -> str:
        """Export report as CSV"""
        try:
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write summary data
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Report ID', report_data.report_id])
                writer.writerow(['Report Type', report_data.report_type])
                writer.writerow(['Generation Time', report_data.generation_time])
                writer.writerow(['Period Start', report_data.period_start])
                writer.writerow(['Period End', report_data.period_end])
                
                # Write summary metrics
                writer.writerow([])
                writer.writerow(['Summary Metrics'])
                for key, value in report_data.summary.items():
                    writer.writerow([key, value])
                
                # Write recommendations
                writer.writerow([])
                writer.writerow(['Recommendations'])
                for i, rec in enumerate(report_data.recommendations, 1):
                    writer.writerow([f'Recommendation {i}', rec])
            
            logger.info(f"Report exported to CSV: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting CSV report: {e}")
            raise
    
    def _export_html_report(self, report_data: QualityReportData, output_file: Path) -> str:
        """Export report as HTML"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report - {report_data.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .recommendation {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Report</h1>
        <p><strong>Report ID:</strong> {report_data.report_id}</p>
        <p><strong>Type:</strong> {report_data.report_type}</p>
        <p><strong>Generated:</strong> {report_data.generation_time}</p>
        <p><strong>Period:</strong> {report_data.period_start} to {report_data.period_end}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
            
            for key, value in report_data.summary.items():
                html_content += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
"""
            
            for i, rec in enumerate(report_data.recommendations, 1):
                html_content += f'        <div class="recommendation"><strong>Recommendation {i}:</strong> {rec}</div>\n'
            
            html_content += """
    </div>
</body>
</html>
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report exported to HTML: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting HTML report: {e}")
            raise
    
    def get_cached_report(self, report_id: str) -> Optional[QualityReportData]:
        """Get cached report by ID"""
        return self.report_cache.get(report_id)
    
    def clear_report_cache(self):
        """Clear report cache"""
        self.report_cache.clear()
        logger.info("Report cache cleared")
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status and statistics"""
        return {
            'dashboard_cache_age': (datetime.now() - self.dashboard_cache.timestamp).total_seconds() if self.dashboard_cache else None,
            'cached_reports_count': len(self.report_cache),
            'config': self.config,
            'modules_available': {
                'quality_validator': self.quality_validator is not None,
                'data_cleaner': self.data_cleaner is not None,
                'quality_monitor': self.quality_monitor is not None,
                'database_manager': self.db_manager is not None
            }
        }


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality Reporting Dashboard for LawFirmAI')
    parser.add_argument('--action', type=str, required=True,
                       choices=['dashboard', 'report', 'export'],
                       help='Action to perform')
    parser.add_argument('--report-type', type=str, default='summary',
                       choices=['summary', 'detailed', 'trends', 'performance'],
                       help='Type of report to generate')
    parser.add_argument('--period-days', type=int, default=7,
                       help='Number of days to include in report')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'csv', 'html'],
                       help='Export format')
    parser.add_argument('--output', type=str,
                       help='Output file path')
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
    
    # Create dashboard
    dashboard = QualityReportingDashboard(args.db_path, config)
    
    try:
        if args.action == 'dashboard':
            dashboard_data = dashboard.get_dashboard_data(force_refresh=True)
            print(json.dumps(asdict(dashboard_data), ensure_ascii=False, indent=2, default=str))
        
        elif args.action == 'report':
            report_data = dashboard.generate_quality_report(args.report_type, args.period_days)
            print(json.dumps(asdict(report_data), ensure_ascii=False, indent=2, default=str))
        
        elif args.action == 'export':
            report_data = dashboard.generate_quality_report(args.report_type, args.period_days)
            output_path = dashboard.export_report(report_data, args.format, args.output)
            print(f"Report exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Dashboard operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

