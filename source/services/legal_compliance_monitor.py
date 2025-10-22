# -*- coding: utf-8 -*-
"""
Legal Compliance Monitoring System
법적 준수 모니터링 및 로깅 시스템
"""

import logging
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from pathlib import Path

from .improved_legal_restriction_system import ImprovedRestrictionResult, RestrictionLevel, LegalArea
from .content_filter_engine import FilterResult, IntentType, ContextType
from .response_validation_system import ValidationResult, ValidationStatus, ValidationLevel

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """준수 상태"""
    COMPLIANT = "compliant"          # 준수
    NON_COMPLIANT = "non_compliant"  # 비준수
    AT_RISK = "at_risk"              # 위험
    UNKNOWN = "unknown"              # 알 수 없음


class AlertLevel(Enum):
    """경고 수준"""
    INFO = "info"           # 정보
    WARNING = "warning"     # 경고
    ERROR = "error"         # 오류
    CRITICAL = "critical"   # 심각


@dataclass
class ComplianceEvent:
    """준수 이벤트"""
    id: str
    timestamp: datetime
    event_type: str
    severity: AlertLevel
    description: str
    details: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    resolved: bool = False


@dataclass
class ComplianceMetrics:
    """준수 메트릭"""
    total_requests: int
    compliant_requests: int
    non_compliant_requests: int
    at_risk_requests: int
    compliance_rate: float
    average_response_time: float
    critical_events: int
    warning_events: int
    last_updated: datetime


class LegalComplianceMonitor:
    """법적 준수 모니터링 시스템"""
    
    def __init__(self, db_path: str = "data/compliance_monitoring.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.compliance_events = []
        self.metrics = ComplianceMetrics(
            total_requests=0,
            compliant_requests=0,
            non_compliant_requests=0,
            at_risk_requests=0,
            compliance_rate=0.0,
            average_response_time=0.0,
            critical_events=0,
            warning_events=0,
            last_updated=datetime.now()
        )
        
        # 데이터베이스 초기화
        self._initialize_database()
        
        # 로깅 설정
        self._setup_logging()
        
    def _initialize_database(self):
        """데이터베이스 초기화"""
        try:
            # 메모리 데이터베이스가 아닌 경우에만 디렉토리 생성
            if self.db_path != ":memory:" and os.path.dirname(self.db_path):
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 준수 이벤트 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_events (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        description TEXT NOT NULL,
                        details TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # 준수 메트릭 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_requests INTEGER NOT NULL,
                        compliant_requests INTEGER NOT NULL,
                        non_compliant_requests INTEGER NOT NULL,
                        at_risk_requests INTEGER NOT NULL,
                        compliance_rate REAL NOT NULL,
                        average_response_time REAL NOT NULL,
                        critical_events INTEGER NOT NULL,
                        warning_events INTEGER NOT NULL
                    )
                """)
                
                # 사용자 활동 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_activity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT,
                        compliance_status TEXT NOT NULL,
                        validation_result TEXT,
                        processing_time REAL
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def _setup_logging(self):
        """로깅 설정"""
        try:
            # 준수 모니터링 전용 로거
            compliance_logger = logging.getLogger("compliance_monitor")
            compliance_logger.setLevel(logging.INFO)
            
            # 파일 핸들러
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"compliance_monitor_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.INFO)
            
            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            compliance_logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.error(f"Error setting up logging: {e}")
    
    def log_compliance_event(self, event_type: str, severity: AlertLevel, 
                           description: str, details: Dict[str, Any],
                           user_id: Optional[str] = None, 
                           session_id: Optional[str] = None) -> str:
        """준수 이벤트 로깅"""
        try:
            event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            event = ComplianceEvent(
                id=event_id,
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                description=description,
                details=details,
                user_id=user_id,
                session_id=session_id
            )
            
            # 메모리에 저장
            self.compliance_events.append(event)
            
            # 데이터베이스에 저장
            self._save_compliance_event(event)
            
            # 로그 기록
            compliance_logger = logging.getLogger("compliance_monitor")
            compliance_logger.info(f"Compliance Event: {event_type} - {description}")
            
            # 심각한 이벤트는 즉시 알림
            if severity == AlertLevel.CRITICAL:
                self._send_critical_alert(event)
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Error logging compliance event: {e}")
            return ""
    
    def monitor_request(self, query: str, response: str, 
                       restriction_result: ImprovedRestrictionResult,
                       filter_result: FilterResult,
                       validation_result: ValidationResult,
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       processing_time: float = 0.0) -> ComplianceStatus:
        """요청 모니터링"""
        try:
            # 준수 상태 결정
            compliance_status = self._determine_compliance_status(
                restriction_result, filter_result, validation_result
            )
            
            # 사용자 활동 로깅
            self._log_user_activity(
                user_id, session_id, query, response, compliance_status,
                validation_result, processing_time
            )
            
            # 준수 이벤트 생성
            if compliance_status != ComplianceStatus.COMPLIANT:
                self._create_compliance_event(
                    compliance_status, restriction_result, filter_result,
                    validation_result, user_id, session_id
                )
            
            # 메트릭 업데이트
            self._update_metrics(compliance_status, processing_time)
            
            return compliance_status
            
        except Exception as e:
            self.logger.error(f"Error monitoring request: {e}")
            return ComplianceStatus.UNKNOWN
    
    def _determine_compliance_status(self, restriction_result: ImprovedRestrictionResult,
                                   filter_result: FilterResult,
                                   validation_result: ValidationResult) -> ComplianceStatus:
        """준수 상태 결정"""
        # 심각한 제한 사항이 있으면 비준수
        if restriction_result.restriction_level == RestrictionLevel.CRITICAL:
            return ComplianceStatus.NON_COMPLIANT
        
        # 차단된 요청이면 비준수
        if filter_result.is_blocked:
            return ComplianceStatus.NON_COMPLIANT
        
        # 검증 실패하면 비준수
        if validation_result.status == ValidationStatus.REJECTED:
            return ComplianceStatus.NON_COMPLIANT
        
        # 높은 위험도면 위험
        if (restriction_result.restriction_level == RestrictionLevel.HIGH or
            filter_result.intent_analysis.risk_level == "high"):
            return ComplianceStatus.AT_RISK
        
        # 검증 수정이 필요하면 위험
        if validation_result.status == ValidationStatus.MODIFIED:
            return ComplianceStatus.AT_RISK
        
        # 모든 검사 통과하면 준수
        return ComplianceStatus.COMPLIANT
    
    def _log_user_activity(self, user_id: Optional[str], session_id: Optional[str],
                          query: str, response: str, compliance_status: ComplianceStatus,
                          validation_result: ValidationResult, processing_time: float):
        """사용자 활동 로깅"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_activity 
                    (user_id, session_id, timestamp, query, response, compliance_status, 
                     validation_result, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    datetime.now().isoformat(),
                    query[:1000],  # 길이 제한
                    response[:1000] if response else None,  # 길이 제한
                    compliance_status.value,
                    json.dumps(asdict(validation_result), default=str),
                    processing_time
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging user activity: {e}")
    
    def _create_compliance_event(self, compliance_status: ComplianceStatus,
                               restriction_result: ImprovedRestrictionResult,
                               filter_result: FilterResult,
                               validation_result: ValidationResult,
                               user_id: Optional[str],
                               session_id: Optional[str]):
        """준수 이벤트 생성"""
        try:
            # 이벤트 유형 결정
            if compliance_status == ComplianceStatus.NON_COMPLIANT:
                event_type = "non_compliance"
                severity = AlertLevel.ERROR
                description = "법적 준수 위반이 감지되었습니다."
            else:  # AT_RISK
                event_type = "risk_detected"
                severity = AlertLevel.WARNING
                description = "법적 위험이 감지되었습니다."
            
            # 상세 정보
            details = {
                "compliance_status": compliance_status.value,
                "restriction_level": restriction_result.restriction_level.value,
                "restricted_areas": [rule.area.value for rule in restriction_result.matched_rules],
                "intent_type": filter_result.intent_analysis.intent_type.value,
                "risk_level": filter_result.intent_analysis.risk_level,
                "validation_status": validation_result.status.value,
                "validation_level": validation_result.validation_level.value,
                "issues": validation_result.issues,
                "recommendations": validation_result.recommendations
            }
            
            # 이벤트 로깅
            self.log_compliance_event(
                event_type=event_type,
                severity=severity,
                description=description,
                details=details,
                user_id=user_id,
                session_id=session_id
            )
            
        except Exception as e:
            self.logger.error(f"Error creating compliance event: {e}")
    
    def _update_metrics(self, compliance_status: ComplianceStatus, processing_time: float):
        """메트릭 업데이트"""
        try:
            self.metrics.total_requests += 1
            
            if compliance_status == ComplianceStatus.COMPLIANT:
                self.metrics.compliant_requests += 1
            elif compliance_status == ComplianceStatus.NON_COMPLIANT:
                self.metrics.non_compliant_requests += 1
            elif compliance_status == ComplianceStatus.AT_RISK:
                self.metrics.at_risk_requests += 1
            
            # 준수율 계산
            if self.metrics.total_requests > 0:
                self.metrics.compliance_rate = (
                    self.metrics.compliant_requests / self.metrics.total_requests
                )
            
            # 평균 응답 시간 업데이트
            if self.metrics.total_requests > 0:
                current_avg = self.metrics.average_response_time
                self.metrics.average_response_time = (
                    (current_avg * (self.metrics.total_requests - 1) + processing_time) /
                    self.metrics.total_requests
                )
            
            # 이벤트 수 업데이트
            self.metrics.critical_events = len([
                event for event in self.compliance_events 
                if event.severity == AlertLevel.CRITICAL
            ])
            self.metrics.warning_events = len([
                event for event in self.compliance_events 
                if event.severity == AlertLevel.WARNING
            ])
            
            self.metrics.last_updated = datetime.now()
            
            # 주기적으로 데이터베이스에 저장
            if self.metrics.total_requests % 100 == 0:
                self._save_metrics()
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _save_compliance_event(self, event: ComplianceEvent):
        """준수 이벤트 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO compliance_events 
                    (id, timestamp, event_type, severity, description, details, 
                     user_id, session_id, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.severity.value,
                    event.description,
                    json.dumps(event.details, default=str),
                    event.user_id,
                    event.session_id,
                    event.resolved
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving compliance event: {e}")
    
    def _save_metrics(self):
        """메트릭 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO compliance_metrics 
                    (timestamp, total_requests, compliant_requests, non_compliant_requests,
                     at_risk_requests, compliance_rate, average_response_time,
                     critical_events, warning_events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.metrics.last_updated.isoformat(),
                    self.metrics.total_requests,
                    self.metrics.compliant_requests,
                    self.metrics.non_compliant_requests,
                    self.metrics.at_risk_requests,
                    self.metrics.compliance_rate,
                    self.metrics.average_response_time,
                    self.metrics.critical_events,
                    self.metrics.warning_events
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def _send_critical_alert(self, event: ComplianceEvent):
        """심각한 경고 전송"""
        try:
            # 로그에 심각한 경고 기록
            compliance_logger = logging.getLogger("compliance_monitor")
            compliance_logger.critical(
                f"CRITICAL ALERT: {event.event_type} - {event.description}"
            )
            
            # 필요시 외부 알림 시스템 연동
            # self._send_external_alert(event)
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}")
    
    def get_compliance_report(self, days: int = 7) -> Dict[str, Any]:
        """준수 보고서 생성"""
        try:
            # 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 사용자 활동 통계
                cursor.execute("""
                    SELECT compliance_status, COUNT(*) as count
                    FROM user_activity
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY compliance_status
                """, (start_date.isoformat(), end_date.isoformat()))
                
                activity_stats = dict(cursor.fetchall())
                
                # 준수 이벤트 통계
                cursor.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM compliance_events
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY severity
                """, (start_date.isoformat(), end_date.isoformat()))
                
                event_stats = dict(cursor.fetchall())
                
                # 최근 이벤트
                cursor.execute("""
                    SELECT event_type, severity, description, timestamp
                    FROM compliance_events
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (start_date.isoformat(), end_date.isoformat()))
                
                recent_events = cursor.fetchall()
                
                # 준수율 계산
                total_requests = sum(activity_stats.values())
                compliant_requests = activity_stats.get('compliant', 0)
                compliance_rate = (compliant_requests / total_requests) if total_requests > 0 else 0
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "compliance_metrics": {
                        "total_requests": total_requests,
                        "compliant_requests": compliant_requests,
                        "non_compliant_requests": activity_stats.get('non_compliant', 0),
                        "at_risk_requests": activity_stats.get('at_risk', 0),
                        "compliance_rate": compliance_rate
                    },
                    "event_statistics": event_stats,
                    "recent_events": [
                        {
                            "event_type": event[0],
                            "severity": event[1],
                            "description": event[2],
                            "timestamp": event[3]
                        }
                        for event in recent_events
                    ],
                    "recommendations": self._generate_recommendations(activity_stats, event_stats)
                }
                
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "compliance_metrics": {
                    "total_requests": 0,
                    "compliant_requests": 0,
                    "non_compliant_requests": 0,
                    "at_risk_requests": 0,
                    "compliance_rate": 0.0
                },
                "event_statistics": {},
                "recent_events": [],
                "recommendations": ["데이터베이스 초기화가 필요합니다."]
            }
    
    def _generate_recommendations(self, activity_stats: Dict[str, int], 
                                event_stats: Dict[str, int]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 준수율이 낮으면 권장사항
        total_requests = sum(activity_stats.values())
        if total_requests > 0:
            compliance_rate = activity_stats.get('compliant', 0) / total_requests
            if compliance_rate < 0.8:
                recommendations.append("준수율이 낮습니다. 제한 규칙을 검토하고 강화하세요.")
        
        # 심각한 이벤트가 많으면 권장사항
        critical_events = event_stats.get('critical', 0)
        if critical_events > 0:
            recommendations.append("심각한 준수 위반이 발생했습니다. 즉시 조치가 필요합니다.")
        
        # 경고 이벤트가 많으면 권장사항
        warning_events = event_stats.get('warning', 0)
        if warning_events > 10:
            recommendations.append("경고 이벤트가 많이 발생했습니다. 모니터링을 강화하세요.")
        
        return recommendations
    
    def get_current_metrics(self) -> ComplianceMetrics:
        """현재 메트릭 반환"""
        return self.metrics
    
    def get_compliance_events(self, limit: int = 100) -> List[ComplianceEvent]:
        """준수 이벤트 목록 반환"""
        return self.compliance_events[-limit:] if self.compliance_events else []
    
    def resolve_event(self, event_id: str) -> bool:
        """이벤트 해결"""
        try:
            # 메모리에서 이벤트 찾기
            for event in self.compliance_events:
                if event.id == event_id:
                    event.resolved = True
                    
                    # 데이터베이스 업데이트
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE compliance_events 
                            SET resolved = TRUE 
                            WHERE id = ?
                        """, (event_id,))
                        conn.commit()
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving event: {e}")
            return False
