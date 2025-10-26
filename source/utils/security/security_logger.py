# -*- coding: utf-8 -*-
"""
Security Audit Logger
보안 감사 로그 시스템
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import hashlib


class SecurityEventType(Enum):
    """보안 이벤트 유형"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    INPUT_VALIDATION = "input_validation"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"


class SecurityLevel(Enum):
    """보안 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAuditLogger:
    """보안 감사 로그 시스템"""
    
    def __init__(self, log_dir: str = "logs/security"):
        """보안 감사 로그 초기화"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 보안 로그 파일 설정
        self.security_log_file = self.log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 로거 설정
        self.logger = logging.getLogger('security_audit')
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        self.logger.handlers.clear()
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(self.security_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 로그 로테이션 설정
        self._setup_log_rotation()
        
        # 초기화 로그
        self.log_event(
            SecurityEventType.SYSTEM_ACCESS,
            SecurityLevel.LOW,
            "Security audit logger initialized",
            {"log_file": str(self.security_log_file)}
        )
    
    def _setup_log_rotation(self):
        """로그 로테이션 설정"""
        from logging.handlers import RotatingFileHandler
        
        # 기존 핸들러 제거
        self.logger.handlers.clear()
        
        # 로테이팅 파일 핸들러 설정
        rotating_handler = RotatingFileHandler(
            self.security_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=30,  # 30일간 보관
            encoding='utf-8'
        )
        rotating_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        rotating_handler.setFormatter(formatter)
        
        self.logger.addHandler(rotating_handler)
    
    def log_event(self, 
                  event_type: SecurityEventType,
                  level: SecurityLevel,
                  message: str,
                  details: Optional[Dict[str, Any]] = None,
                  user_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  session_id: Optional[str] = None) -> str:
        """보안 이벤트 로그 기록"""
        
        # 이벤트 ID 생성 (해시 기반)
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type.value,
            'level': level.value,
            'message': message,
            'user_id': user_id,
            'ip_address': ip_address,
            'session_id': session_id,
            'details': details or {}
        }
        
        event_id = hashlib.sha256(
            json.dumps(event_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        event_data['event_id'] = event_id
        
        # 로그 레벨에 따른 처리
        log_level = self._get_log_level(level)
        
        # 로그 메시지 포맷팅
        log_message = self._format_log_message(event_data)
        
        # 로그 기록
        self.logger.log(log_level, log_message)
        
        # 중요도가 높은 이벤트는 별도 파일에도 기록
        if level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self._log_critical_event(event_data)
        
        return event_id
    
    def _get_log_level(self, security_level: SecurityLevel) -> int:
        """보안 레벨을 로그 레벨로 변환"""
        level_mapping = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }
        return level_mapping[security_level]
    
    def _format_log_message(self, event_data: Dict[str, Any]) -> str:
        """로그 메시지 포맷팅"""
        return json.dumps(event_data, ensure_ascii=False, indent=None)
    
    def _log_critical_event(self, event_data: Dict[str, Any]):
        """중요 이벤트 별도 기록"""
        critical_log_file = self.log_dir / f"critical_events_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(critical_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + '\n')
    
    def log_authentication(self, 
                          user_id: str,
                          success: bool,
                          ip_address: Optional[str] = None,
                          session_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None) -> str:
        """인증 이벤트 로그"""
        level = SecurityLevel.LOW if success else SecurityLevel.HIGH
        message = f"Authentication {'successful' if success else 'failed'} for user {user_id}"
        
        return self.log_event(
            SecurityEventType.AUTHENTICATION,
            level,
            message,
            details,
            user_id,
            ip_address,
            session_id
        )
    
    def log_data_access(self,
                       user_id: str,
                       resource: str,
                       action: str,
                       success: bool = True,
                       ip_address: Optional[str] = None,
                       session_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """데이터 접근 이벤트 로그"""
        level = SecurityLevel.LOW if success else SecurityLevel.MEDIUM
        message = f"Data access: {action} on {resource} by user {user_id}"
        
        access_details = {
            'resource': resource,
            'action': action,
            'success': success,
            **(details or {})
        }
        
        return self.log_event(
            SecurityEventType.DATA_ACCESS,
            level,
            message,
            access_details,
            user_id,
            ip_address,
            session_id
        )
    
    def log_security_violation(self,
                              violation_type: str,
                              description: str,
                              user_id: Optional[str] = None,
                              ip_address: Optional[str] = None,
                              session_id: Optional[str] = None,
                              details: Optional[Dict[str, Any]] = None) -> str:
        """보안 위반 이벤트 로그"""
        message = f"Security violation: {violation_type} - {description}"
        
        violation_details = {
            'violation_type': violation_type,
            'description': description,
            **(details or {})
        }
        
        return self.log_event(
            SecurityEventType.SECURITY_VIOLATION,
            SecurityLevel.CRITICAL,
            message,
            violation_details,
            user_id,
            ip_address,
            session_id
        )
    
    def log_input_validation(self,
                            input_data: str,
                            validation_result: bool,
                            user_id: Optional[str] = None,
                            ip_address: Optional[str] = None,
                            session_id: Optional[str] = None,
                            details: Optional[Dict[str, Any]] = None) -> str:
        """입력 검증 이벤트 로그"""
        level = SecurityLevel.LOW if validation_result else SecurityLevel.MEDIUM
        message = f"Input validation {'passed' if validation_result else 'failed'}"
        
        validation_details = {
            'input_length': len(input_data),
            'validation_result': validation_result,
            'input_preview': input_data[:100] + "..." if len(input_data) > 100 else input_data,
            **(details or {})
        }
        
        return self.log_event(
            SecurityEventType.INPUT_VALIDATION,
            level,
            message,
            validation_details,
            user_id,
            ip_address,
            session_id
        )
    
    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """보안 보고서 생성"""
        report = {
            'period_days': days,
            'generated_at': datetime.now().isoformat(),
            'total_events': 0,
            'events_by_type': {},
            'events_by_level': {},
            'critical_events': [],
            'security_violations': []
        }
        
        # 로그 파일 분석 (간단한 구현)
        try:
            log_files = list(self.log_dir.glob("security_audit_*.log"))
            for log_file in log_files[-days:]:  # 최근 N일간의 로그
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            report['total_events'] += 1
                            
                            # 이벤트 유형별 집계
                            event_type = event_data.get('event_type', 'unknown')
                            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1
                            
                            # 레벨별 집계
                            level = event_data.get('level', 'unknown')
                            report['events_by_level'][level] = report['events_by_level'].get(level, 0) + 1
                            
                            # 중요 이벤트 수집
                            if level in ['high', 'critical']:
                                report['critical_events'].append(event_data)
                            
                            # 보안 위반 수집
                            if event_type == 'security_violation':
                                report['security_violations'].append(event_data)
                                
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            report['error'] = str(e)
        
        return report


# 전역 보안 로거 인스턴스
security_logger = SecurityAuditLogger()


def get_security_logger() -> SecurityAuditLogger:
    """보안 로거 인스턴스 반환"""
    return security_logger
