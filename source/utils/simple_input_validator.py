# -*- coding: utf-8 -*-
"""
간소화된 입력 검증 시스템 (정규식 오류 수정)
"""

import re
import html
import json
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
from .security_logger import get_security_logger, SecurityEventType, SecurityLevel


class ValidationResult(Enum):
    """검증 결과"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


@dataclass
class ValidationReport:
    """검증 보고서"""
    result: ValidationResult
    message: str
    violations: List[str]
    sanitized_input: str
    risk_score: float  # 0.0 ~ 1.0


class SimpleInputValidator:
    """간소화된 입력 검증 시스템"""
    
    def __init__(self):
        self.security_logger = get_security_logger()
        
        # 간소화된 악성 패턴 (정규식 오류 방지)
        self.malicious_patterns = {
            'xss': [
                r'<script',
                r'javascript:',
                r'onload=',
                r'onerror=',
                r'<iframe',
                r'<object',
                r'<embed',
                r'<link',
                r'<meta',
                r'<style',
                r'expression\(',
                r'url\(',
            ],
            'sql_injection': [
                r'union\s+select',
                r'drop\s+table',
                r'delete\s+from',
                r'insert\s+into',
                r'update\s+set',
                r'alter\s+table',
                r'create\s+table',
                r'exec\s*\(',
                r'execute\s*\(',
                r'sp_',
                r'xp_',
                r'--',
                r'/\*',
                r'\*/',
            ],
            'command_injection': [
                r';\s*rm\s+',
                r';\s*del\s+',
                r';\s*format\s+',
                r';\s*shutdown',
                r';\s*reboot',
                r';\s*halt',
                r';\s*kill\s+',
                r';\s*ps\s+',
                r';\s*cat\s+',
                r';\s*type\s+',
                r';\s*ls\s+',
                r';\s*dir\s+',
                r';\s*cd\s+',
                r';\s*pwd',
                r';\s*whoami',
                r';\s*echo\s+',
                r';\s*wget\s+',
                r';\s*curl\s+',
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'\.\.%2f',
                r'\.\.%5c',
                r'\.\.%252f',
                r'\.\.%255c',
            ],
        }
        
        # 허용된 문자 패턴 (수정됨)
        self.allowed_pattern = re.compile(r'^[가-힣a-zA-Z0-9\s.,!?()\-_]+$')
        
        # 최대 길이 제한
        self.max_length = 10000
        
        # 의심스러운 키워드
        self.suspicious_keywords = [
            'admin', 'administrator', 'root', 'system', 'config', 'password',
            'secret', 'key', 'token', 'session', 'cookie', 'login', 'auth',
            'bypass', 'hack', 'exploit', 'vulnerability', 'injection', 'xss',
            'sql', 'script', 'javascript', 'vbscript', 'eval', 'exec',
            'system', 'shell', 'cmd', 'command', 'run', 'download', 'upload',
            'file', 'path', 'directory', 'folder', 'backup', 'restore',
            'import', 'export', 'dump', 'load', 'delete', 'remove', 'drop',
            'truncate', 'alter', 'create', 'insert', 'update', 'select',
            'union', 'join', 'where', 'from', 'into', 'values', 'set',
            'table', 'database', 'db', 'user', 'group', 'role', 'permission',
            'access', 'grant', 'revoke', 'deny', 'allow', 'block', 'ban',
            'unban', 'kick', 'mute', 'unmute', 'warn', 'warning', 'alert',
            'notice', 'error', 'exception', 'fault', 'bug', 'issue', 'problem',
            'fix', 'patch', 'update', 'upgrade', 'downgrade', 'rollback',
            'restart', 'reboot', 'shutdown', 'stop', 'start', 'pause',
            'resume', 'continue', 'abort', 'cancel', 'exit', 'quit',
            'close', 'open', 'lock', 'unlock', 'secure', 'unsecure',
            'encrypt', 'decrypt', 'hash', 'encode', 'decode', 'compress',
            'decompress', 'archive', 'extract', 'zip', 'unzip', 'tar',
            'gzip', 'gunzip', 'bzip2', 'bunzip2', '7z', 'rar', 'unrar',
        ]
    
    def validate_input(self, 
                      input_data: str,
                      user_id: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      session_id: Optional[str] = None) -> ValidationReport:
        """간소화된 입력 검증"""
        
        violations = []
        risk_score = 0.0
        
        # 1. 기본 길이 검증
        if len(input_data) > self.max_length:
            violations.append(f"입력 길이 초과: {len(input_data)} > {self.max_length}")
            risk_score += 0.3
        
        # 2. 빈 입력 검증
        if not input_data.strip():
            violations.append("빈 입력")
            risk_score += 0.1
        
        # 3. 악성 패턴 검증 (안전한 방식)
        for pattern_type, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, input_data, re.IGNORECASE):
                        violations.append(f"{pattern_type.upper()} 패턴 탐지: {pattern}")
                        risk_score += 0.4
                except re.error:
                    # 정규식 오류 시 무시
                    continue
        
        # 4. 의심스러운 키워드 검증
        input_lower = input_data.lower()
        for keyword in self.suspicious_keywords:
            if keyword in input_lower:
                violations.append(f"의심스러운 키워드: {keyword}")
                risk_score += 0.1
        
        # 5. 허용된 문자 패턴 검증 (안전한 방식)
        try:
            if not self.allowed_pattern.match(input_data):
                violations.append("허용되지 않은 문자가 포함됨")
                risk_score += 0.2
        except re.error:
            # 정규식 오류 시 무시
            pass
        
        # 6. 연속된 특수문자 검증
        if re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{3,}', input_data):
            violations.append("연속된 특수문자")
            risk_score += 0.2
        
        # 7. HTML 태그 검증
        if re.search(r'<[^>]+>', input_data):
            violations.append("HTML 태그 포함")
            risk_score += 0.3
        
        # 8. URL 패턴 검증
        if re.search(r'https?://[^\s]+', input_data):
            violations.append("URL 포함")
            risk_score += 0.2
        
        # 9. 이메일 패턴 검증
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', input_data):
            violations.append("이메일 주소 포함")
            risk_score += 0.1
        
        # 10. 전화번호 패턴 검증
        if re.search(r'\d{2,3}-\d{3,4}-\d{4}', input_data):
            violations.append("전화번호 포함")
            risk_score += 0.1
        
        # 11. 주민번호 패턴 검증
        if re.search(r'\d{6}-\d{7}', input_data):
            violations.append("주민번호 포함")
            risk_score += 0.3
        
        # 12. 신용카드 번호 패턴 검증
        if re.search(r'\d{4}-\d{4}-\d{4}-\d{4}', input_data):
            violations.append("신용카드 번호 포함")
            risk_score += 0.3
        
        # 13. IP 주소 패턴 검증
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', input_data):
            violations.append("IP 주소 포함")
            risk_score += 0.2
        
        # 위험 점수 정규화
        risk_score = min(risk_score, 1.0)
        
        # 검증 결과 결정
        if risk_score >= 0.8:
            result = ValidationResult.BLOCKED
            message = "입력이 차단되었습니다. 보안상 위험한 내용이 포함되어 있습니다."
        elif risk_score >= 0.5:
            result = ValidationResult.SUSPICIOUS
            message = "입력이 의심스럽습니다. 검토가 필요합니다."
        elif risk_score >= 0.2:
            result = ValidationResult.INVALID
            message = "입력이 유효하지 않습니다. 일부 내용을 수정해주세요."
        else:
            result = ValidationResult.VALID
            message = "입력이 유효합니다."
        
        # 입력 정화
        sanitized_input = self.sanitize_input(input_data)
        
        # 보안 로그 기록
        self.security_logger.log_input_validation(
            input_data,
            result == ValidationResult.VALID,
            user_id,
            ip_address,
            session_id,
            {
                'violations': violations,
                'risk_score': risk_score,
                'sanitized_length': len(sanitized_input)
            }
        )
        
        return ValidationReport(
            result=result,
            message=message,
            violations=violations,
            sanitized_input=sanitized_input,
            risk_score=risk_score
        )
    
    def sanitize_input(self, input_data: str) -> str:
        """입력 데이터 정화"""
        # HTML 이스케이프
        sanitized = html.escape(input_data)
        
        # 특수문자 제거 (일부 허용)
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        # 연속된 공백 정리
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # 앞뒤 공백 제거
        sanitized = sanitized.strip()
        
        return sanitized


# 전역 간소화된 입력 검증기 인스턴스
simple_input_validator = SimpleInputValidator()


def get_simple_input_validator() -> SimpleInputValidator:
    """간소화된 입력 검증기 인스턴스 반환"""
    return simple_input_validator
