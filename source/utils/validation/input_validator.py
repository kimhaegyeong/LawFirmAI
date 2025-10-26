# -*- coding: utf-8 -*-
"""
Input Validation and Security
입력 검증 및 보안 시스템
"""

import re
import html
import json
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
from ..security.security_logger import get_security_logger, SecurityEventType, SecurityLevel


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


class InputValidator:
    """포괄적인 입력 검증 시스템"""

    def __init__(self):
        self.security_logger = get_security_logger()

        # 악성 패턴 정의
        self.malicious_patterns = {
            'xss': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',
                r'<link[^>]*>',
                r'<meta[^>]*>',
                r'<style[^>]*>.*?</style>',
                r'expression\s*\(',
                r'url\s*\(',
                r'@import',
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
                r'/\*.*?\*/',
                r';\s*drop',
                r';\s*delete',
                r';\s*insert',
                r';\s*update',
                r';\s*alter',
                r';\s*create',
            ],
            'command_injection': [
                r';\s*rm\s+',
                r';\s*del\s+',
                r';\s*format\s+',
                r';\s*shutdown',
                r';\s*reboot',
                r';\s*halt',
                r';\s*poweroff',
                r';\s*init\s+\d',
                r';\s*kill\s+',
                r';\s*ps\s+',
                r';\s*netstat',
                r';\s*ifconfig',
                r';\s*ipconfig',
                r';\s*ping\s+',
                r';\s*nslookup',
                r';\s*tracert',
                r';\s*route',
                r';\s*arp',
                r';\s*cat\s+',
                r';\s*type\s+',
                r';\s*more\s+',
                r';\s*less\s+',
                r';\s*head\s+',
                r';\s*tail\s+',
                r';\s*grep\s+',
                r';\s*find\s+',
                r';\s*ls\s+',
                r';\s*dir\s+',
                r';\s*cd\s+',
                r';\s*pwd',
                r';\s*whoami',
                r';\s*id\s*$',
                r';\s*uname',
                r';\s*env\s*$',
                r';\s*set\s*$',
                r';\s*echo\s+',
                r';\s*print\s+',
                r';\s*printf\s+',
                r';\s*wget\s+',
                r';\s*curl\s+',
                r';\s*nc\s+',
                r';\s*netcat\s+',
                r';\s*telnet\s+',
                r';\s*ssh\s+',
                r';\s*ftp\s+',
                r';\s*scp\s+',
                r';\s*rsync\s+',
                r';\s*tar\s+',
                r';\s*zip\s+',
                r';\s*unzip\s+',
                r';\s*gzip\s+',
                r';\s*gunzip\s+',
                r';\s*bzip2\s+',
                r';\s*bunzip2\s+',
                r';\s*7z\s+',
                r';\s*rar\s+',
                r';\s*unrar\s+',
                r';\s*cp\s+',
                r';\s*mv\s+',
                r';\s*mkdir\s+',
                r';\s*rmdir\s+',
                r';\s*touch\s+',
                r';\s*chmod\s+',
                r';\s*chown\s+',
                r';\s*chgrp\s+',
                r';\s*ln\s+',
                r';\s*symlink\s+',
                r';\s*du\s+',
                r';\s*df\s+',
                r';\s*free\s*$',
                r';\s*top\s*$',
                r';\s*htop\s*$',
                r';\s*ps\s+',
                r';\s*killall\s+',
                r';\s*pkill\s+',
                r';\s*kill\s+',
                r';\s*nohup\s+',
                r';\s*bg\s*$',
                r';\s*fg\s*$',
                r';\s*jobs\s*$',
                r';\s*history\s*$',
                r';\s*alias\s+',
                r';\s*unalias\s+',
                r';\s*export\s+',
                r';\s*unset\s+',
                r';\s*readonly\s+',
                r';\s*declare\s+',
                r';\s*typeset\s+',
                r';\s*local\s+',
                r';\s*function\s+',
                r';\s*return\s+',
                r';\s*exit\s*$',
                r';\s*logout\s*$',
                r';\s*source\s+',
                r';\s*\.\s+',
                r';\s*eval\s+',
                r';\s*exec\s+',
                r';\s*command\s+',
                r';\s*builtin\s+',
                r';\s*enable\s+',
                r';\s*disable\s+',
                r';\s*compgen\s+',
                r';\s*complete\s+',
                r';\s*compopt\s+',
                r';\s*compctl\s+',
                r';\s*compadd\s+',
                r';\s*compcall\s+',
                r';\s*compdef\s+',
                r';\s*compdescribe\s+',
                r';\s*compfiles\s+',
                r';\s*compgroups\s+',
                r';\s*compquote\s+',
                r';\s*comptags\s+',
                r';\s*comptry\s+',
                r';\s*compvalues\s+',
                r';\s*compwidgets\s+',
                r';\s*compadd\s+',
                r';\s*compcall\s+',
                r';\s*compdef\s+',
                r';\s*compdescribe\s+',
                r';\s*compfiles\s+',
                r';\s*compgroups\s+',
                r';\s*compquote\s+',
                r';\s*comptags\s+',
                r';\s*comptry\s+',
                r';\s*compvalues\s+',
                r';\s*compwidgets\s+',
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'\.\.%2f',
                r'\.\.%5c',
                r'\.\.%252f',
                r'\.\.%255c',
                r'\.\.%c0%af',
                r'\.\.%c1%9c',
                r'\.\.%c0%2f',
                r'\.\.%c1%af',
                r'\.\.%e0%80%af',
                r'\.\.%f0%80%80%af',
                r'\.\.%f8%80%80%80%af',
                r'\.\.%fc%80%80%80%80%af',
                r'\.\.%2e%2e%2f',
                r'\.\.%2e%2e%5c',
                r'\.\.%252e%252e%252f',
                r'\.\.%252e%252e%255c',
                r'\.\.%c0%ae%c0%ae%c0%af',
                r'\.\.%c0%ae%c0%ae%c1%9c',
                r'\.\.%c1%9c%c1%9c%c0%af',
                r'\.\.%c1%9c%c1%9c%c1%9c',
            ],
            'ldap_injection': [
                r'\([^)]*\)',
                r'&[^&]*',
                r'\|[^|]*',
                r'![^!]*',
                r'=',
                r'<',
                r'>',
                r'~',
                r'\*',
                r'\(&',
                r'\(!',
                r'\(|',
                r'\(=',
                r'\(<',
                r'\(>',
                r'\(~',
                r'\(\*',
            ],
            'xml_injection': [
                r'<!DOCTYPE',
                r'<!ENTITY',
                r'<![CDATA[',
                r'<?xml',
                r'<xsl:',
                r'<script',
                r'<iframe',
                r'<object',
                r'<embed',
                r'<link',
                r'<meta',
                r'<style',
                r'<form',
                r'<input',
                r'<textarea',
                r'<select',
                r'<option',
                r'<button',
                r'<a\s+href',
                r'<img\s+src',
                r'<video\s+src',
                r'<audio\s+src',
                r'<source\s+src',
                r'<track\s+src',
                r'<area\s+href',
                r'<base\s+href',
                r'<param\s+value',
                r'<applet\s+code',
                r'<applet\s+codebase',
                r'<object\s+data',
                r'<object\s+codebase',
                r'<embed\s+src',
                r'<iframe\s+src',
                r'<frame\s+src',
                r'<frameset',
                r'<noframes',
                r'<noscript',
                r'<script',
                r'<style',
                r'<link',
                r'<meta',
                r'<title',
                r'<head',
                r'<body',
                r'<html',
                r'<xml',
                r'<svg',
                r'<math',
                r'<mi',
                r'<mo',
                r'<mn',
                r'<ms',
                r'<mtext',
                r'<mspace',
                r'<msline',
                r'<mfrac',
                r'<msqrt',
                r'<mroot',
                r'<msub',
                r'<msup',
                r'<msubsup',
                r'<munder',
                r'<mover',
                r'<munderover',
                r'<mtable',
                r'<mtr',
                r'<mtd',
                r'<mlabeledtr',
                r'<maligngroup',
                r'<malignmark',
                r'<mgroup',
                r'<maction',
                r'<merror',
                r'<mpadded',
                r'<mphantom',
                r'<mrow',
                r'<mscarries',
                r'<mscarry',
                r'<msline',
                r'<mstack',
                r'<mlongdiv',
                r'<msgroup',
                r'<msrow',
                r'<mscolumn',
                r'<mscarries',
                r'<mscarry',
                r'<msline',
                r'<mstack',
                r'<mlongdiv',
                r'<msgroup',
                r'<msrow',
                r'<mscolumn',
                r'<mscarries',
                r'<mscarry',
                r'<msline',
                r'<mstack',
                r'<mlongdiv',
                r'<msgroup',
                r'<msrow',
                r'<mscolumn',
            ],
            'no_sql_injection': [
                r'\$where',
                r'\$ne',
                r'\$gt',
                r'\$gte',
                r'\$lt',
                r'\$lte',
                r'\$in',
                r'\$nin',
                r'\$exists',
                r'\$regex',
                r'\$text',
                r'\$search',
                r'\$geoWithin',
                r'\$geoIntersects',
                r'\$near',
                r'\$nearSphere',
                r'\$all',
                r'\$elemMatch',
                r'\$size',
                r'\$type',
                r'\$mod',
                r'\$bitsAllSet',
                r'\$bitsAnySet',
                r'\$bitsAllClear',
                r'\$bitsAnyClear',
                r'\$rand',
                r'\$expr',
                r'\$jsonSchema',
                r'\$comment',
                r'\$hint',
                r'\$max',
                r'\$min',
                r'\$orderby',
                r'\$returnKey',
                r'\$showDiskLoc',
                r'\$natural',
                r'\$explain',
                r'\$snapshot',
                r'\$tailable',
                r'\$oplogReplay',
                r'\$noCursorTimeout',
                r'\$awaitData',
                r'\$partial',
                r'\$allowDiskUse',
                r'\$collation',
                r'\$readConcern',
                r'\$writeConcern',
                r'\$readPreference',
                r'\$maxTimeMS',
                r'\$minTimeMS',
                r'\$maxScan',
                r'\$returnKey',
                r'\$showDiskLoc',
                r'\$natural',
                r'\$explain',
                r'\$snapshot',
                r'\$tailable',
                r'\$oplogReplay',
                r'\$noCursorTimeout',
                r'\$awaitData',
                r'\$partial',
                r'\$allowDiskUse',
                r'\$collation',
                r'\$readConcern',
                r'\$writeConcern',
                r'\$readPreference',
                r'\$maxTimeMS',
                r'\$minTimeMS',
                r'\$maxScan',
            ],
        }

        # 허용된 문자 패턴 (한글, 영문, 숫자, 기본 특수문자)
        self.allowed_pattern = re.compile(r'^[가-힣a-zA-Z0-9\s.,!?()\-_]+$')

        # 최대 길이 제한
        self.max_length = 10000

        # 의심스러운 키워드
        self.suspicious_keywords = [
            'admin', 'administrator', 'root', 'system', 'config', 'password',
            'secret', 'key', 'token', 'session', 'cookie', 'login', 'auth',
            'bypass', 'hack', 'exploit', 'vulnerability', 'injection', 'xss',
            'sql', 'script', 'javascript', 'vbscript', 'onload', 'onerror',
            'eval', 'exec', 'system', 'shell', 'cmd', 'command', 'run',
            'download', 'upload', 'file', 'path', 'directory', 'folder',
            'backup', 'restore', 'import', 'export', 'dump', 'load',
            'delete', 'remove', 'drop', 'truncate', 'alter', 'create',
            'insert', 'update', 'select', 'union', 'join', 'where',
            'from', 'into', 'values', 'set', 'table', 'database', 'db',
            'user', 'group', 'role', 'permission', 'access', 'grant',
            'revoke', 'deny', 'allow', 'block', 'ban', 'unban', 'kick',
            'mute', 'unmute', 'warn', 'warning', 'alert', 'notice',
            'error', 'exception', 'fault', 'bug', 'issue', 'problem',
            'fix', 'patch', 'update', 'upgrade', 'downgrade', 'rollback',
            'restart', 'reboot', 'shutdown', 'stop', 'start', 'pause',
            'resume', 'continue', 'abort', 'cancel', 'exit', 'quit',
            'close', 'open', 'lock', 'unlock', 'secure', 'unsecure',
            'encrypt', 'decrypt', 'hash', 'encode', 'decode', 'compress',
            'decompress', 'archive', 'extract', 'zip', 'unzip', 'tar',
            'gzip', 'gunzip', 'bzip2', 'bunzip2', '7z', 'rar', 'unrar',
            'copy', 'move', 'rename', 'link', 'unlink', 'symlink',
            'chmod', 'chown', 'chgrp', 'umask', 'su', 'sudo', 'suid',
            'sgid', 'sticky', 'setuid', 'setgid', 'seteuid', 'setegid',
            'setreuid', 'setregid', 'setresuid', 'setresgid', 'setfsuid',
            'setfsgid', 'getuid', 'getgid', 'geteuid', 'getegid', 'getpuid',
            'getpgid', 'getppid', 'getpid', 'gettid', 'gettid', 'gettid',
            'getpgrp', 'getsid', 'getpgid', 'getppid', 'getpid', 'gettid',
            'getpgrp', 'getsid', 'getpgid', 'getppid', 'getpid', 'gettid',
            'getpgrp', 'getsid', 'getpgid', 'getppid', 'getpid', 'gettid',
        ]

    def validate_input(self,
                      input_data: str,
                      user_id: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      session_id: Optional[str] = None) -> ValidationReport:
        """포괄적인 입력 검증"""

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

        # 3. 악성 패턴 검증
        for pattern_type, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    violations.append(f"{pattern_type.upper()} 패턴 탐지: {pattern}")
                    risk_score += 0.4

        # 4. 의심스러운 키워드 검증
        input_lower = input_data.lower()
        for keyword in self.suspicious_keywords:
            if keyword in input_lower:
                violations.append(f"의심스러운 키워드: {keyword}")
                risk_score += 0.1

        # 5. 허용된 문자 패턴 검증
        if not self.allowed_pattern.match(input_data):
            violations.append("허용되지 않은 문자가 포함됨")
            risk_score += 0.2

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

        # 14. 파일 경로 패턴 검증
        if re.search(r'[C-Z]:\\[^\\]+', input_data) or re.search(r'/[^/]+', input_data):
            violations.append("파일 경로 포함")
            risk_score += 0.2

        # 15. Base64 패턴 검증
        if re.search(r'[A-Za-z0-9+/]{4,}={0,2}', input_data):
            violations.append("Base64 인코딩 패턴")
            risk_score += 0.2

        # 16. Hex 패턴 검증
        if re.search(r'[0-9a-fA-F]{8,}', input_data):
            violations.append("Hex 인코딩 패턴")
            risk_score += 0.2

        # 17. Unicode 이스케이프 패턴 검증
        if re.search(r'\\u[0-9a-fA-F]{4}', input_data):
            violations.append("Unicode 이스케이프 패턴")
            risk_score += 0.2

        # 18. SQL 주석 패턴 검증
        if re.search(r'--|/\*.*?\*/', input_data):
            violations.append("SQL 주석 패턴")
            risk_score += 0.3

        # 19. 명령어 체이닝 패턴 검증
        if re.search(r'[;&|]\s*[a-zA-Z]', input_data):
            violations.append("명령어 체이닝 패턴")
            risk_score += 0.4

        # 20. 환경변수 패턴 검증
        if re.search(r'\$[A-Za-z_][A-Za-z0-9_]*', input_data):
            violations.append("환경변수 패턴")
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

    def validate_json_input(self, json_data: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """JSON 입력 검증"""
        try:
            parsed_data = json.loads(json_data)

            # JSON 깊이 제한
            if self._get_json_depth(parsed_data) > 10:
                return False, None

            # JSON 크기 제한
            if len(json_data) > 100000:  # 100KB
                return False, None

            return True, parsed_data

        except json.JSONDecodeError:
            return False, None

    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """JSON 객체의 깊이 계산"""
        if isinstance(obj, dict):
            return max((self._get_json_depth(v, depth + 1) for v in obj.values()), default=depth)
        elif isinstance(obj, list):
            return max((self._get_json_depth(item, depth + 1) for item in obj), default=depth)
        else:
            return depth


# 전역 입력 검증기 인스턴스
input_validator = InputValidator()


def get_input_validator() -> InputValidator:
    """입력 검증기 인스턴스 반환"""
    return input_validator
