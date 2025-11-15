"""
로깅 보안 유틸리티
민감 정보 필터링
"""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 민감 정보 패턴
SENSITIVE_PATTERNS = [
    r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    r'jwt[_-]?secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    r'secret[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    r'authorization["\']?\s*[:=]\s*["\']?bearer\s+([^"\'\s]+)',
    r'authorization["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
]


def mask_sensitive_info(text: str) -> str:
    """민감 정보 마스킹"""
    if not isinstance(text, str):
        return str(text)
    
    masked_text = text
    for pattern in SENSITIVE_PATTERNS:
        masked_text = re.sub(
            pattern,
            lambda m: m.group(0).replace(m.group(1), "***MASKED***") if len(m.groups()) > 0 else m.group(0),
            masked_text,
            flags=re.IGNORECASE
        )
    
    return masked_text


class SecureFormatter(logging.Formatter):
    """민감 정보를 필터링하는 로깅 포매터"""
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드 포맷팅 시 민감 정보 마스킹"""
        original_msg = record.getMessage()
        masked_msg = mask_sensitive_info(original_msg)
        
        if original_msg != masked_msg:
            record.msg = masked_msg
            record.args = ()
        
        return super().format(record)


def setup_secure_logging():
    """보안 로깅 설정"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler.formatter, logging.Formatter):
            handler.setFormatter(SecureFormatter(handler.formatter._fmt))

