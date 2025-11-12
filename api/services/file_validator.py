"""
파일 검증 서비스 (보안)
"""
import logging
import base64
import re
from typing import Tuple, Optional, NamedTuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 허용된 파일 타입 정의
ALLOWED_MIME_TYPES = {
    'image/jpeg': {'ext': ['.jpg', '.jpeg'], 'max_size_mb': 10},
    'image/png': {'ext': ['.png'], 'max_size_mb': 10},
    'image/gif': {'ext': ['.gif'], 'max_size_mb': 10},
    'image/webp': {'ext': ['.webp'], 'max_size_mb': 10},
    'text/plain': {'ext': ['.txt'], 'max_size_mb': 5},
    'text/csv': {'ext': ['.csv'], 'max_size_mb': 5},
    'application/pdf': {'ext': ['.pdf'], 'max_size_mb': 20},
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {
        'ext': ['.docx'], 'max_size_mb': 15
    },
    'application/msword': {'ext': ['.doc'], 'max_size_mb': 15},
}

DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
    '.jar', '.app', '.deb', '.rpm', '.sh', '.ps1', '.dll', '.so',
    '.dylib', '.zip', '.rar', '.7z', '.tar', '.gz'
}

DANGEROUS_PATTERNS = [
    r'\.\./', r'\.\.\\', r'^/', r'^\\', r'[<>:"|?*]',
]

ERROR_MESSAGES = {
    'INVALID_DATA': "파일 데이터가 올바르지 않습니다.",
    'INVALID_BASE64_FORMAT': "Base64 인코딩 형식이 올바르지 않습니다.",
    'BASE64_DECODE_FAILED': "Base64 디코딩 실패: {error}",
    'FILE_TOO_LARGE': "파일 크기가 {max_size}MB를 초과합니다.",
    'UNSUPPORTED_MIME_TYPE': "허용되지 않은 파일 타입입니다: {mime_type}",
    'MIME_TYPE_UNKNOWN': "파일 타입을 확인할 수 없습니다.",
    'MIME_TYPE_NOT_SUPPORTED': "지원하지 않는 파일 타입입니다: {mime_type}",
    'EXTENSION_MISMATCH': "파일 확장자가 MIME 타입과 일치하지 않습니다.",
    'VALIDATION_ERROR': "파일 검증 중 오류가 발생했습니다: {error}",
    'FILENAME_EMPTY': "파일명이 비어있습니다.",
    'FILENAME_TOO_LONG': "파일명이 너무 깁니다 (최대 255자).",
    'FILENAME_INVALID_CHARS': "파일명에 허용되지 않은 문자가 포함되어 있습니다.",
    'FILENAME_DANGEROUS_EXT': "허용되지 않은 파일 확장자입니다: {ext}",
    'FILENAME_PATH_TRAVERSAL': "파일명에 경로 정보가 포함되어 있습니다.",
    'FILENAME_VALIDATION_ERROR': "파일명 검증 중 오류가 발생했습니다: {error}",
}


class FileInfo(NamedTuple):
    """파일 정보"""
    mime_type: str
    size: int
    extension: Optional[str]
    bytes: bytes
    
    def __getitem__(self, key):
        """딕셔너리처럼 접근 가능하도록 지원"""
        if key == 'mime_type':
            return self.mime_type
        elif key == 'size':
            return self.size
        elif key == 'extension':
            return self.extension
        elif key == 'bytes':
            return self.bytes
        raise KeyError(key)
    
    def get(self, key, default=None):
        """딕셔너리처럼 get 메서드 지원"""
        try:
            return self[key]
        except KeyError:
            return default


class ValidationResult(NamedTuple):
    """검증 결과"""
    is_valid: bool
    error_message: Optional[str]
    file_info: Optional[FileInfo]


def validate_file_base64(
    file_base64: str,
    filename: Optional[str] = None,
    max_total_size_mb: int = 50
) -> Tuple[bool, Optional[str], Optional[FileInfo]]:
    """
    Base64 인코딩된 파일 검증
    
    Args:
        file_base64: Base64 인코딩된 파일
        filename: 파일명 (선택사항)
        max_total_size_mb: 최대 전체 크기 (MB)
        
    Returns:
        (is_valid, error_message, file_info)
        file_info: FileInfo 타입 (딕셔너리처럼 접근 가능)
    """
    try:
        decode_result = _decode_base64(file_base64)
        if not decode_result[0]:
            return False, decode_result[1], None
        mime_type, file_bytes = decode_result[1], decode_result[2]
        
        size_check = _validate_total_size(file_bytes, max_total_size_mb)
        if not size_check[0]:
            return False, size_check[1], None
        
        if filename:
            name_check = validate_filename(filename)
            if not name_check[0]:
                return False, name_check[1], None
        
        mime_result = _validate_and_determine_mime(file_bytes, mime_type)
        if not mime_result[0]:
            return False, mime_result[1], None
        final_mime = mime_result[1]
        
        size_limit_check = _validate_file_type_size(file_bytes, final_mime)
        if not size_limit_check[0]:
            return False, size_limit_check[1], None
        
        if filename:
            ext_check = _validate_extension(filename, final_mime)
            if not ext_check[0]:
                return False, ext_check[1], None
        
        file_info = FileInfo(
            mime_type=final_mime,
            size=len(file_bytes),
            extension=Path(filename).suffix.lower() if filename else None,
            bytes=file_bytes
        )
        
        return True, None, file_info
        
    except Exception as e:
        logger.error(f"File validation error: {e}", exc_info=True)
        return False, ERROR_MESSAGES['VALIDATION_ERROR'].format(error=str(e)), None


def _decode_base64(file_base64: str) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """Base64 디코딩 및 MIME 타입 추출"""
    if not file_base64 or not isinstance(file_base64, str):
        return False, ERROR_MESSAGES['INVALID_DATA'], None
    
    mime_type = None
    if file_base64.startswith('data:'):
        parts = file_base64.split(',', 1)
        if len(parts) != 2:
            return False, ERROR_MESSAGES['INVALID_BASE64_FORMAT'], None
        mime_type_part = parts[0]
        file_base64 = parts[1]
        
        if ';base64' in mime_type_part:
            mime_type = mime_type_part.split(';')[0].replace('data:', '')
        else:
            mime_type = mime_type_part.replace('data:', '')
    
    try:
        file_bytes = base64.b64decode(file_base64, validate=True)
        return True, mime_type, file_bytes
    except Exception as e:
        return False, ERROR_MESSAGES['BASE64_DECODE_FAILED'].format(error=str(e)), None


def _validate_total_size(file_bytes: bytes, max_total_size_mb: int) -> Tuple[bool, Optional[str]]:
    """전체 크기 검증"""
    max_total_bytes = max_total_size_mb * 1024 * 1024
    if len(file_bytes) > max_total_bytes:
        return False, ERROR_MESSAGES['FILE_TOO_LARGE'].format(max_size=max_total_size_mb)
    return True, None


def _validate_and_determine_mime(
    file_bytes: bytes,
    declared_mime: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """MIME 타입 검증 및 최종 결정"""
    detected_mime = detect_mime_type(file_bytes)
    
    if declared_mime and declared_mime not in ALLOWED_MIME_TYPES:
        return False, ERROR_MESSAGES['UNSUPPORTED_MIME_TYPE'].format(mime_type=declared_mime)
    
    if detected_mime and detected_mime not in ALLOWED_MIME_TYPES:
        return False, ERROR_MESSAGES['UNSUPPORTED_MIME_TYPE'].format(mime_type=detected_mime)
    
    if declared_mime and detected_mime and declared_mime != detected_mime:
        logger.warning(f"MIME 타입 불일치: 선언={declared_mime}, 실제={detected_mime}")
        declared_mime = detected_mime
    
    final_mime = declared_mime or detected_mime
    if not final_mime:
        return False, ERROR_MESSAGES['MIME_TYPE_UNKNOWN'], None
    
    if final_mime not in ALLOWED_MIME_TYPES:
        return False, ERROR_MESSAGES['MIME_TYPE_NOT_SUPPORTED'].format(mime_type=final_mime)
    
    return True, final_mime


def _validate_file_type_size(file_bytes: bytes, mime_type: str) -> Tuple[bool, Optional[str]]:
    """파일 타입별 크기 제한 확인"""
    file_config = ALLOWED_MIME_TYPES.get(mime_type)
    if not file_config:
        return False, ERROR_MESSAGES['MIME_TYPE_NOT_SUPPORTED'].format(mime_type=mime_type)
    
    max_size_bytes = file_config['max_size_mb'] * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        return False, ERROR_MESSAGES['FILE_TOO_LARGE'].format(max_size=file_config['max_size_mb'])
    
    return True, None


def _validate_extension(filename: str, mime_type: str) -> Tuple[bool, Optional[str]]:
    """확장자 검증"""
    file_config = ALLOWED_MIME_TYPES.get(mime_type)
    if not file_config:
        return False, ERROR_MESSAGES['MIME_TYPE_NOT_SUPPORTED'].format(mime_type=mime_type)
    
    file_ext = Path(filename).suffix.lower()
    if file_ext not in file_config['ext']:
        return False, ERROR_MESSAGES['EXTENSION_MISMATCH']
    
    return True, None


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    파일명 검증
    
    Args:
        filename: 파일명
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if not filename or not filename.strip():
            return False, ERROR_MESSAGES['FILENAME_EMPTY']
        
        if len(filename) > 255:
            return False, ERROR_MESSAGES['FILENAME_TOO_LONG']
        
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                return False, ERROR_MESSAGES['FILENAME_INVALID_CHARS']
        
        file_ext = Path(filename).suffix.lower()
        if file_ext in DANGEROUS_EXTENSIONS:
            return False, ERROR_MESSAGES['FILENAME_DANGEROUS_EXT'].format(ext=file_ext)
        
        normalized = Path(filename).name
        if normalized != filename:
            return False, ERROR_MESSAGES['FILENAME_PATH_TRAVERSAL']
        
        return True, None
        
    except Exception as e:
        logger.error(f"Filename validation error: {e}", exc_info=True)
        return False, ERROR_MESSAGES['FILENAME_VALIDATION_ERROR'].format(error=str(e))


def detect_mime_type(file_bytes: bytes) -> Optional[str]:
    """
    매직 바이트를 사용한 실제 파일 타입 감지
    
    Args:
        file_bytes: 파일 바이트 데이터
        
    Returns:
        MIME 타입 또는 None
    """
    try:
        # python-magic 사용 (권장)
        try:
            import magic
            mime = magic.Magic(mime=True)
            return mime.from_buffer(file_bytes)
        except ImportError:
            # filetype 라이브러리 사용 (대체)
            try:
                import filetype
                kind = filetype.guess(file_bytes)
                if kind:
                    return kind.mime
            except ImportError:
                # 기본 검증 (매직 바이트 직접 확인)
                return detect_mime_by_magic_bytes(file_bytes)
    except Exception as e:
        logger.warning(f"MIME type detection error: {e}")
        return None


def detect_mime_by_magic_bytes(file_bytes: bytes) -> Optional[str]:
    """
    매직 바이트를 직접 확인하여 파일 타입 감지
    
    Args:
        file_bytes: 파일 바이트 데이터
        
    Returns:
        MIME 타입 또는 None
    """
    if len(file_bytes) < 4:
        return None
    
    if file_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    
    if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    
    if file_bytes[:6] in [b'GIF87a', b'GIF89a']:
        return 'image/gif'
    
    if file_bytes[:4] == b'%PDF':
        return 'application/pdf'
    
    if file_bytes[:2] == b'PK':
        if b'word/' in file_bytes[:1024]:
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        return 'application/zip'
    
    if 'text/plain' in ALLOWED_MIME_TYPES:
        try:
            decoded = file_bytes[:1024].decode('utf-8')
            if all(ord(c) >= 32 or c in '\n\r\t' for c in decoded[:100]):
                return 'text/plain'
        except (UnicodeDecodeError, ValueError):
            pass
    
    return None

