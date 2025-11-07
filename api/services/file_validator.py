"""
파일 검증 서비스 (보안)
"""
import logging
import base64
import io
import re
from typing import Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# 허용된 파일 타입 정의
ALLOWED_MIME_TYPES = {
    # 이미지
    'image/jpeg': {'ext': ['.jpg', '.jpeg'], 'max_size_mb': 10},
    'image/png': {'ext': ['.png'], 'max_size_mb': 10},
    'image/gif': {'ext': ['.gif'], 'max_size_mb': 10},
    'image/webp': {'ext': ['.webp'], 'max_size_mb': 10},
    
    # 텍스트
    'text/plain': {'ext': ['.txt'], 'max_size_mb': 5},
    'text/csv': {'ext': ['.csv'], 'max_size_mb': 5},
    
    # 문서
    'application/pdf': {'ext': ['.pdf'], 'max_size_mb': 20},
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {
        'ext': ['.docx'], 'max_size_mb': 15
    },
    'application/msword': {'ext': ['.doc'], 'max_size_mb': 15},
}

# 위험한 확장자 차단
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
    '.jar', '.app', '.deb', '.rpm', '.sh', '.ps1', '.dll', '.so',
    '.dylib', '.zip', '.rar', '.7z', '.tar', '.gz'
}

# 위험한 파일명 패턴
DANGEROUS_PATTERNS = [
    r'\.\./',  # 경로 탐색
    r'\.\.\\',  # 경로 탐색 (Windows)
    r'^/',  # 절대 경로
    r'^\\',  # 절대 경로 (Windows)
    r'[<>:"|?*]',  # Windows 금지 문자
]


def validate_file_base64(
    file_base64: str,
    filename: Optional[str] = None,
    max_total_size_mb: int = 50
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Base64 인코딩된 파일 검증
    
    Args:
        file_base64: Base64 인코딩된 파일
        filename: 파일명 (선택사항)
        max_total_size_mb: 최대 전체 크기 (MB)
        
    Returns:
        (is_valid, error_message, file_info)
        file_info: {'mime_type': str, 'size': int, 'extension': str, 'bytes': bytes}
    """
    try:
        # 1. Base64 형식 검증
        if not file_base64 or not isinstance(file_base64, str):
            return False, "파일 데이터가 올바르지 않습니다.", None
        
        # data:image/png;base64, 형태 처리
        mime_type = None
        if file_base64.startswith('data:'):
            parts = file_base64.split(',', 1)
            if len(parts) != 2:
                return False, "Base64 인코딩 형식이 올바르지 않습니다.", None
            mime_type_part = parts[0]
            file_base64 = parts[1]
            
            # MIME 타입 추출
            if ';base64' in mime_type_part:
                mime_type = mime_type_part.split(';')[0].replace('data:', '')
            else:
                mime_type = mime_type_part.replace('data:', '')
        
        # 2. Base64 디코딩
        try:
            file_bytes = base64.b64decode(file_base64, validate=True)
        except Exception as e:
            return False, f"Base64 디코딩 실패: {str(e)}", None
        
        # 3. 크기 검증 (Base64 오버헤드 고려)
        max_total_bytes = max_total_size_mb * 1024 * 1024
        if len(file_bytes) > max_total_bytes:
            return False, f"파일 크기가 {max_total_size_mb}MB를 초과합니다.", None
        
        # 4. 파일명 검증
        if filename:
            is_valid_name, name_error = validate_filename(filename)
            if not is_valid_name:
                return False, name_error, None
        
        # 5. 실제 파일 타입 검증 (매직 바이트)
        detected_mime = detect_mime_type(file_bytes)
        
        # 6. MIME 타입 검증
        if mime_type and mime_type not in ALLOWED_MIME_TYPES:
            return False, f"허용되지 않은 파일 타입입니다: {mime_type}", None
        
        # 7. 감지된 MIME 타입 검증
        if detected_mime and detected_mime not in ALLOWED_MIME_TYPES:
            return False, f"허용되지 않은 파일 타입입니다: {detected_mime}", None
        
        # 8. MIME 타입 일치 검증
        if mime_type and detected_mime and mime_type != detected_mime:
            logger.warning(f"MIME 타입 불일치: 선언={mime_type}, 실제={detected_mime}")
            # 보안상 실제 타입 우선
            mime_type = detected_mime
        
        # 9. 최종 MIME 타입 결정
        final_mime = mime_type or detected_mime
        if not final_mime:
            return False, "파일 타입을 확인할 수 없습니다.", None
        
        # 10. 크기 제한 확인
        file_config = ALLOWED_MIME_TYPES.get(final_mime)
        if not file_config:
            return False, f"지원하지 않는 파일 타입입니다: {final_mime}", None
        
        max_size_bytes = file_config['max_size_mb'] * 1024 * 1024
        if len(file_bytes) > max_size_bytes:
            return False, f"파일 크기가 {file_config['max_size_mb']}MB를 초과합니다.", None
        
        # 11. 확장자 검증
        if filename:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in file_config['ext']:
                return False, f"파일 확장자가 MIME 타입과 일치하지 않습니다.", None
        
        # 12. 파일 정보 반환
        file_info = {
            'mime_type': final_mime,
            'size': len(file_bytes),
            'extension': Path(filename).suffix.lower() if filename else None,
            'bytes': file_bytes
        }
        
        return True, None, file_info
        
    except Exception as e:
        logger.error(f"File validation error: {e}", exc_info=True)
        return False, f"파일 검증 중 오류가 발생했습니다: {str(e)}", None


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    파일명 검증
    
    Args:
        filename: 파일명
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # 1. 빈 파일명 검증
        if not filename or not filename.strip():
            return False, "파일명이 비어있습니다."
        
        # 2. 파일명 길이 검증
        if len(filename) > 255:
            return False, "파일명이 너무 깁니다 (최대 255자)."
        
        # 3. 위험한 패턴 검증
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                return False, "파일명에 허용되지 않은 문자가 포함되어 있습니다."
        
        # 4. 위험한 확장자 검증
        file_ext = Path(filename).suffix.lower()
        if file_ext in DANGEROUS_EXTENSIONS:
            return False, f"허용되지 않은 파일 확장자입니다: {file_ext}"
        
        # 5. 파일명 정규화 (경로 탐색 제거)
        normalized = Path(filename).name
        if normalized != filename:
            return False, "파일명에 경로 정보가 포함되어 있습니다."
        
        return True, None
        
    except Exception as e:
        logger.error(f"Filename validation error: {e}", exc_info=True)
        return False, f"파일명 검증 중 오류가 발생했습니다: {str(e)}"


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
    
    # JPEG
    if file_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    
    # PNG
    if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    
    # GIF
    if file_bytes[:6] in [b'GIF87a', b'GIF89a']:
        return 'image/gif'
    
    # PDF
    if file_bytes[:4] == b'%PDF':
        return 'application/pdf'
    
    # ZIP (DOCX는 ZIP 기반)
    if file_bytes[:2] == b'PK':
        # DOCX 확인
        if b'word/' in file_bytes[:1024]:
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        return 'application/zip'
    
    # 텍스트 파일 (UTF-8, ASCII)
    try:
        file_bytes[:1024].decode('utf-8')
        return 'text/plain'
    except:
        pass
    
    return None

