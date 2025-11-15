"""
통합 파일 처리 서비스
이미지, 텍스트, PDF, DOCX 등 다양한 파일 형식 처리
"""
import logging
import base64
import io
from typing import Optional, Tuple
from pathlib import Path

from api.services.file_validator import validate_file_base64, validate_filename
from api.services.ocr_service import extract_text_from_image, is_ocr_available

logger = logging.getLogger(__name__)


def process_file(
    file_base64: str,
    filename: Optional[str] = None,
    max_text_length: int = 100000
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    파일 처리 (텍스트 추출)
    
    Args:
        file_base64: Base64 인코딩된 파일
        filename: 파일명 (선택사항)
        max_text_length: 최대 텍스트 길이
        
    Returns:
        (success, extracted_text, error_message)
    """
    try:
        # 1. 파일 검증
        is_valid, error_msg, file_info = validate_file_base64(file_base64, filename)
        if not is_valid:
            return False, None, error_msg
        
        mime_type = file_info['mime_type']
        file_bytes = file_info['bytes']
        
        # 2. 파일 타입별 처리
        if mime_type.startswith('image/'):
            # 이미지: OCR 처리
            return process_image(file_bytes, max_text_length)
        
        elif mime_type == 'text/plain':
            # 텍스트 파일: 직접 읽기
            return process_text_file(file_bytes, max_text_length)
        
        elif mime_type == 'application/pdf':
            # PDF: pdfplumber 사용
            return process_pdf(file_bytes, max_text_length)
        
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # DOCX: python-docx 사용
            return process_docx(file_bytes, max_text_length)
        
        elif mime_type == 'application/msword':
            # DOC: 변환 필요 (구형 형식)
            return False, None, "DOC 형식은 지원하지 않습니다. DOCX 형식으로 변환해주세요."
        
        else:
            return False, None, f"지원하지 않는 파일 타입입니다: {mime_type}"
            
    except Exception as e:
        logger.error(f"File processing error: {e}", exc_info=True)
        return False, None, f"파일 처리 중 오류가 발생했습니다: {str(e)}"


def process_image(file_bytes: bytes, max_text_length: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """이미지 처리 (OCR)"""
    try:
        if not is_ocr_available():
            return False, None, "OCR 서비스를 사용할 수 없습니다."
        
        text = extract_text_from_image(file_bytes)
        
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"Extracted text truncated to {max_text_length} characters")
        
        return True, text, None
    except Exception as e:
        logger.error(f"Image processing error: {e}", exc_info=True)
        return False, None, f"이미지 처리 중 오류가 발생했습니다: {str(e)}"


def process_text_file(file_bytes: bytes, max_text_length: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """텍스트 파일 처리"""
    try:
        # 인코딩 자동 감지
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']
        
        for encoding in encodings:
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return False, None, "텍스트 파일 인코딩을 확인할 수 없습니다."
        
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"Text truncated to {max_text_length} characters")
        
        return True, text, None
    except Exception as e:
        logger.error(f"Text file processing error: {e}", exc_info=True)
        return False, None, f"텍스트 파일 처리 중 오류가 발생했습니다: {str(e)}"


def process_pdf(file_bytes: bytes, max_text_length: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """PDF 처리"""
    try:
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        text = '\n\n'.join(text_parts)
        
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"PDF text truncated to {max_text_length} characters")
        
        return True, text, None
    except ImportError:
        return False, None, "PDF 처리를 위한 라이브러리가 설치되지 않았습니다. (pdfplumber)"
    except Exception as e:
        logger.error(f"PDF processing error: {e}", exc_info=True)
        return False, None, f"PDF 처리 중 오류가 발생했습니다: {str(e)}"


def process_docx(file_bytes: bytes, max_text_length: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """DOCX 처리"""
    try:
        from docx import Document
        
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        text = '\n\n'.join(text_parts)
        
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"DOCX text truncated to {max_text_length} characters")
        
        return True, text, None
    except ImportError:
        return False, None, "DOCX 처리를 위한 라이브러리가 설치되지 않았습니다. (python-docx)"
    except Exception as e:
        logger.error(f"DOCX processing error: {e}", exc_info=True)
        return False, None, f"DOCX 처리 중 오류가 발생했습니다: {str(e)}"

