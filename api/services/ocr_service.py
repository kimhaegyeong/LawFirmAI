"""
OCR 서비스
이미지에서 텍스트를 추출하는 서비스
"""
import logging
import base64
import io
from typing import Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# EasyOCR은 지연 로딩 (초기 로딩 시간이 오래 걸릴 수 있음)
_ocr_reader = None


def _get_ocr_reader():
    """OCR 리더 인스턴스 가져오기 (싱글톤 패턴)"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader...")
            # 한국어와 영어 지원
            _ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
            logger.info("✅ EasyOCR reader initialized successfully")
        except ImportError:
            logger.error("EasyOCR is not installed. Please install it with: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}", exc_info=True)
            raise
    return _ocr_reader


def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
    """
    이미지 유효성 검증
    
    Args:
        image_bytes: 이미지 바이트 데이터
        max_size_mb: 최대 크기 (MB)
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # 크기 검증
        max_size_bytes = max_size_mb * 1024 * 1024
        if len(image_bytes) > max_size_bytes:
            return False, f"이미지 크기가 {max_size_mb}MB를 초과합니다."
        
        # 이미지 형식 검증
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
        except Exception as e:
            return False, f"유효하지 않은 이미지 형식입니다: {str(e)}"
        
        return True, None
    except Exception as e:
        logger.error(f"Image validation error: {e}", exc_info=True)
        return False, f"이미지 검증 중 오류가 발생했습니다: {str(e)}"


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    이미지 전처리 (회전, 노이즈 제거, 해상도 조정)
    
    Args:
        image_bytes: 원본 이미지 바이트 데이터
        
    Returns:
        전처리된 이미지 바이트 데이터
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGB로 변환 (RGBA인 경우)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 해상도 조정 (너무 큰 이미지는 리사이즈)
        max_dimension = 2000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Image resized to {new_size}")
        
        # 바이트로 변환
        output = io.BytesIO()
        image.save(output, format='PNG', quality=95)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}", exc_info=True)
        # 전처리 실패 시 원본 반환
        return image_bytes


def extract_text_from_image(image_bytes: bytes, preprocess: bool = True) -> str:
    """
    이미지에서 텍스트 추출
    
    Args:
        image_bytes: 이미지 바이트 데이터
        preprocess: 전처리 여부
        
    Returns:
        추출된 텍스트
    """
    try:
        # 이미지 검증
        is_valid, error_message = validate_image(image_bytes)
        if not is_valid:
            raise ValueError(error_message)
        
        # 이미지 전처리
        if preprocess:
            processed_bytes = preprocess_image(image_bytes)
        else:
            processed_bytes = image_bytes
        
        # OCR 리더 가져오기
        reader = _get_ocr_reader()
        
        # 이미지를 numpy 배열로 변환
        image = Image.open(io.BytesIO(processed_bytes))
        image_np = np.array(image)
        
        # OCR 수행
        logger.info("Performing OCR on image...")
        results = reader.readtext(image_np)
        
        # 텍스트 추출 및 결합
        extracted_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # 신뢰도 30% 이상만 사용
                extracted_texts.append(text)
                logger.debug(f"Extracted text: {text} (confidence: {confidence:.2f})")
        
        # 텍스트 결합
        full_text = '\n'.join(extracted_texts)
        
        if not full_text.strip():
            logger.warning("No text extracted from image")
            return ""
        
        logger.info(f"OCR completed. Extracted {len(extracted_texts)} text blocks, total length: {len(full_text)}")
        return full_text.strip()
        
    except ValueError as e:
        logger.error(f"Image validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"OCR extraction error: {e}", exc_info=True)
        raise RuntimeError(f"이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {str(e)}")


def extract_text_from_base64(image_base64: str, preprocess: bool = True) -> str:
    """
    Base64 인코딩된 이미지에서 텍스트 추출
    
    Args:
        image_base64: Base64 인코딩된 이미지 문자열
        preprocess: 전처리 여부
        
    Returns:
        추출된 텍스트
    """
    try:
        # Base64 디코딩
        if image_base64.startswith('data:image'):
            # data:image/png;base64, 형태인 경우
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        return extract_text_from_image(image_bytes, preprocess)
    except Exception as e:
        logger.error(f"Base64 decoding error: {e}", exc_info=True)
        raise ValueError(f"Base64 이미지 디코딩 중 오류가 발생했습니다: {str(e)}")


def is_ocr_available() -> bool:
    """OCR 서비스 사용 가능 여부"""
    try:
        _get_ocr_reader()
        return True
    except Exception:
        return False

