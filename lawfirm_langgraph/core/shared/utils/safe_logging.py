#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
안전한 로깅 설정 유틸리티
로깅 스트림 문제를 해결하기 위한 안전한 로깅 설정 함수들
"""

import logging
import sys
import os
from typing import Optional


class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream=None, original_stdout_ref=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
        self._fallback_stream = None
    
    def _get_safe_stream(self):
        """안전한 스트림 반환"""
        streams_to_try = []
        if self.stream and hasattr(self.stream, 'write'):
            streams_to_try.append(self.stream)
        if self._original_stdout is not None and hasattr(self._original_stdout, 'write'):
            streams_to_try.append(self._original_stdout)
        if sys.stdout and hasattr(sys.stdout, 'write'):
            streams_to_try.append(sys.stdout)
        if sys.stderr and hasattr(sys.stderr, 'write'):
            streams_to_try.append(sys.stderr)
        
        for stream in streams_to_try:
            try:
                if hasattr(stream, 'buffer') or hasattr(stream, 'write'):
                    return stream
            except (ValueError, AttributeError, OSError):
                continue
        return None
    
    def _is_stream_valid(self, stream):
        """스트림이 유효한지 확인"""
        if stream is None:
            return False
        try:
            if hasattr(stream, 'buffer'):
                buffer = stream.buffer
                if buffer is None:
                    return False
                if hasattr(buffer, 'raw'):
                    raw = buffer.raw
                    if raw is None:
                        return False
            if not hasattr(stream, 'write'):
                return False
            return True
        except (ValueError, AttributeError, OSError):
            return False
    
    def emit(self, record):
        """안전한 로그 출력 (버퍼 분리 오류 방지)"""
        try:
            msg = self.format(record) + self.terminator
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    if hasattr(safe_stream, 'buffer'):
                        try:
                            buffer = safe_stream.buffer
                            if buffer is None:
                                raise ValueError("Buffer is None")
                        except (ValueError, AttributeError):
                            if hasattr(safe_stream, 'write'):
                                safe_stream.write(msg)
                                return
                            else:
                                raise ValueError("No write method")
                    else:
                        safe_stream.write(msg)
                    
                    try:
                        safe_stream.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError) as e:
                    if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                        pass
                    else:
                        pass
            
            try:
                if sys.stderr and hasattr(sys.stderr, 'write'):
                    try:
                        sys.stderr.write(msg)
                        try:
                            sys.stderr.flush()
                        except (ValueError, AttributeError, OSError):
                            pass
                        return
                    except (ValueError, AttributeError, OSError) as e:
                        if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                            pass
            except (ValueError, AttributeError, OSError):
                pass
        except Exception:
            self.handleError(record)


def disable_external_logging():
    """
    외부 라이브러리의 로깅을 비활성화합니다.
    HuggingFace 관련 모든 로거를 포함하여 완전히 비활성화합니다.
    """
    # 환경 변수로 HuggingFace 로깅 비활성화
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
    
    # FAISS 로깅 비활성화
    faiss_loggers = ['faiss', 'faiss.loader']
    for logger_name in faiss_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # Transformers 로깅 비활성화 (더 완전하게)
    transformers_loggers = [
        'transformers',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
        'transformers.configuration_utils',
        'transformers.modeling_utils',
        'transformers.file_utils',
        'transformers.trainer',
        'transformers.trainer_utils',
        'transformers.modeling_tf_utils',
        'transformers.modeling_flax_utils',
    ]
    for logger_name in transformers_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # Sentence-Transformers 로깅 비활성화
    sentence_transformers_loggers = [
        'sentence_transformers',
        'sentence_transformers.SentenceTransformer',
        'sentence_transformers.models',
        'sentence_transformers.util',
    ]
    for logger_name in sentence_transformers_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # HuggingFace Hub 로깅 비활성화
    hf_hub_loggers = [
        'huggingface_hub',
        'huggingface_hub.file_download',
        'huggingface_hub.utils',
        'huggingface_hub.hf_api',
    ]
    for logger_name in hf_hub_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # PyTorch 로깅 비활성화
    torch_loggers = [
        'torch',
        'torch.distributed',
        'torch.nn',
        'torch.optim',
    ]
    for logger_name in torch_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # 기타 외부 라이브러리 로깅 비활성화
    other_loggers = [
        'urllib3',
        'requests',
        'tokenizers',
        'datasets',
        'accelerate',
        'bitsandbytes',
    ]
    for logger_name in other_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def setup_safe_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    안전한 로깅 설정을 구성합니다.
    HuggingFace 관련 로깅을 자동으로 비활성화합니다.
    
    Args:
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: 로그 포맷 문자열
        log_file: 로그 파일 경로 (선택사항)
    
    Returns:
        설정된 로거
    """
    # HuggingFace 로깅 비활성화 (가장 먼저 실행)
    disable_external_logging()
    # 기본 포맷 설정
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 기존 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새로운 핸들러 설정
    handlers = []
    
    # 콘솔 핸들러 (SafeStreamHandler 사용)
    try:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    except Exception as e:
        print(f"Warning: Could not setup console logging: {e}")
    
    # 파일 핸들러 (선택사항)
    if log_file:
        try:
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # 루트 로거 설정
    root_logger.setLevel(getattr(logging, level.upper()))
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 특정 라이브러리의 로깅 레벨 조정
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def get_safe_logger(name: str) -> logging.Logger:
    """
    안전한 로거를 반환합니다.
    
    Args:
        name: 로거 이름
    
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)


def setup_script_logging(script_name: str) -> logging.Logger:
    """
    스크립트용 로깅을 설정합니다.
    
    Args:
        script_name: 스크립트 이름
    
    Returns:
        설정된 로거
    """
    # 로그 파일 경로 설정
    log_file = f"logs/{script_name}.log"
    
    # 안전한 로깅 설정
    logger = setup_safe_logging(
        level="INFO",
        log_file=log_file
    )
    
    # 외부 라이브러리 로깅 비활성화
    disable_external_logging()
    
    return logger


if __name__ == "__main__":
    # 테스트
    logger = setup_script_logging("test_safe_logging")
    logger.info("안전한 로깅 테스트 성공!")
    logger.warning("경고 메시지 테스트")
    logger.error("오류 메시지 테스트")
