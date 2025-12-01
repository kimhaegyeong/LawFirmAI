# -*- coding: utf-8 -*-
"""
LangGraph 질의 테스트 스크립트

Usage:
    python lawfirm_langgraph/tests/runners/run_query_test.py "질의 내용"
    python lawfirm_langgraph/tests/runners/run_query_test.py  # 기본 질의 사용
"""

import sys
import os
import asyncio
import logging
import signal
import atexit
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# TRACE 레벨 추가 (DEBUG보다 낮은 레벨, 값: 5)
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, "TRACE")
    
    # Logger 클래스에 trace 메서드 추가
    def trace(self, message, *args, **kwargs):
        """TRACE 레벨 로그"""
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, message, args, **kwargs)
    
    logging.Logger.trace = trace

# UTF-8 인코딩 설정 (Windows 호환)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
runners_dir = script_dir.parent
tests_dir = runners_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

# LineBufferedFileHandler 클래스 정의 (줄 단위로 즉시 파일에 쓰기)
class LineBufferedFileHandler(logging.FileHandler):
    """줄 단위로 즉시 파일에 쓰는 파일 핸들러
    
    buffering=1 (line buffering)을 사용하여 각 줄이 즉시 파일에 쓰여집니다.
    이는 QueueHandler + QueueListener보다 더 안정적입니다 (백그라운드 스레드 없음).
    
    장점:
    - 예외 발생 시에도 로그가 즉시 파일에 저장됨
    - 백그라운드 스레드 없이 직접 파일에 쓰므로 더 안정적
    - flush 호출 불필요 (line buffering이 자동 처리)
    """
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False, errors=None):
        """줄 단위 버퍼링 파일 핸들러 초기화
        
        Args:
            filename: 로그 파일 경로
            mode: 파일 모드 ('a' 또는 'w')
            encoding: 파일 인코딩
            delay: 파일 열기 지연 여부
            errors: 인코딩 오류 처리 방식
        """
        # 부모 클래스 초기화 (filters 등 속성 초기화를 위해 필요)
        # delay=True로 설정하여 부모가 파일을 열지 않도록 함
        super().__init__(filename, mode=mode, encoding=encoding, delay=True, errors=errors)
        
        # delay가 False이면 즉시 파일 열기 (line buffering)
        if not delay:
            # 기존 stream이 있으면 닫기
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.close()
                except Exception:
                    pass
            
            # line buffering (buffering=1)로 파일 열기
            # buffering=1: 줄 단위로 즉시 쓰기 (자동 flush)
            self.stream = open(
                self.baseFilename, 
                mode, 
                buffering=1,  # line buffering
                encoding=encoding, 
                errors=errors
            )
    
    def emit(self, record):
        """로그를 줄 단위로 즉시 파일에 쓰기"""
        try:
            # stream이 없으면 line buffering으로 열기
            if self.stream is None:
                self.stream = open(
                    self.baseFilename, 
                    self.mode, 
                    buffering=1,  # line buffering
                    encoding=self.encoding, 
                    errors=self.errors
                )
            
            # 로그 포맷팅
            msg = self.format(record)
            
            # 줄 단위로 즉시 쓰기 (line buffering이므로 \n 만나면 자동 flush)
            self.stream.write(msg + self.terminator)
            
            # Windows에서 추가 보장을 위해 명시적 flush
            if sys.platform == 'win32':
                try:
                    self.stream.flush()
                    # Windows에서 파일 동기화
                    if hasattr(self.stream, 'fileno'):
                        try:
                            os.fsync(self.stream.fileno())
                        except (OSError, AttributeError):
                            pass
                except Exception:
                    pass
                    
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """강화된 flush 메서드"""
        try:
            # 부모 클래스의 flush 호출
            super().flush()
            
            # Windows에서 파일 동기화 (추가 보장)
            if self.stream and sys.platform == 'win32' and hasattr(self.stream, 'fileno'):
                try:
                    os.fsync(self.stream.fileno())
                except (OSError, AttributeError):
                    pass
        except Exception:
            pass


# SafeStreamHandler 클래스 정의 (Windows 환경 호환)
class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream, original_stdout_ref=None):
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
    
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
                        safe_stream.write(msg)
                    
                    try:
                        safe_stream.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError):
                    pass
            
            if sys.stderr and hasattr(sys.stderr, 'write'):
                try:
                    sys.stderr.write(msg)
                    try:
                        sys.stderr.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError):
                    pass
        except Exception:
            pass
    
    def flush(self):
        """안전한 flush (오류 무시)"""
        try:
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    safe_stream.flush()
                except (ValueError, AttributeError, OSError):
                    pass
        except (ValueError, AttributeError, OSError):
            pass


# 원본 stdout/stderr 저장
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# 🔥 개선: 글로벌 로그 파일 경로 저장 (signal handler에서 사용)
_global_log_file_path = None
# 🔥 개선: 글로벌 로그 파일 핸들 저장 (stdout/stderr 리다이렉트용)
_log_file_handle = None


# Tee 클래스 정의 (콘솔과 파일 모두에 쓰기)
class Tee:
    """콘솔과 파일 모두에 쓰는 클래스
    
    stdout/stderr를 리다이렉트하여 콘솔과 로그 파일 모두에 출력합니다.
    """
    
    def __init__(self, *files):
        """여러 파일 객체를 받아서 모두에 쓰기
        
        Args:
            *files: 파일 객체들 (예: sys.stdout, log_file_handle)
        """
        self.files = files
    
    def write(self, obj):
        """모든 파일에 쓰기"""
        for f in self.files:
            try:
                f.write(obj)
                # line buffering이므로 \n 만나면 자동 flush되지만, 명시적 flush도 수행
                if hasattr(f, 'flush'):
                    f.flush()
            except Exception:
                pass
    
    def flush(self):
        """모든 파일 flush"""
        for f in self.files:
            try:
                if hasattr(f, 'flush'):
                    f.flush()
            except Exception:
                pass
    
    def isatty(self):
        """터미널인지 확인 (일부 라이브러리에서 필요)
        
        Returns:
            False: 파일이므로 터미널이 아님
        """
        return False
    
    def fileno(self):
        """파일 디스크립터 반환 (일부 라이브러리에서 필요)
        
        Returns:
            첫 번째 파일의 fileno (있는 경우)
        """
        if self.files:
            try:
                return self.files[0].fileno()
            except (AttributeError, OSError):
                pass
        raise OSError("fileno not available")


def _signal_handler(signum, frame):
    """시그널 핸들러 (프로세스 종료 시 로그 flush)"""
    try:
        # 로그 파일 핸들러 flush
        global _log_file_handle
        if '_log_file_handle' in globals() and _log_file_handle:
            try:
                _log_file_handle.flush()
                if sys.platform == 'win32' and hasattr(_log_file_handle, 'fileno'):
                    try:
                        os.fsync(_log_file_handle.fileno())
                    except (OSError, AttributeError):
                        pass
            except Exception:
                pass
        
        # 모든 파일 핸들러 flush
        flush_all_log_handlers()
        if _global_log_file_path:
            print(f"\n[시그널 수신] 로그 파일: {_global_log_file_path}")
    except Exception:
        pass
    # 원래 시그널 동작 수행
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    sys.exit(0)


def _atexit_handler():
    """프로세스 종료 시 로그 flush (atexit 사용)"""
    try:
        # 로그 파일 핸들러 flush
        global _log_file_handle
        if '_log_file_handle' in globals() and _log_file_handle:
            try:
                _log_file_handle.flush()
                if sys.platform == 'win32' and hasattr(_log_file_handle, 'fileno'):
                    try:
                        os.fsync(_log_file_handle.fileno())
                    except (OSError, AttributeError):
                        pass
            except Exception:
                pass
        
        # 모든 파일 핸들러 flush
        flush_all_log_handlers()
    except Exception:
        pass


# 🔥 개선: 시그널 핸들러 등록 (프로세스 종료 시 로그 저장 보장)
if sys.platform != 'win32':
    # Unix/Linux: SIGTERM, SIGINT 처리
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
else:
    # Windows: SIGINT만 처리 (SIGTERM 없음)
    signal.signal(signal.SIGINT, _signal_handler)

# 🔥 개선: atexit 핸들러 등록 (정상 종료 시 로그 저장 보장)
atexit.register(_atexit_handler)


def flush_all_log_handlers():
    """모든 로거의 모든 핸들러 flush (전역 함수)
    
    LineBufferedFileHandler는 line buffering을 사용하지만,
    명시적 flush를 통해 최종 로그 저장을 보장합니다.
    """
    try:
        # 모든 로거의 모든 핸들러 flush
        loggers_to_flush = [
            logging.getLogger(),  # 루트 로거
        ]
        
        # 모든 등록된 로거 추가
        try:
            for logger_name in list(logging.Logger.manager.loggerDict.keys()):
                try:
                    sub_logger = logging.getLogger(logger_name)
                    if sub_logger not in loggers_to_flush:
                        loggers_to_flush.append(sub_logger)
                except (ValueError, AttributeError, RuntimeError):
                    pass
        except (AttributeError, RuntimeError):
            pass
        
        # 각 로거의 모든 핸들러 flush
        for logger_to_flush in loggers_to_flush:
            try:
                for handler in logger_to_flush.handlers:
                    try:
                        # 모든 핸들러 타입에 대해 flush
                        handler.flush()
                        
                        # FileHandler의 경우 추가 처리
                        if isinstance(handler, logging.FileHandler):
                            if hasattr(handler, 'stream') and handler.stream:
                                try:
                                    handler.stream.flush()
                                    # Windows에서 강제 동기화
                                    if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                                        try:
                                            os.fsync(handler.stream.fileno())
                                        except (OSError, AttributeError):
                                            pass
                                except (ValueError, AttributeError, OSError):
                                    pass
                        
                        # StreamHandler도 flush (콘솔 출력 보장)
                        if isinstance(handler, logging.StreamHandler):
                            try:
                                if hasattr(handler, 'stream') and handler.stream:
                                    handler.stream.flush()
                            except (ValueError, AttributeError, OSError):
                                pass
                    except Exception:
                        pass
            except (AttributeError, RuntimeError):
                pass
        
        # Python의 표준 출력 스트림도 flush
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (ValueError, AttributeError, OSError):
            pass
    except Exception:
        pass


# 로깅 설정
def setup_logging(log_level: Optional[str] = None, log_file_path: Optional[str] = None) -> logging.Logger:
    """로깅 설정
    
    Args:
        log_level: 로그 레벨 (기본값: 환경 변수 LOG_LEVEL 또는 INFO)
        log_file_path: 로그 파일 경로 (지정하지 않으면 자동 생성)
    
    Returns:
        설정된 로거
    """
    global _global_log_file_path, _log_file_handle, _original_stdout, _original_stderr
    
    # 로그 레벨 결정
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        log_level = log_level.upper()
    
    log_level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    log_level_value = log_level_map.get(log_level, logging.INFO)
    
    # 로그 파일 경로 결정
    if log_file_path:
        log_file = Path(log_file_path)
    else:
        # 로그 디렉토리 생성 (환경 변수로 경로 지정 가능)
        log_dir_env = os.getenv("TEST_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env)
        else:
            log_dir = project_root / "logs" / "langgraph"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 경로 (환경 변수로 파일명 지정 가능)
        log_file_env = os.getenv("TEST_LOG_FILE")
        if log_file_env:
            log_file = Path(log_file_env)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"test_langgraph_query_{timestamp}.log"
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # 기존 핸들러 제거
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # 🔥 개선: 줄 단위 버퍼링 파일 핸들러 사용 (line buffering)
    # 장점: 예외 발생 시에도 로그가 즉시 파일에 저장됨, 백그라운드 스레드 없이 더 안정적
    # mode='a'로 설정하여 main()에서 미리 생성한 로그를 보존
    try:
        file_handler = LineBufferedFileHandler(
            log_file, 
            encoding='utf-8', 
            mode='a',  # append 모드로 변경하여 기존 로그 보존
            delay=False
        )
        file_handler.setLevel(log_level_value)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # 🔥 개선: 파일 핸들러가 제대로 작동하는지 즉시 테스트
        test_record = logging.LogRecord(
            name="lawfirm_langgraph.tests.runners.run_query_test",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg="[로그 핸들러 테스트] 파일 핸들러가 정상적으로 작동합니다.",
            args=(),
            exc_info=None
        )
        file_handler.emit(test_record)
        file_handler.flush()
    except Exception as handler_error:
        # 파일 핸들러 생성 실패 시 콘솔에 출력
        print(f"\n[경고] 파일 핸들러 생성 실패: {handler_error}")
        import traceback
        traceback.print_exc()
        raise
    
    # 콘솔 핸들러 추가 (SafeStreamHandler 사용)
    try:
        base_handler = logging.StreamHandler(_original_stdout)
    except (ValueError, AttributeError):
        try:
            base_handler = logging.StreamHandler(sys.stdout)
        except (ValueError, AttributeError):
            base_handler = logging.StreamHandler(sys.stderr)
    
    safe_handler = SafeStreamHandler(base_handler.stream, _original_stdout)
    safe_handler.setLevel(log_level_value)
    safe_handler.setFormatter(file_formatter)
    root_logger.addHandler(safe_handler)
    
    # 🔥 개선: 모든 주요 로거가 루트 로거로 전파되도록 강제 설정
    # 모든 기존 로거의 propagate를 True로 설정하고, 별도 핸들러 제거
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        try:
            existing_logger = logging.getLogger(logger_name)
            existing_logger.propagate = True
            existing_logger.disabled = False
            
            # 🔥 개선: 기존 로거의 파일 핸들러 제거 (우리 파일 핸들러만 사용)
            for handler in list(existing_logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    # 우리 파일 핸들러가 아니면 제거
                    if handler != file_handler:
                        existing_logger.removeHandler(handler)
        except (ValueError, AttributeError, RuntimeError):
            pass
    
    # lawfirm_langgraph 로거 설정
    langgraph_logger = logging.getLogger("lawfirm_langgraph")
    langgraph_logger.setLevel(log_level_value)
    langgraph_logger.propagate = True
    langgraph_logger.disabled = False
    
    # 🔥 개선: core 네임스페이스 로거들도 루트 로거로 전파되도록 설정
    core_logger = logging.getLogger("core")
    core_logger.setLevel(log_level_value)
    core_logger.propagate = True
    core_logger.disabled = False
    
    # 🔥 개선: 주요 서브 로거들 설정 (propagate=True로 루트 로거의 핸들러 사용)
    # QueueHandler + QueueListener 패턴에서는 모든 로거가 같은 큐를 사용하므로
    # 서브 로거에 직접 핸들러를 추가할 필요가 없습니다.
    important_loggers = [
        "core.search.engines.semantic_search_engine_v2",
        "core.data.db_adapter",
        "core.workflow.workflow_service",
        "core.workflow.legal_workflow_enhanced",
    ]
    
    for logger_name in important_loggers:
        try:
            sub_logger = logging.getLogger(logger_name)
            sub_logger.setLevel(log_level_value)
            sub_logger.propagate = True  # 루트 로거로 전파
            sub_logger.disabled = False
        except (ValueError, AttributeError, RuntimeError):
            pass
    
    # Few-shot examples 경고 필터링 (선택적)
    if os.getenv("SUPPRESS_FEW_SHOT_WARNING", "false").lower() == "true":
        few_shot_logger = logging.getLogger("lawfirm_langgraph.core.generation.formatters.answer_structure_enhancer")
        few_shot_logger.setLevel(logging.ERROR)  # WARNING 이상만 표시
    
    # asyncio 로거는 TRACE 레벨로 설정 (proactor 로그 과다 방지)
    try:
        asyncio_logger = logging.getLogger("asyncio")
        asyncio_logger.setLevel(logging.TRACE)
        asyncio_logger.propagate = False
    except (ValueError, AttributeError, RuntimeError):
        pass
    
    # 🔥 개선: langsmith.client 로거도 우리 파일 핸들러 사용하도록 설정
    try:
        langsmith_logger = logging.getLogger("langsmith.client")
        langsmith_logger.setLevel(log_level_value)
        langsmith_logger.propagate = True  # 루트 로거로 전파
        langsmith_logger.disabled = False
        
        # 기존 파일 핸들러 제거 (우리 파일 핸들러만 사용)
        for handler in list(langsmith_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                langsmith_logger.removeHandler(handler)
    except (ValueError, AttributeError, RuntimeError):
        pass
    
    # 테스트 로거 (파일명과 일치)
    logger = logging.getLogger("lawfirm_langgraph.tests.runners.run_query_test")
    logger.setLevel(log_level_value)
    logger.propagate = True
    logger.disabled = False
    
    # 🔥 개선: 로그 파일 경로를 명시적으로 출력 (파일 생성 확인용 - 한 번만)
    # 먼저 루트 로거로 로그 기록 (로거 설정이 제대로 되었는지 확인)
    root_logger = logging.getLogger()
    root_logger.info(f"로그 파일: {log_file.absolute()} | 로그 레벨: {log_level}")
    
    # 테스트 로거로도 로그 기록
    logger.info(f"로그 파일: {log_file.absolute()} | 로그 레벨: {log_level}")
    
    # 🔥 개선: 로그가 제대로 기록되었는지 확인하기 위해 즉시 flush
    flush_all_log_handlers()
    
    # 🔥 개선: 글로벌 로그 파일 경로 저장 (signal handler에서 사용)
    _global_log_file_path = str(log_file.absolute())
    
    # 🔥 개선: stdout/stderr를 Tee로 감싸서 콘솔과 파일 모두에 쓰기
    # 파일 핸들러와 동일한 파일을 사용하되, write 모드로 열어서 충돌 방지
    try:
        # 로그 파일을 텍스트 모드로 열기 (write, line buffering)
        # 파일 핸들러가 이미 파일을 열었으므로, 같은 파일을 다시 열면 충돌 발생 가능
        # 따라서 파일 핸들러의 stream을 직접 사용하거나, 별도 파일 핸들을 사용
        # 여기서는 파일 핸들러의 stream을 직접 사용하는 대신, 별도 핸들을 사용하되
        # 같은 파일을 append 모드로 열어서 추가 기록
        _log_file_handle = open(log_file, 'a', encoding='utf-8', buffering=1)
        
        # stdout과 stderr를 Tee로 감싸기
        sys.stdout = Tee(_original_stdout, _log_file_handle)
        sys.stderr = Tee(_original_stderr, _log_file_handle)
        
        # 🔥 개선: Tee 설정 후 즉시 테스트 출력
        print(f"[Tee 테스트] stdout/stderr 리다이렉트가 정상적으로 작동합니다.")
        _log_file_handle.flush()
    except Exception as tee_error:
        # Tee 설정 실패 시 경고만 출력하고 계속 진행
        print(f"\n[경고] stdout/stderr 리다이렉트 실패: {tee_error}")
        import traceback
        traceback.print_exc()
        _log_file_handle = None
    
    # 🔥 개선: 콘솔에도 로그 파일 경로 출력 (로그 파일이 생성되지 않을 경우 대비)
    print(f"\n[로그 설정]")
    print(f"  로그 파일: {log_file.absolute()}")
    print(f"  로그 레벨: {log_level}")
    print()
    
    return logger


def get_query_from_args() -> str:
    """명령줄 인자에서 질의 추출"""
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    
    # 환경 변수 확인
    test_query = os.getenv('TEST_QUERY', '').strip()
    if test_query:
        return test_query
    
    # 명령줄 인자 확인
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        # 숫자로 기본 질의 선택
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                return default_queries[idx]
        
        # 질의 내용 직접 입력
        return " ".join(sys.argv[1:])
    
    # 기본 질의 반환
    return default_queries[0]


async def test_langgraph_query(query: str, logger: logging.Logger):
    """LangGraph 질의 테스트 실행
    
    Args:
        query: 테스트할 질의
        logger: 로거
    """
    logger.info("=" * 80)
    logger.info("LangGraph 질의 테스트")
    logger.info("=" * 80)
    logger.info(f"질의: {query}")
    
    try:
        # 초기화 시간 측정 시작
        import time
        total_start_time = time.time()
        
        # 설정 로드
        logger.info("1. 설정 로드 중...")
        setup_start = time.time()
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.config.app_config import Config as AppConfig
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        setup_time = time.time() - setup_start
        logger.info(f"   LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   체크포인트: {config.enable_checkpoint}")
        logger.info(f"   설정 로드 시간: {setup_time:.3f}초")
        
        # 데이터베이스 및 벡터 검색 설정 확인
        logger.info("\n1.1. 데이터베이스 및 벡터 검색 설정 확인...")
        db_check_start = time.time()
        app_config = AppConfig()
        
        # SQLite URL 검증 (테스트 시작 전)
        if app_config.database_url.startswith("sqlite://"):
            logger.error("   ❌ SQLite는 더 이상 지원하지 않습니다. PostgreSQL을 사용하세요.")
            logger.error("   PostgreSQL URL 설정 방법:")
            logger.error("   - DATABASE_URL=postgresql://user:password@host:port/database")
            logger.error("   - 또는 POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD 환경 변수 설정")
            raise ValueError("SQLite is no longer supported. Please configure PostgreSQL.")
        
        logger.info(f"   ✅ Database URL 설정됨 (PostgreSQL)")
        logger.info(f"   VECTOR_SEARCH_METHOD: {app_config.vector_search_method}")
        if app_config.faiss_index_path:
            logger.info(f"   FAISS_INDEX_PATH: {app_config.faiss_index_path}")
        
        # DatabaseAdapter 확인
        db_adapter_start = time.time()
        try:
            from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
            if app_config.database_url:
                db_adapter = DatabaseAdapter(app_config.database_url)
                logger.info(f"   ✅ DatabaseAdapter 초기화 성공: type={db_adapter.db_type}")
                if db_adapter.db_type == 'postgresql':
                    logger.info(f"   ✅ PostgreSQL 사용 중")
                else:
                    logger.error(f"   ❌ 지원하지 않는 데이터베이스 타입: {db_adapter.db_type} (PostgreSQL만 지원)")
                    logger.error("   PostgreSQL URL 설정 방법:")
                    logger.error("   - DATABASE_URL=postgresql://user:password@host:port/database")
                    logger.error("   - 또는 POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD 환경 변수 설정")
                    raise ValueError(f"Unsupported database type: {db_adapter.db_type}. Only PostgreSQL is supported.")
        except ValueError as e:
            logger.error(f"   ❌ DatabaseAdapter 초기화 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"   ❌ DatabaseAdapter 초기화 실패: {e}")
            logger.error("   PostgreSQL 연결을 확인하세요.")
            raise
        
        # VectorSearchFactory 확인
        try:
            from lawfirm_langgraph.core.search.engines.vector_search_adapter import VectorSearchFactory
            logger.info(f"   ✅ VectorSearchFactory 사용 가능")
            if app_config.vector_search_method.lower() == 'pgvector':
                try:
                    import pgvector
                    logger.info(f"   ✅ pgvector 사용 중 (pgvector 패키지 설치됨)")
                except ImportError:
                    logger.warning(f"   ⚠️  pgvector 설정되었으나 패키지가 설치되지 않음")
                    logger.warning(f"   설치 방법: pip install pgvector")
            elif app_config.vector_search_method.lower() == 'faiss':
                logger.info(f"   ✅ FAISS 사용 중")
            elif app_config.vector_search_method.lower() == 'hybrid':
                logger.info(f"   ✅ Hybrid (pgvector + FAISS) 사용 중")
        except Exception as e:
            logger.warning(f"   ⚠️  VectorSearchFactory 사용 불가: {e}")
        
        db_check_time = time.time() - db_check_start
        logger.info(f"   데이터베이스 확인 시간: {db_check_time:.3f}초")
        
        # 서비스 초기화
        logger.info("\n2. LangGraphWorkflowService 초기화 중...")
        service_start = time.time()
        
        # 🔥 개선: 초기화 전 로그 flush
        try:
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
                    if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                        try:
                            os.fsync(handler.stream.fileno())
                        except (OSError, AttributeError):
                            pass
        except Exception:
            pass
        
        try:
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            # 단계별 시간 측정
            component_times = {}
            
            # 2.1. 서비스 인스턴스 생성
            service_create_start = time.time()
            service = LangGraphWorkflowService(config)
            component_times['service_create'] = time.time() - service_create_start
            
            service_time = time.time() - service_start
            logger.info(f"   서비스 초기화 완료 (총 시간: {service_time:.3f}초)")
            
            # 단계별 시간 상세 로깅
            if component_times:
                logger.info(f"   - 서비스 생성: {component_times.get('service_create', 0):.3f}초")
            
            # 🔥 개선: 초기화 직후 즉시 flush
            try:
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                        if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                            try:
                                os.fsync(handler.stream.fileno())
                            except (OSError, AttributeError):
                                pass
            except Exception:
                pass
            
            # 서비스 내부 컴포넌트 확인
            if hasattr(service, 'db_manager') and service.db_manager:
                if hasattr(service.db_manager, '_db_adapter') and service.db_manager._db_adapter:
                    logger.info(f"   ✅ LegalDataConnectorV2 DatabaseAdapter: type={service.db_manager._db_adapter.db_type}")
            
            if hasattr(service, 'semantic_search_engine') and service.semantic_search_engine:
                if hasattr(service.semantic_search_engine, '_db_adapter') and service.semantic_search_engine._db_adapter:
                    logger.info(f"   ✅ SemanticSearchEngineV2 DatabaseAdapter: type={service.semantic_search_engine._db_adapter.db_type}")
                if hasattr(service.semantic_search_engine, 'vector_adapter') and service.semantic_search_engine.vector_adapter:
                    adapter_type = type(service.semantic_search_engine.vector_adapter).__name__
                    logger.info(f"   ✅ SemanticSearchEngineV2 VectorAdapter: {adapter_type}")
            
            # 초기화 총 시간 계산
            init_total_time = time.time() - total_start_time
            logger.info(f"\n초기화 완료 (총 시간: {init_total_time:.3f}초)")
            
            # 🔥 개선: 초기화 완료 후 즉시 flush
            try:
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                        if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                            try:
                                os.fsync(handler.stream.fileno())
                            except (OSError, AttributeError):
                                pass
            except Exception:
                pass
                
        except Exception as e:
            # 🔥 개선: 초기화 실패 시 즉시 로그 기록
            logger.error(f"   ❌ 서비스 초기화 실패: {type(e).__name__}: {e}")
            logger.debug("상세 스택 트레이스:", exc_info=True)
            
            # 🔥 개선: 예외 발생 시 즉시 flush
            try:
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                        if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                            try:
                                os.fsync(handler.stream.fileno())
                            except (OSError, AttributeError):
                                pass
            except Exception:
                pass
            
            raise
        
        # 질의 처리
        logger.info("\n3. 질의 처리 중...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        query_start_time = time.time()
        
        logger.debug("   3.1. 검색 단계 시작...")
        
        # 🔥 개선: QueueHandler + QueueListener 패턴에서는 백그라운드 flush 태스크 불필요
        # QueueListener가 자동으로 큐에서 로그를 읽어 파일에 쓰므로 flush 호출 불필요
        try:
            
            # 🔥 개선: process_query 실행 (QueueHandler + QueueListener가 자동으로 로그 처리)
            result = None
            try:
                logger.info("   🔄 process_query 실행 시작...")
                
                result = await service.process_query(
                    query=query,
                    session_id="test_langgraph_query",
                    enable_checkpoint=False,
                    use_astream_events=True
                )
                
                logger.info("   ✅ process_query 실행 완료")
                
            except Exception as query_error:
                # 🔥 개선: 예외 발생 시 즉시 로그 기록
                try:
                    logger.error(f"   ❌ 질의 처리 중 오류 발생: {type(query_error).__name__}: {query_error}")
                    logger.error(f"   - 오류 타입: {type(query_error).__name__}")
                    logger.error(f"   - 오류 메시지: {str(query_error)}")
                    if hasattr(query_error, '__cause__') and query_error.__cause__:
                        logger.error(f"   - 원인: {query_error.__cause__}")
                    logger.debug("   상세 스택 트레이스:", exc_info=True)
                except Exception:
                    pass
                
                # 예외를 다시 발생시켜 상위에서 처리
                raise
        finally:
            # QueueHandler + QueueListener 패턴에서는 flush 불필요
            # QueueListener가 자동으로 큐에 남아있는 모든 로그를 처리함
            pass
        
        query_end_time = time.time()
        query_elapsed_time = query_end_time - query_start_time
        total_elapsed_time = query_end_time - total_start_time
        logger.info(f"   질의 처리 완료 (질의 처리 시간: {query_elapsed_time:.2f}초, 총 시간: {total_elapsed_time:.2f}초)")
        
        # 🔥 개선: result가 None인 경우 처리
        if result is None:
            logger.error("   ❌ 질의 처리 결과가 None입니다. 오류가 발생했을 수 있습니다.")
            raise ValueError("Query processing returned None result")
        
        # 결과 출력
        logger.info("\n4. 결과:")
        logger.info("=" * 80)
        
        # 답변 추출 (여러 위치에서 찾기)
        answer = result.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("answer", "") or answer.get("content", "") or ""
        answer = str(answer).strip() if answer else ""
        
        # answer가 비어있으면 다른 위치에서 찾기
        if not answer:
            # output 필드 확인
            output = result.get("output", {})
            if isinstance(output, dict):
                answer = output.get("answer", "") or output.get("content", "")
        
        # 최종 문자열 변환
        answer = str(answer).strip() if answer else ""
        
        if answer:
            logger.info(f"\n답변 ({len(answer)}자):")
            logger.info("-" * 80)
            logger.info(answer)
            # 🔥 개선: 답변 출력 후 즉시 flush
            flush_all_log_handlers()
        else:
            logger.warning("\n답변이 없습니다!")
            # 디버깅: result의 모든 키 출력
            logger.debug(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            if isinstance(result, dict) and "answer" in result:
                logger.debug(f"Answer type: {type(result['answer'])}, value: {str(result['answer'])[:100]}")
            # 🔥 개선: 경고 출력 후 즉시 flush
            flush_all_log_handlers()
        
        # 검색 결과 (품질 정보 포함)
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            logger.info(f"\n검색된 참고자료 ({len(retrieved_docs)}개):")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                if isinstance(doc, dict):
                    # 🔥 개선: 메타데이터 보강 - 최상위 필드를 metadata에 복사 (DocumentType 추론을 위해)
                    metadata = doc.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # 최상위 필드를 metadata에 복사 (DocumentType 추론을 위해)
                    for key in ["statute_name", "law_name", "article_no", "case_id", "court", "doc_id", "casenames", "precedent_id", "type", "source_type"]:
                        if key in doc and key not in metadata:
                            metadata[key] = doc[key]
                    
                    # metadata의 정보도 최상위 필드로 복사 (일관성 유지)
                    for key in ["statute_name", "law_name", "article_no", "case_id", "court", "doc_id", "casenames", "precedent_id", "type", "source_type"]:
                        if key in metadata and key not in doc:
                            doc[key] = metadata[key]
                    
                    doc["metadata"] = metadata
                    
                    # DocumentType Enum 사용하여 타입 추출
                    try:
                        from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                        # 디버깅: doc의 타입 관련 필드 확인
                        debug_type_info = {
                            "type": doc.get("type"),
                            "source_type": doc.get("source_type"),
                            "metadata_type": metadata.get("type"),
                            "metadata_source_type": metadata.get("source_type"),
                            "has_statute_fields": any(key in doc or key in metadata for key in ["statute_name", "law_name", "article_no"]),
                            "has_case_fields": any(key in doc or key in metadata for key in ["case_id", "court", "doc_id", "casenames", "precedent_id"]),
                        }
                        logger.info(f"🔍 [DOC TYPE DEBUG] Doc {i} type info: {debug_type_info}")
                        
                        doc_type_enum = DocumentType.from_metadata(doc)
                        doc_type = doc_type_enum.value
                        # 타입 이름을 한글로 변환
                        type_names = {
                            "statute_article": "법령",
                            "precedent_content": "판례",
                            "unknown": "알 수 없음"
                        }
                        doc_type_display = type_names.get(doc_type, doc_type)
                        
                        # 디버깅: 추론된 타입 로깅
                        if doc_type == "unknown":
                            logger.info(f"⚠️ [DOC TYPE DEBUG] Doc {i} inferred as UNKNOWN. Full doc keys: {list(doc.keys())[:20]}, metadata keys: {list(metadata.keys())[:20] if isinstance(metadata, dict) else 'N/A'}")
                    except Exception as e:
                        # 예외 발생 시 직접 필드 확인
                        logger.debug(f"⚠️ [DOC TYPE ERROR] Doc {i} type inference error: {e}")
                        doc_type = doc.get("type") or doc.get("source_type") or metadata.get("type") or metadata.get("source_type", "unknown")
                        # 레거시 호환: "case" -> "precedent_content"
                        if doc_type == "case":
                            doc_type = "precedent_content"
                        type_names = {
                            "statute_article": "법령",
                            "precedent_content": "판례",
                            "unknown": "알 수 없음"
                        }
                        doc_type_display = type_names.get(doc_type, doc_type)
                    
                    # 제목 추출 (여러 필드에서 시도)
                    title = (
                        doc.get("title") or 
                        doc.get("name") or 
                        doc.get("source") or
                        (doc.get("content", "")[:100] if doc.get("content") else "") or
                        (doc.get("text", "")[:100] if doc.get("text") else "") or
                        "제목 없음"
                    )
                    
                    # 점수 추출 (정규화된 점수 우선)
                    score = (
                        doc.get("relevance_score") or 
                        doc.get("final_weighted_score") or
                        doc.get("score") or 
                        doc.get("similarity") or 
                        0.0
                    )
                    score_display = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                    
                    # 내용 미리보기 (딕셔너리 형태 처리)
                    content = doc.get("content") or doc.get("text") or ""
                    if isinstance(content, dict):
                        # content가 딕셔너리인 경우 text 필드 추출
                        content = content.get("text", "") or content.get("content", "") or str(content)
                    if not isinstance(content, str):
                        content = str(content)
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    
                    logger.info(f"   {i}. [{doc_type_display}] {title}")
                    logger.info(f"       점수: {score_display}, 내용: {content_preview}")
                    # 🔥 개선: 각 문서 출력 후 주기적으로 flush (5개마다)
                    if i % 5 == 0:
                        flush_all_log_handlers()
                else:
                    logger.info(f"   {i}. {str(doc)[:100]}")
            if len(retrieved_docs) > 5:
                logger.info(f"   ... (총 {len(retrieved_docs)}개)")
            # 🔥 개선: 검색 결과 출력 완료 후 flush
            flush_all_log_handlers()
        else:
            logger.warning("\n검색된 참고자료가 없습니다!")
            # 🔥 개선: 경고 출력 후 즉시 flush
            flush_all_log_handlers()
        
        # 소스
        sources = result.get("sources", [])
        if sources:
            logger.info(f"\n소스 ({len(sources)}개):")
            for i, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    source_name = source.get("name") or source.get("title") or "제목 없음"
                    logger.info(f"   {i}. {source_name}")
                else:
                    logger.info(f"   {i}. {source}")
            if len(sources) > 5:
                logger.info(f"   ... (총 {len(sources)}개)")
            # 🔥 개선: 소스 출력 완료 후 flush
            flush_all_log_handlers()
        
        # 처리 시간 (측정된 시간과 결과의 시간 모두 표시)
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n처리 시간 (결과): {processing_time:.2f}초")
        if 'query_elapsed_time' in locals():
            logger.info(f"처리 시간 (측정): {query_elapsed_time:.2f}초")
        # 🔥 개선: 처리 시간 출력 후 flush
        flush_all_log_handlers()
        
        # 오류 확인
        errors = result.get("errors", [])
        if errors:
            logger.warning(f"\n오류 발생 ({len(errors)}개):")
            for i, error in enumerate(errors[:5], 1):
                logger.warning(f"   {i}. {error}")
            if len(errors) > 5:
                logger.warning(f"   ... (총 {len(errors)}개)")
            # 🔥 개선: 오류 출력 후 즉시 flush (중요!)
            flush_all_log_handlers()
        
        # 5. 결과 요약
        logger.info("\n5. 결과 요약:")
        logger.info("=" * 80)
        
        # 요약 정보 수집
        summary = {
            "질의": query,
            "답변 길이": len(answer) if answer else 0,
            "검색된 문서 수": len(retrieved_docs) if retrieved_docs else 0,
            "소스 수": len(sources) if sources else 0,
            "처리 시간": f"{processing_time:.2f}초" if processing_time else "N/A",
            "오류 수": len(errors) if errors else 0
        }
        
        logger.info("   요약 정보:")
        for key, value in summary.items():
            logger.info(f"   - {key}: {value}")
        
        # 🔥 개선: 요약 정보 출력 후 flush
        flush_all_log_handlers()
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 완료!")
        logger.info("=" * 80)
        
        # 🔥 개선: 테스트 완료 직후 즉시 flush (모든 로그 저장 보장)
        flush_all_log_handlers()
        
        # 🔥 개선: 리소스 정리 (데이터베이스 연결 풀 등)
        try:
            # 서비스가 cleanup 메서드를 가지고 있으면 호출
            if hasattr(service, 'cleanup'):
                service.cleanup()
            # 데이터베이스 연결 풀 정리 및 통계 출력
            if hasattr(service, 'legal_workflow') and service.legal_workflow:
                if hasattr(service.legal_workflow, 'data_connector') and service.legal_workflow.data_connector:
                    if hasattr(service.legal_workflow.data_connector, '_db_adapter') and service.legal_workflow.data_connector._db_adapter:
                        db_adapter = service.legal_workflow.data_connector._db_adapter
                        # 연결 통계 로깅 (최적화 방안 4)
                        if hasattr(db_adapter, 'log_connection_stats'):
                            try:
                                logger.info("\n" + "=" * 80)
                                logger.info("데이터베이스 연결 통계")
                                logger.info("=" * 80)
                                db_adapter.log_connection_stats()
                                pool_status = db_adapter.get_pool_status()
                                if pool_status.get('connection_stats'):
                                    stats = pool_status['connection_stats']
                                    logger.info(f"연결 횟수 상세:")
                                    logger.info(f"  - 총 연결 획득: {stats.get('total_getconn', 0)}회")
                                    logger.info(f"  - 총 연결 반환: {stats.get('total_putconn', 0)}회")
                                    if stats.get('getconn_by_method'):
                                        logger.debug(f"  - 메서드별 연결 획득: {stats['getconn_by_method']}")
                                    if stats.get('putconn_by_method'):
                                        logger.debug(f"  - 메서드별 연결 반환: {stats['putconn_by_method']}")
                            except Exception as stats_error:
                                logger.debug(f"연결 통계 로깅 실패 (무시): {stats_error}")
                        if hasattr(db_adapter, 'connection_pool') and db_adapter.connection_pool:
                            try:
                                # 연결 풀의 모든 연결 닫기
                                db_adapter.connection_pool.closeall()
                                logger.debug("데이터베이스 연결 풀 정리 완료")
                            except Exception as e:
                                logger.debug(f"연결 풀 정리 중 오류 (무시): {e}")
        except Exception as e:
            logger.debug(f"리소스 정리 중 오류 (무시): {e}")
        
        # 🔥 개선: 로그 파일에 모든 내용이 저장되도록 flush (강화)
        # UnbufferedFileHandler를 사용하므로 이미 flush되었지만, 최종 확인을 위해 다시 flush
        try:
            # 모든 로거의 모든 핸들러 flush
            loggers_to_flush = [
                logging.getLogger(),  # 루트 로거
                logging.getLogger("lawfirm_langgraph"),  # 하위 로거
                logging.getLogger("lawfirm_langgraph.tests.runners.run_query_test"),  # 테스트 로거
            ]
            
            for logger_to_flush in loggers_to_flush:
                for handler in logger_to_flush.handlers:
                    try:
                        handler.flush()
                        # FileHandler의 경우 stream도 직접 flush
                        if isinstance(handler, logging.FileHandler):
                            if hasattr(handler, 'stream') and handler.stream:
                                try:
                                    handler.stream.flush()
                                    # Windows에서 강제 동기화
                                    if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                                        try:
                                            os.fsync(handler.stream.fileno())
                                        except (OSError, AttributeError):
                                            pass
                                except (ValueError, AttributeError, OSError):
                                    pass
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"로그 flush 중 오류 (무시): {e}")
        
        return result
        
    except ImportError as e:
        # 🔥 개선: 예외 발생 전 기존 로그 flush
        flush_all_log_handlers()
        
        logger.error(f"\nImport 오류: {e}")
        logger.error("필요한 패키지가 설치되어 있는지 확인하세요.")
        logger.error("   패키지 설치: pip install -r requirements.txt")
        
        # 🔥 개선: 예외 발생 시 로그 기록 후 flush (여러 번 반복)
        for _ in range(5):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        raise
    except ValueError as e:
        # 🔥 개선: 예외 발생 전 기존 로그 flush
        flush_all_log_handlers()
        
        logger.error(f"\n설정 오류: {e}")
        logger.error("환경 변수 설정을 확인하세요.")
        logger.error("   PostgreSQL URL 설정:")
        logger.error("   - DATABASE_URL=postgresql://user:password@host:port/database")
        logger.error("   - 또는 POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD 환경 변수 설정")
        
        # 🔥 개선: 예외 발생 시 로그 기록 후 flush (여러 번 반복)
        for _ in range(5):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        raise
    except KeyboardInterrupt:
        # 🔥 개선: 예외 발생 전 기존 로그 flush
        flush_all_log_handlers()
        
        logger.warning("\n\n사용자에 의해 중단되었습니다.")
        
        # 🔥 개선: 중단 시 즉시 flush (여러 번 반복)
        for _ in range(5):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        # 중단 시에도 리소스 정리 시도
        try:
            if 'service' in locals():
                if hasattr(service, 'legal_workflow') and service.legal_workflow:
                    if hasattr(service.legal_workflow, 'data_connector') and service.legal_workflow.data_connector:
                        if hasattr(service.legal_workflow.data_connector, '_db_adapter') and service.legal_workflow.data_connector._db_adapter:
                            db_adapter = service.legal_workflow.data_connector._db_adapter
                            if hasattr(db_adapter, 'connection_pool') and db_adapter.connection_pool:
                                try:
                                    db_adapter.connection_pool.closeall()
                                except Exception:
                                    pass
        except Exception:
            pass
        
        # 🔥 개선: 리소스 정리 후 다시 flush (여러 번 반복)
        for _ in range(5):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        raise
    except Exception as e:
        # 🔥 개선: 예외 발생 전 기존 로그 flush (중요!) - 여러 번 반복
        for _ in range(3):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        # 🔥 개선: 예외 발생 시 즉시 로그 기록 및 flush
        try:
            logger.error(f"\n오류 발생: {type(e).__name__}: {e}")
            logger.error("   상세 정보:")
            logger.error(f"   - 오류 타입: {type(e).__name__}")
            logger.error(f"   - 오류 메시지: {str(e)}")
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"   - 원인: {e.__cause__}")
            logger.debug("   전체 스택 트레이스:", exc_info=True)
            
            # 🔥 개선: 각 로그 기록 후 즉시 flush
            for _ in range(3):
                flush_all_log_handlers()
                if sys.platform == 'win32':
                    time.sleep(0.01)
        except Exception:
            # 로그 기록 중 오류 발생 시에도 flush 시도
            try:
                flush_all_log_handlers()
            except Exception:
                pass
        
        # 🔥 개선: 예외 발생 시 즉시 flush (중요!) - 여러 번 반복 및 파일 동기화
        for _ in range(10):  # 더 많이 반복
            flush_all_log_handlers()
            # Windows에서 파일 동기화
            if sys.platform == 'win32':
                try:
                    for handler in logging.getLogger().handlers:
                        if isinstance(handler, logging.FileHandler) and hasattr(handler, 'stream') and handler.stream:
                            if hasattr(handler.stream, 'fileno'):
                                try:
                                    os.fsync(handler.stream.fileno())
                                except (OSError, AttributeError):
                                    pass
                except Exception:
                    pass
            time.sleep(0.02)  # 더 긴 대기
        
        # 오류 발생 시에도 리소스 정리 시도
        try:
            if 'service' in locals():
                if hasattr(service, 'legal_workflow') and service.legal_workflow:
                    if hasattr(service.legal_workflow, 'data_connector') and service.legal_workflow.data_connector:
                        if hasattr(service.legal_workflow.data_connector, '_db_adapter') and service.legal_workflow.data_connector._db_adapter:
                            db_adapter = service.legal_workflow.data_connector._db_adapter
                            # 연결 통계 로깅 (최적화 방안 4)
                            if hasattr(db_adapter, 'log_connection_stats'):
                                try:
                                    logger.info("\n" + "=" * 80)
                                    logger.info("데이터베이스 연결 통계 (오류 발생 시)")
                                    logger.info("=" * 80)
                                    db_adapter.log_connection_stats()
                                    pool_status = db_adapter.get_pool_status()
                                    if pool_status.get('connection_stats'):
                                        logger.info(f"연결 횟수 상세:")
                                        stats = pool_status['connection_stats']
                                        logger.info(f"  - 총 연결 획득: {stats.get('total_getconn', 0)}회")
                                        logger.info(f"  - 총 연결 반환: {stats.get('total_putconn', 0)}회")
                                        if stats.get('getconn_by_method'):
                                            logger.debug(f"  - 메서드별 연결 획득: {stats['getconn_by_method']}")
                                        if stats.get('putconn_by_method'):
                                            logger.debug(f"  - 메서드별 연결 반환: {stats['putconn_by_method']}")
                                except Exception as stats_error:
                                    logger.debug(f"연결 통계 로깅 실패 (무시): {stats_error}")
                            if hasattr(db_adapter, 'connection_pool') and db_adapter.connection_pool:
                                try:
                                    db_adapter.connection_pool.closeall()
                                except Exception:
                                    pass
        except Exception:
            pass
        
        # 🔥 개선: 리소스 정리 후 다시 flush (여러 번 반복)
        for _ in range(5):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        raise


def main():
    """메인 실행 함수"""
    global _global_log_file_path, _log_file_handle, _original_stdout, _original_stderr
    
    logger = None
    log_file_path = None
    
    try:
        # 🔥 개선: setup_logging() 호출 전에 로그 파일 경로 미리 결정
        # 이렇게 하면 setup_logging() 실패 시에도 로그 파일에 기록 가능
        log_dir_env = os.getenv("TEST_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env)
        else:
            log_dir = project_root / "logs" / "langgraph"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file_env = os.getenv("TEST_LOG_FILE")
        if log_file_env:
            log_file_path = str(Path(log_file_env))
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = str(log_dir / f"test_langgraph_query_{timestamp}.log")
        
        # 🔥 개선: 로그 파일을 미리 생성하고 초기 로그 기록
        # 이 로그는 setup_logging()이 실행되기 전에 기록되므로 매우 중요합니다
        try:
            pre_log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)
            pre_log_file.write(f"[프로그램 시작] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            pre_log_file.write(f"[로그 파일 경로] {log_file_path}\n")
            pre_log_file.write(f"[Python 버전] {sys.version}\n")
            pre_log_file.write(f"[작업 디렉토리] {os.getcwd()}\n")
            pre_log_file.write(f"[명령줄 인자] {sys.argv}\n")
            pre_log_file.flush()
            # Windows에서 파일 동기화
            if sys.platform == 'win32':
                try:
                    os.fsync(pre_log_file.fileno())
                except (OSError, AttributeError):
                    pass
            # 파일을 닫고, setup_logging()에서 append 모드로 열도록 함
            pre_log_file.close()
        except Exception as pre_log_error:
            # 로그 파일 미리 생성 실패 시 경고만 출력하고 계속 진행
            print(f"[경고] 로그 파일 미리 생성 실패: {pre_log_error}")
            import traceback
            traceback.print_exc()
    except Exception as init_error:
        # 초기화 중 오류 발생 시 콘솔에 출력
        print(f"[초기화 오류] {init_error}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        # 로깅 설정
        try:
            logger = setup_logging(log_file_path=log_file_path)
            
            # 🔥 개선: 로깅 설정 후 즉시 로그 기록 테스트
            if logger:
                logger.info("=" * 80)
                logger.info("테스트 시작")
                logger.info("=" * 80)
                flush_all_log_handlers()
        except Exception as setup_error:
            # setup_logging 실패 시에도 로그 파일에 기록
            try:
                with open(log_file_path, 'a', encoding='utf-8', buffering=1) as f:
                    f.write(f"\n[로그 설정 실패] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"  오류: {setup_error}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.flush()
            except Exception:
                pass
            
            print(f"\n[로그 설정 실패]")
            print(f"  오류: {setup_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # 질의 가져오기
        query = get_query_from_args()
        
        if not query:
            if logger:
                logger.error("질의를 입력해주세요.")
                logger.info("\n사용법:")
                logger.info("  python run_query_test.py \"질의 내용\"")
                logger.info("  python run_query_test.py 0  # 기본 질의 선택")
                logger.info("  $env:TEST_QUERY='질의내용'; python run_query_test.py")
            else:
                print("질의를 입력해주세요.")
            return 1
        
        # 테스트 실행
        try:
            # 🔥 개선: asyncio.run 호출 전 로그 flush 보장
            flush_all_log_handlers()
            
            # 🔥 개선: asyncio.run을 try-except로 감싸서 예외 처리 강화
            try:
                asyncio.run(test_langgraph_query(query, logger))
            except KeyboardInterrupt:
                # 🔥 개선: 예외 발생 전 기존 로그 flush
                flush_all_log_handlers()
                
                # 🔥 개선: KeyboardInterrupt는 별도 처리 (로그 기록 후 재발생)
                if logger:
                    logger.warning("\n\n사용자에 의해 중단되었습니다 (asyncio.run 내부).")
                
                # 🔥 개선: 로그 기록 후 flush (여러 번)
                for _ in range(5):
                    flush_all_log_handlers()
                    time.sleep(0.01)
                raise
            except Exception as async_error:
                # 🔥 개선: 예외 발생 전 기존 로그 flush (중요!) - 여러 번 반복
                for _ in range(3):
                    flush_all_log_handlers()
                    time.sleep(0.01)
                
                # 🔥 개선: 비동기 작업 중 예외 발생 시 즉시 로그 기록 및 flush
                if logger:
                    try:
                        logger.error(f"\n\n비동기 작업 중 오류 발생: {type(async_error).__name__}: {async_error}")
                        logger.error(f"   오류 타입: {type(async_error).__name__}")
                        logger.error(f"   오류 메시지: {str(async_error)}")
                        if hasattr(async_error, '__cause__') and async_error.__cause__:
                            logger.error(f"   원인: {async_error.__cause__}")
                        logger.debug("   전체 스택 트레이스:", exc_info=True)
                        
                        # 🔥 개선: 각 로그 기록 후 즉시 flush
                        for _ in range(3):
                            flush_all_log_handlers()
                            time.sleep(0.01)
                    except Exception:
                        # 로그 기록 중 오류 발생 시에도 flush 시도
                        try:
                            flush_all_log_handlers()
                        except Exception:
                            pass
                
                # 🔥 개선: 예외 발생 시 즉시 flush (여러 번) 및 파일 동기화
                for _ in range(10):  # 더 많이 반복
                    flush_all_log_handlers()
                    # Windows에서 파일 동기화
                    if sys.platform == 'win32':
                        try:
                            for handler in logging.getLogger().handlers:
                                if isinstance(handler, logging.FileHandler) and hasattr(handler, 'stream') and handler.stream:
                                    if hasattr(handler.stream, 'fileno'):
                                        try:
                                            os.fsync(handler.stream.fileno())
                                        except (OSError, AttributeError):
                                            pass
                        except Exception:
                            pass
                    time.sleep(0.02)  # 더 긴 대기
                raise
        except Exception as e:
            # 🔥 개선: 예외 발생 전 기존 로그 flush (중요!) - 여러 번 반복
            for _ in range(3):
                flush_all_log_handlers()
                if sys.platform == 'win32':
                    time.sleep(0.01)
            
            # 🔥 개선: 예외 발생 시 즉시 로그 기록 및 flush
            if logger:
                try:
                    logger.error(f"테스트 실행 중 오류 발생: {type(e).__name__}: {e}")
                    logger.error(f"   오류 타입: {type(e).__name__}")
                    logger.error(f"   오류 메시지: {str(e)}")
                    if hasattr(e, '__cause__') and e.__cause__:
                        logger.error(f"   원인: {e.__cause__}")
                    logger.debug("상세 스택 트레이스:", exc_info=True)
                    
                    # 🔥 개선: 각 로그 기록 후 즉시 flush
                    for _ in range(3):
                        flush_all_log_handlers()
                        if sys.platform == 'win32':
                            time.sleep(0.01)
                except Exception:
                    # 로그 기록 중 오류 발생 시에도 flush 시도
                    try:
                        flush_all_log_handlers()
                    except Exception:
                        pass
            
            # 🔥 개선: 예외 발생 시 즉시 flush (여러 번) 및 파일 동기화
            for _ in range(10):  # 더 많이 반복
                flush_all_log_handlers()
                # Windows에서 파일 동기화
                if sys.platform == 'win32':
                    try:
                        for handler in logging.getLogger().handlers:
                            if isinstance(handler, logging.FileHandler) and hasattr(handler, 'stream') and handler.stream:
                                if hasattr(handler.stream, 'fileno'):
                                    try:
                                        os.fsync(handler.stream.fileno())
                                    except (OSError, AttributeError):
                                        pass
                    except Exception:
                        pass
                time.sleep(0.02)  # 더 긴 대기
            raise
        finally:
            # 🔥 개선: 비동기 작업 완료 직후 즉시 flush (중요!) - 여러 번 반복
            for _ in range(5):
                flush_all_log_handlers()
                if sys.platform == 'win32':
                    time.sleep(0.01)
        
        # 🔥 개선: 테스트 완료 후 로그 파일 경로 출력
        if log_file_path:
            print(f"\n[테스트 완료]")
            print(f"  로그 파일: {log_file_path}")
            print(f"  로그 파일을 확인하여 메타데이터 보존 여부를 검증하세요.")
        
        # 🔥 개선: 최종 flush (테스트 완료 직후)
        flush_all_log_handlers()
        
        # 🔥 개선: 모든 로그 핸들러 flush 및 close (로그 파일 완전 저장 보장) - 강화
        try:
            # 모든 로거의 모든 핸들러 flush 및 close
            loggers_to_close = [
                logging.getLogger(),  # 루트 로거
                logging.getLogger("lawfirm_langgraph"),  # 하위 로거
                logging.getLogger("lawfirm_langgraph.tests.runners.run_query_test"),  # 테스트 로거
            ]
            
            for logger_to_close in loggers_to_close:
                for handler in logger_to_close.handlers[:]:  # 복사본으로 순회 (제거 중 변경 방지)
                    try:
                        # 먼저 flush
                        handler.flush()
                        
                        # FileHandler의 경우 stream도 직접 flush 및 동기화
                        if isinstance(handler, logging.FileHandler):
                            if hasattr(handler, 'stream') and handler.stream:
                                try:
                                    handler.stream.flush()
                                    # Windows에서 강제 동기화
                                    if sys.platform == 'win32' and hasattr(handler.stream, 'fileno'):
                                        try:
                                            os.fsync(handler.stream.fileno())
                                        except (OSError, AttributeError):
                                            pass
                                except (ValueError, AttributeError, OSError):
                                    pass
                            # 그 다음 close
                            handler.close()
                    except Exception:
                        pass
        except Exception as e:
            if logger:
                logger.debug(f"로그 핸들러 정리 중 오류 (무시): {e}")
        
        return 0
        
    except KeyboardInterrupt:
        if logger:
            logger.warning("\n\n사용자에 의해 중단되었습니다.")
        else:
            print("\n\n사용자에 의해 중단되었습니다.")
        
        # 🔥 개선: 중단 시에도 로그 파일 경로 출력
        if log_file_path:
            print(f"\n[테스트 중단]")
            print(f"  로그 파일: {log_file_path}")
        
        # 모든 핸들러 flush
        flush_all_log_handlers()
        
        return 1
    except Exception as e:
        # 🔥 개선: 예외 발생 전 기존 로그 flush (중요!) - 여러 번 반복
        for _ in range(3):
            flush_all_log_handlers()
            if sys.platform == 'win32':
                time.sleep(0.01)
        
        if logger:
            try:
                logger.error(f"\n\n테스트 실패: {e}")
                logger.error(f"   오류 타입: {type(e).__name__}")
                logger.error(f"   오류 메시지: {str(e)}")
                if hasattr(e, '__cause__') and e.__cause__:
                    logger.error(f"   원인: {e.__cause__}")
                logger.debug("상세 스택 트레이스:", exc_info=True)
                
                # 🔥 개선: 각 로그 기록 후 즉시 flush
                for _ in range(3):
                    flush_all_log_handlers()
                    if sys.platform == 'win32':
                        time.sleep(0.01)
            except Exception:
                # 로그 기록 중 오류 발생 시에도 flush 시도
                try:
                    flush_all_log_handlers()
                except Exception:
                    pass
        else:
            print(f"\n\n테스트 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 🔥 개선: 예외 발생 시에도 로그 파일 경로 출력
        if log_file_path:
            print(f"\n[오류 발생]")
            print(f"  로그 파일: {log_file_path}")
            print(f"  로그 파일을 확인하여 오류 원인을 파악하세요.")
        
        # 모든 핸들러 flush
        flush_all_log_handlers()
        
        return 1
    finally:
        # 🔥 개선: stdout/stderr 복원 및 로그 파일 핸들 정리
        global _log_file_handle, _original_stdout, _original_stderr
        
        # stdout/stderr 복원
        try:
            if '_original_stdout' in globals() and _original_stdout:
                sys.stdout = _original_stdout
        except Exception:
            pass
        
        try:
            if '_original_stderr' in globals() and _original_stderr:
                sys.stderr = _original_stderr
        except Exception:
            pass
        
        # 로그 파일 핸들러 flush 및 close
        try:
            if '_log_file_handle' in globals() and _log_file_handle:
                try:
                    _log_file_handle.flush()
                    # Windows에서 파일 동기화
                    if sys.platform == 'win32' and hasattr(_log_file_handle, 'fileno'):
                        try:
                            os.fsync(_log_file_handle.fileno())
                        except (OSError, AttributeError):
                            pass
                    _log_file_handle.close()
                except Exception:
                    pass
        except Exception:
            pass
        
        # 🔥 개선: finally 블록에서 모든 핸들러 flush (최종 보장)
        flush_all_log_handlers()


if __name__ == "__main__":
    sys.exit(main())

