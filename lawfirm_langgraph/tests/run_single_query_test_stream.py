# -*- coding: utf-8 -*-
"""
LangGraph 단일 질의 테스트 스크립트

Usage:
    python lawfirm_langgraph/tests/run_single_query_test.py "질의 내용"
    질의 내용이 없으면 기본 법률 질문을 사용합니다.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# UTF-8 인코딩 설정 (Windows PowerShell 호환)
# 주의: sys.stdout 재설정은 로깅 설정 전에 수행해야 함
# 로깅 핸들러는 원본 sys.stdout을 참조하도록 설정
_original_stdout = sys.stdout
_original_stderr = sys.stderr

if sys.platform == 'win32':
    # Windows에서 UTF-8 출력 설정
    import io
    
    # 표준 출력/에러 스트림을 UTF-8로 설정
    # 단, 로깅 핸들러는 원본을 사용하도록 주의
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            # 버퍼가 이미 분리된 경우 원본 사용
            pass
    if hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            # 버퍼가 이미 분리된 경우 원본 사용
            pass
    
    # 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # PowerShell 인코딩 설정 시도
    try:
        import subprocess
        # PowerShell 코드 페이지를 UTF-8로 설정
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
    except Exception:
        pass  # chcp 명령 실패해도 계속 진행

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))

# 로깅 설정
def setup_test_logging(log_to_file: bool = False, log_level: str = "INFO"):
    """
    테스트 로깅 설정
    
    Args:
        log_to_file: 로그를 파일로 저장할지 여부
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # 환경 변수에서 로깅 레벨 읽기
    env_log_level = os.getenv("TEST_LOG_LEVEL", log_level).upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = log_level_map.get(env_log_level, logging.INFO)
    
    # 로거 설정
    logger = logging.getLogger("lawfirm_langgraph.tests")
    logger.setLevel(level)
    logger.propagate = False  # 중복 로그 방지를 위해 False로 설정
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in list(logger.handlers):
        try:
            logger.removeHandler(handler)
        except Exception:
            pass
    
    # 콘솔 핸들러 추가
    # Windows에서 sys.stdout 재설정 후 버퍼 분리 문제 방지를 위해
    # 원본 stdout을 사용하거나 안전한 방식으로 처리
    console_handler = None
    
    # 모듈 레벨 변수 참조 (함수 내에서 직접 접근 가능)
    # Python에서는 함수 내에서 모듈 레벨 변수를 읽을 수 있음
    try:
        # 모듈 레벨에서 정의된 _original_stdout 참조
        # 함수 내에서 모듈 레벨 변수는 직접 참조 가능 (global 선언 없이 읽기 가능)
        original_stdout_ref = _original_stdout
    except NameError:
        # 모듈 레벨 변수가 없는 경우 None
        original_stdout_ref = None
    
    try:
        # 원본 stdout 사용 시도 (가장 안전한 방법)
        if original_stdout_ref is not None:
            try:
                console_handler = logging.StreamHandler(original_stdout_ref)
            except (ValueError, AttributeError, OSError):
                # 원본 stdout 사용 실패 시 현재 stdout 사용
                console_handler = logging.StreamHandler(sys.stdout)
        else:
            # 원본이 없는 경우 현재 stdout 사용
            console_handler = logging.StreamHandler(sys.stdout)
    except (NameError, AttributeError, ValueError, OSError):
        # 원본 stdout 사용 실패 시 현재 stdout 사용
        try:
            console_handler = logging.StreamHandler(sys.stdout)
        except (ValueError, AttributeError, OSError):
            # 모든 시도 실패 시 stderr 사용 (최후의 수단)
            try:
                console_handler = logging.StreamHandler(sys.stderr)
            except Exception:
                # 모든 핸들러 생성 실패
                console_handler = None
    
    if console_handler is None:
        # 핸들러 생성 실패 시 파일 핸들러만 사용
        pass
    else:
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 안전한 emit 메서드 생성 (버퍼 분리 오류 방지)
        # 클로저를 사용하여 original_stdout_ref를 캡처
        def create_safe_handler(base_handler, original_stdout):
            """안전한 핸들러 생성 함수"""
            class SafeStreamHandler(logging.StreamHandler):
                """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
                
                def __init__(self, stream, original_stdout_ref=None):
                    super().__init__(stream)
                    self._original_stdout = original_stdout_ref
                
                def emit(self, record):
                    """안전한 로그 출력 (버퍼 분리 오류 방지)"""
                    try:
                        # 원본 emit 시도
                        super().emit(record)
                    except (ValueError, AttributeError, OSError) as e:
                        # 버퍼 분리 오류 발생 시 대체 방법 시도
                        try:
                            # 포맷된 메시지를 직접 출력 시도
                            msg = self.format(record) + self.terminator
                            
                            # 스트림이 유효한지 확인
                            stream = self.stream
                            if stream is None:
                                stream = sys.stderr
                            
                            # 스트림에 직접 쓰기 시도
                            try:
                                if hasattr(stream, 'write'):
                                    stream.write(msg)
                                    if hasattr(stream, 'flush'):
                                        stream.flush()
                            except (ValueError, AttributeError, OSError):
                                # 원본 stdout에 직접 쓰기 시도
                                if self._original_stdout is not None:
                                    try:
                                        if hasattr(self._original_stdout, 'write'):
                                            self._original_stdout.write(msg)
                                            if hasattr(self._original_stdout, 'flush'):
                                                self._original_stdout.flush()
                                    except (ValueError, AttributeError, OSError):
                                        # 원본 stdout 실패 시 현재 stdout 시도
                                        try:
                                            if hasattr(sys.stdout, 'write'):
                                                sys.stdout.write(msg)
                                                if hasattr(sys.stdout, 'flush'):
                                                    sys.stdout.flush()
                                        except (ValueError, AttributeError, OSError):
                                            # 모든 시도 실패 시 stderr 사용
                                            try:
                                                sys.stderr.write(msg)
                                                sys.stderr.flush()
                                            except Exception:
                                                pass
                                else:
                                    # 원본이 없는 경우 현재 stdout 시도
                                    try:
                                        if hasattr(sys.stdout, 'write'):
                                            sys.stdout.write(msg)
                                            if hasattr(sys.stdout, 'flush'):
                                                sys.stdout.flush()
                                    except (ValueError, AttributeError, OSError):
                                        try:
                                            sys.stderr.write(msg)
                                            sys.stderr.flush()
                                        except Exception:
                                            pass
                        except Exception:
                            # 모든 로깅 시도 실패 시 무시 (안전한 실패)
                            # 테스트가 중단되지 않도록 함
                            pass
            
            # 안전한 핸들러로 교체
            safe_handler = SafeStreamHandler(base_handler.stream, original_stdout)
            safe_handler.setLevel(base_handler.level)
            safe_handler.setFormatter(base_handler.formatter)
            safe_handler.terminator = base_handler.terminator
            return safe_handler
        
        # 안전한 핸들러 생성 및 추가
        safe_handler = create_safe_handler(console_handler, original_stdout_ref)
        logger.addHandler(safe_handler)
        
        # 전역 로깅 설정: 모든 로거에 안전한 핸들러 적용 (멀티스레드 환경 대비)
        # transformers, sentence_transformers 등 외부 라이브러리 로거에도 적용
        
        # 로깅 예외 무시 설정 (멀티스레드 환경에서 버퍼 분리 오류 방지)
        logging.raiseExceptions = False
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 기존 핸들러 중 버퍼 분리 문제가 있는 핸들러 제거
        handlers_to_remove = []
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                try:
                    # 버퍼 분리 문제가 있는 핸들러 확인
                    if hasattr(handler, 'stream'):
                        stream = handler.stream
                        if stream is not None:
                            try:
                                # 버퍼 접근 테스트
                                if hasattr(stream, 'buffer'):
                                    _ = stream.buffer
                            except (ValueError, AttributeError, OSError):
                                # 버퍼 분리 문제가 있는 핸들러 제거
                                handlers_to_remove.append(handler)
                                continue
                except Exception:
                    handlers_to_remove.append(handler)
        
        # 안전하게 핸들러 제거
        for handler in handlers_to_remove:
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass
        
        # root logger에 안전한 핸들러 추가 (없는 경우에만)
        has_safe_handler = any(
            isinstance(h, type(safe_handler)) for h in root_logger.handlers
        )
        if not has_safe_handler:
            root_safe_handler = create_safe_handler(console_handler, original_stdout_ref)
            root_logger.addHandler(root_safe_handler)
        
        # transformers 라이브러리 로거에도 안전한 핸들러 적용
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)  # WARNING 이상만 출력
        transformers_logger.propagate = False  # root logger로 전파 방지
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in list(transformers_logger.handlers):
            try:
                transformers_logger.removeHandler(handler)
            except Exception:
                pass
        
        # 안전한 핸들러 추가
        transformers_safe_handler = create_safe_handler(console_handler, original_stdout_ref)
        transformers_logger.addHandler(transformers_safe_handler)
        
        # sentence_transformers 라이브러리 로거에도 안전한 핸들러 적용
        sentence_transformers_logger = logging.getLogger("sentence_transformers")
        sentence_transformers_logger.setLevel(logging.WARNING)  # WARNING 이상만 출력
        sentence_transformers_logger.propagate = False  # root logger로 전파 방지
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in list(sentence_transformers_logger.handlers):
            try:
                sentence_transformers_logger.removeHandler(handler)
            except Exception:
                pass
        
        # 안전한 핸들러 추가
        st_safe_handler = create_safe_handler(console_handler, original_stdout_ref)
        sentence_transformers_logger.addHandler(st_safe_handler)
        
        # transformers.utils.logging 로거에도 적용
        transformers_utils_logger = logging.getLogger("transformers.utils.logging")
        transformers_utils_logger.setLevel(logging.WARNING)
        transformers_utils_logger.propagate = False
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in list(transformers_utils_logger.handlers):
            try:
                transformers_utils_logger.removeHandler(handler)
            except Exception:
                pass
        
        # 안전한 핸들러 추가
        transformers_utils_safe_handler = create_safe_handler(console_handler, original_stdout_ref)
        transformers_utils_logger.addHandler(transformers_utils_safe_handler)
    
    # 파일 핸들러 추가 (옵션)
    if log_to_file:
        log_dir = project_root / "logs" / "tests"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"langgraph_test_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"로그 파일 저장: {log_file}")
        return logger, log_file
    
    return logger, None

# 로깅 초기화 (환경 변수로 제어)
LOG_TO_FILE = os.getenv("TEST_LOG_TO_FILE", "false").lower() == "true"
test_logger, log_file = setup_test_logging(log_to_file=LOG_TO_FILE)


def _try_recover_garbled_text(garbled_text: str) -> Optional[str]:
    """
    깨진 텍스트를 복구하는 함수 (PowerShell 인코딩 문제 해결)
    
    PowerShell에서 환경 변수로 한글을 설정할 때:
    - UTF-8 bytes가 CP949로 잘못 해석됨
    - 예: "계약서" (UTF-8: 0xEA 0xB3 0x84 0xEC 0x95 0xBD 0xEC 0x84 0x9C)
    -      -> CP949로 해석: "怨꾩빟"
    
    복구 방법:
    - 깨진 텍스트를 CP949로 인코딩하면 원본 UTF-8 bytes를 얻을 수 있음
    - 그 bytes를 UTF-8로 디코딩하면 원본 텍스트 복구 가능
    
    Args:
        garbled_text: 깨진 텍스트
    
    Returns:
        복구된 텍스트 또는 None
    """
    if not garbled_text:
        return None
    
    # 복구 전략 1: CP949 -> UTF-8 복구 (가장 일반적인 PowerShell 인코딩 문제)
    # PowerShell에서 UTF-8 텍스트가 CP949로 잘못 해석된 경우
    # 예: "계약서" (UTF-8: 0xEA 0xB3 0x84 0xEC 0x95 0xBD 0xEC 0x84 0x9C)
    #     -> CP949로 해석: "怨꾩빟"
    # 복구: "怨꾩빟"을 CP949로 인코딩하면 원본 UTF-8 bytes를 얻을 수 있음
    try:
        # 깨진 텍스트를 CP949 bytes로 인코딩하면 원본 UTF-8 bytes를 얻을 수 있음
        garbled_bytes = garbled_text.encode('cp949', errors='ignore')
        # 원본 UTF-8 bytes로 복구
        recovered = garbled_bytes.decode('utf-8', errors='replace')
        if recovered and recovered != garbled_text and len(recovered) > 0:
            # 복구된 텍스트가 한글을 포함하는지 확인
            has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in recovered)
            # '?' 문자가 적고, 한글이 있으면 복구 성공 가능성 높음
            question_ratio = recovered.count('?') / max(len(recovered), 1)
            # 깨진 문자 비율 확인 (한글 완성형 범위 외의 문자)
            garbled_chars = sum(1 for c in recovered if ord(c) > 0xFF and (ord(c) < 0xAC00 or ord(c) > 0xD7A3))
            garbled_ratio = garbled_chars / max(len(recovered), 1)
            
            # 복구 성공 조건: 한글이 있고, '?' 비율이 낮고, 깨진 문자 비율이 낮음
            if has_korean and question_ratio < 0.2 and garbled_ratio < 0.3:
                test_logger.info(f"✅ Recovered text using CP949->UTF-8: '{garbled_text[:30]}...' -> '{recovered[:30]}...'")
                return recovered
            else:
                test_logger.debug(f"CP949->UTF-8 recovery failed: has_korean={has_korean}, question_ratio={question_ratio:.2f}, garbled_ratio={garbled_ratio:.2f}")
    except Exception as e:
        test_logger.debug(f"CP949->UTF-8 recovery failed: {e}")
        pass
    
    # 복구 전략 1-2: UTF-16 -> UTF-8 복구 (PowerShell이 UTF-16으로 인코딩한 경우)
    try:
        # PowerShell이 UTF-16으로 인코딩한 경우를 대비
        # 깨진 텍스트를 UTF-16으로 인코딩 후 UTF-8로 디코딩 시도
        garbled_bytes = garbled_text.encode('utf-16-le', errors='ignore')
        recovered = garbled_bytes.decode('utf-8', errors='replace')
        if recovered and recovered != garbled_text and len(recovered) > 0:
            has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in recovered)
            question_ratio = recovered.count('?') / max(len(recovered), 1)
            garbled_chars = sum(1 for c in recovered if ord(c) > 0xFF and (ord(c) < 0xAC00 or ord(c) > 0xD7A3))
            garbled_ratio = garbled_chars / max(len(recovered), 1)
            
            if has_korean and question_ratio < 0.2 and garbled_ratio < 0.3:
                test_logger.debug(f"Recovered text using UTF-16->UTF-8: '{garbled_text[:30]}...' -> '{recovered[:30]}...'")
                return recovered
    except Exception as e:
        test_logger.debug(f"UTF-16->UTF-8 recovery failed: {e}")
        pass
    
    # 복구 전략 2: 여러 인코딩 조합 시도
    encodings = ['cp949', 'euc-kr', 'latin1']
    for src_enc in encodings:
        for dst_enc in ['utf-8']:
            if src_enc == dst_enc:
                continue
            try:
                # 소스 인코딩으로 인코딩 후 대상 인코딩으로 디코딩
                recovered = garbled_text.encode(src_enc, errors='ignore').decode(dst_enc, errors='replace')
                if recovered and recovered != garbled_text:
                    # 복구된 텍스트가 한글을 포함하는지 확인
                    has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in recovered)
                    question_ratio = recovered.count('?') / max(len(recovered), 1)
                    if has_korean and question_ratio < 0.2:
                        test_logger.debug(f"Recovered text using {src_enc}->{dst_enc}: '{garbled_text[:30]}...' -> '{recovered[:30]}...'")
                        return recovered
            except Exception:
                continue
    
    return None


def _validate_and_fix_query(query: str, default_query: str) -> str:
    """
    질의 검증 및 복구 함수
    
    Args:
        query: 검증할 질의
        default_query: 기본 질의 (복구 실패 시 사용)
    
    Returns:
        검증된 질의 또는 기본 질의
    """
    if not query or not isinstance(query, str):
        return default_query
    
    query = query.strip()
    
    if not query:
        return default_query
    
    # 깨진 문자 패턴 감지
    garbled_chars = sum(1 for c in query if ord(c) > 0xFF and (ord(c) < 0xAC00 or ord(c) > 0xD7A3))
    garbled_ratio = garbled_chars / max(len(query), 1)
    
    # '?' 문자 비율 확인
    question_mark_ratio = query.count('?') / max(len(query), 1)
    
    # 깨진 문자 비율이 30% 이상이거나 '?' 문자가 20% 이상이면 깨진 것으로 간주
    if garbled_ratio > 0.3 or question_mark_ratio > 0.2:
        # 복구 시도
        try:
            # 여러 인코딩 방식으로 복구 시도
            for encoding in ['cp949', 'euc-kr', 'latin1']:
                try:
                    # 원본을 bytes로 인코딩 후 다시 디코딩
                    fixed = query.encode(encoding, errors='ignore').decode('utf-8', errors='replace')
                    # 복구 후 검증
                    fixed_garbled = sum(1 for c in fixed if ord(c) > 0xFF and (ord(c) < 0xAC00 or ord(c) > 0xD7A3))
                    fixed_garbled_ratio = fixed_garbled / max(len(fixed), 1)
                    fixed_question_mark_ratio = fixed.count('?') / max(len(fixed), 1)
                    
                    if len(fixed) > 0 and fixed_garbled_ratio < 0.3 and fixed_question_mark_ratio < 0.2:
                        return fixed
                except Exception:
                    continue
        except Exception:
            pass
        
        # 복구 실패 시 기본 질의 반환
        return default_query
    
    # 정상적인 질의
    return query


async def run_single_query_test_streaming(query: str):
    """단일 질의 테스트 실행 (스트리밍 버전)"""
    test_logger.info("\n" + "="*80)
    test_logger.info("LangGraph 단일 질의 테스트 (스트리밍)")
    test_logger.info("="*80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.langgraph_core.state.state_definitions import create_initial_legal_state
        import uuid
        
        test_logger.info(f"\n📋 질의: {query}")
        test_logger.info("-" * 80)
        
        # 설정 로드
        test_logger.info("\n1️⃣  설정 로드 중...")
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        test_logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        test_logger.info(f"   ✅ 체크포인트 사용: {config.enable_checkpoint} (테스트 모드: 비활성화)")
        
        # 서비스 초기화
        test_logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        test_logger.info("   ✅ 서비스 초기화 완료")
        
        # 질의 처리 (스트리밍)
        test_logger.info("\n3️⃣  질의 처리 중 (스트리밍)...")
        test_logger.info("   (답변이 생성되는 동안 실시간으로 출력됩니다)")
        test_logger.info("-" * 80)
        
        # 세션 ID 생성
        session_id = "single_query_test"
        
        # 초기 상태 설정
        initial_state = create_initial_legal_state(query, session_id)
        config_dict = {"configurable": {"thread_id": session_id}}
        
        # 스트리밍 변수
        full_answer = ""
        answer_found = False
        tokens_received = 0
        event_count = 0
        llm_stream_count = 0
        
        # 최종 결과 저장
        final_result = None
        
        # astream_events()를 사용하여 스트리밍
        try:
            # 버전 호환성을 위한 래퍼
            async def get_stream_events():
                """버전 호환성을 위한 스트리밍 이벤트 래퍼"""
                try:
                    # version="v2" 시도 (LangGraph 최신 버전)
                    async for event in service.app.astream_events(
                        initial_state, 
                        config_dict,
                        version="v2"
                    ):
                        yield event
                except (TypeError, AttributeError):
                    # version 파라미터가 지원되지 않는 경우 (구버전)
                    async for event in service.app.astream_events(
                        initial_state, 
                        config_dict
                    ):
                        yield event
            
            # 스트리밍 이벤트 처리
            event_types_seen = set()  # 본 이벤트 타입 추적
            node_names_seen = set()  # 본 노드 이름 추적
            
            async for event in get_stream_events():
                event_count += 1
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                
                # 이벤트 타입과 노드 이름 추적
                event_types_seen.add(event_type)
                if event_name:
                    node_names_seen.add(event_name)
                
                # 디버깅: 이벤트 타입 로깅 (처음 10개만, DEBUG 레벨)
                if event_count <= 10:
                    test_logger.debug(f"스트리밍 이벤트 #{event_count}: type={event_type}, name={event_name}")
                
                # LLM 스트리밍 이벤트 감지 (답변 생성 노드에서만)
                # LangGraph/LangChain 최신 버전에서는 on_chat_model_stream도 지원
                if event_type in ["on_llm_stream", "on_chat_model_stream"]:
                    llm_stream_count += 1
                    test_logger.debug(f"{event_type} 이벤트 발견: name={event_name}, 전체 이벤트 키: {list(event.keys())}")
                    
                    # 답변 생성 관련 노드인지 확인 (더 많은 패턴 지원)
                    # ChatGoogleGenerativeAI는 LLM 모델 자체이므로 항상 처리
                    is_answer_node = (
                        "generate_answer" in event_name.lower() or 
                        "generate_and_validate" in event_name.lower() or
                        "answer" in event_name.lower() or
                        event_name in ["generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"] or
                        event_type == "on_chat_model_stream"  # on_chat_model_stream은 항상 처리
                    )
                    
                    # 디버깅: 모든 스트리밍 이벤트 로깅 (처음 5개만, DEBUG 레벨)
                    if llm_stream_count <= 5:
                        test_logger.debug(f"{event_type} 이벤트 #{llm_stream_count}: name={event_name}, is_answer_node={is_answer_node}")
                        # 이벤트 구조 상세 로깅 (처음 3개만)
                        if llm_stream_count <= 3:
                            event_data = event.get("data", {})
                            test_logger.debug(f"  이벤트 구조: event_data type={type(event_data)}, event_data keys={list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}")
                            if isinstance(event_data, dict):
                                chunk_obj = event_data.get("chunk")
                                if chunk_obj is not None:
                                    test_logger.debug(f"  chunk_obj type={type(chunk_obj)}, chunk_obj={chunk_obj}")
                    
                    if is_answer_node:
                        # 첫 번째 이벤트만 INFO 레벨로 로깅
                        if llm_stream_count == 1:
                            test_logger.info(f"✅ 답변 생성 노드에서 {event_type} 이벤트 감지: {event_name}")
                        else:
                            test_logger.debug(f"✅ 답변 생성 노드에서 {event_type} 이벤트 감지: {event_name}")
                    else:
                        # 답변 생성 노드가 아닌 경우에도 로깅 (디버깅용, DEBUG 레벨)
                        if llm_stream_count <= 5:
                            test_logger.debug(f"답변 생성 노드가 아님: {event_name} (무시)")
                    
                    # 노드 이름 필터링 없이 모든 on_chat_model_stream 이벤트에서 토큰 추출 시도
                    # (노드 이름이 정확히 일치하지 않을 수 있으므로)
                    if event_type == "on_chat_model_stream":
                        # 모든 on_chat_model_stream 이벤트에서 토큰 추출 시도
                        if not is_answer_node:
                            # 답변 생성 노드가 아니어도 일단 토큰 추출 시도 (디버깅용, DEBUG 레벨)
                            if llm_stream_count <= 3:
                                test_logger.debug(f"⚠️ 답변 생성 노드가 아니지만 토큰 추출 시도: {event_name}")
                    
                    if is_answer_node or (event_type == "on_chat_model_stream" and llm_stream_count <= 10):
                        # 토큰 추출
                        chunk = None
                        event_data = event.get("data", {})
                        
                        try:
                            # 경우 1: LangChain 표준 형식 - data.chunk.content
                            if isinstance(event_data, dict):
                                chunk_obj = event_data.get("chunk")
                                if chunk_obj is not None:
                                    # AIMessageChunk 객체 처리
                                    if hasattr(chunk_obj, "content"):
                                        content = chunk_obj.content
                                        # content가 문자열이면 그대로 사용
                                        if isinstance(content, str):
                                            chunk = content
                                        # content가 리스트인 경우 (AIMessageChunk의 content는 리스트일 수 있음)
                                        elif isinstance(content, list) and len(content) > 0:
                                            # 리스트의 첫 번째 요소가 문자열이면 사용
                                            if isinstance(content[0], str):
                                                chunk = content[0]
                                            else:
                                                chunk = str(content[0])
                                        else:
                                            chunk = str(content)
                                    elif isinstance(chunk_obj, str):
                                        chunk = chunk_obj
                                    elif hasattr(chunk_obj, "text"):
                                        chunk = chunk_obj.text
                                    # AIMessageChunk 객체의 경우 직접 content 접근 시도
                                    elif hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
                                        try:
                                            content = getattr(chunk_obj, "content", None)
                                            if isinstance(content, str):
                                                chunk = content
                                            elif isinstance(content, list) and len(content) > 0:
                                                if isinstance(content[0], str):
                                                    chunk = content[0]
                                                else:
                                                    chunk = str(content[0])
                                            elif content is not None:
                                                chunk = str(content)
                                        except Exception:
                                            pass
                                
                                # 경우 2: 직접 문자열 형식
                                if not chunk:
                                    chunk = event_data.get("text") or event_data.get("content")
                                
                                # 경우 3: delta 형식 (LangGraph v2)
                                if not chunk and "delta" in event_data:
                                    delta = event_data["delta"]
                                    if isinstance(delta, dict):
                                        chunk = delta.get("content") or delta.get("text")
                                    elif isinstance(delta, str):
                                        chunk = delta
                            
                            # 경우 4: 이벤트 최상위 레벨에 직접 포함
                            if not chunk:
                                chunk = event.get("chunk") or event.get("text") or event.get("content")
                            
                            # 토큰이 있으면 즉시 출력
                            if chunk and isinstance(chunk, str) and len(chunk) > 0:
                                # JSON 응답 필터링 (검증 결과 등)
                                if chunk.strip().startswith('{') or chunk.strip().startswith('```json'):
                                    # JSON 응답은 로깅만 하고 출력하지 않음
                                    if tokens_received <= 5:
                                        test_logger.debug(f"JSON 응답 필터링: {chunk[:100]}...")
                                    continue
                                
                                full_answer += chunk
                                tokens_received += 1
                                answer_found = True
                                # 실시간 출력 (버퍼링 없이)
                                print(chunk, end='', flush=True)
                                # 디버깅: 토큰 추출 성공 로깅 (처음 10개만)
                                if tokens_received <= 10:
                                    test_logger.debug(f"✅ 토큰 추출 성공 #{tokens_received}: chunk='{chunk[:50]}...', length={len(chunk)}")
                            else:
                                # 토큰 추출 실패 로깅 (처음 3개만, DEBUG 레벨)
                                if llm_stream_count <= 3:
                                    test_logger.debug(f"⚠️ 토큰 추출 실패: chunk={chunk}, chunk type={type(chunk) if chunk else 'None'}")
                                    test_logger.debug(f"  event_data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}")
                                    if isinstance(event_data, dict):
                                        chunk_obj = event_data.get("chunk")
                                        if chunk_obj is not None:
                                            test_logger.debug(f"  chunk_obj type={type(chunk_obj)}, chunk_obj={chunk_obj}")
                                
                        except (AttributeError, TypeError, KeyError) as e:
                            # 이벤트 구조가 예상과 다를 경우 로깅만 하고 계속 진행
                            test_logger.debug(f"토큰 추출 실패 (이벤트 구조가 예상과 다름): {e}, event_keys={list(event.keys()) if isinstance(event, dict) else 'N/A'}")
                            # 디버깅: 이벤트 구조 상세 로깅 (처음 3개만)
                            if llm_stream_count <= 3:
                                test_logger.debug(f"이벤트 구조 상세: event_data={event_data}, event_data type={type(event_data)}")
                                if isinstance(event_data, dict):
                                    test_logger.debug(f"event_data keys: {list(event_data.keys())}")
                            continue
                
                # LLM 완료 이벤트 (on_llm_end 또는 on_chat_model_end)
                elif event_type in ["on_llm_end", "on_chat_model_end"]:
                    # 최종 답변 확인 (누락된 부분이 있는지 체크)
                    try:
                        event_data = event.get("data", {})
                        if isinstance(event_data, dict):
                            output = event_data.get("output")
                            if output is not None:
                                final_answer = None
                                
                                # 다양한 출력 형식 지원
                                if hasattr(output, "content"):
                                    final_answer = output.content
                                elif isinstance(output, str):
                                    final_answer = output
                                elif isinstance(output, dict):
                                    final_answer = output.get("content") or output.get("text") or str(output)
                                else:
                                    final_answer = str(output)
                                
                                # 누락된 부분이 있으면 출력
                                if final_answer and isinstance(final_answer, str):
                                    if len(final_answer) > len(full_answer):
                                        missing_part = final_answer[len(full_answer):]
                                        if missing_part:
                                            full_answer = final_answer
                                            print(missing_part, end='', flush=True)
                                            test_logger.debug(f"누락된 부분 출력: {len(missing_part)}자")
                    except (AttributeError, TypeError, KeyError) as e:
                        test_logger.debug(f"on_llm_end 이벤트 처리 실패: {e}")
                        pass
                
                # 노드 완료 이벤트 (최종 포맷팅된 답변 확인)
                elif event_type == "on_chain_end":
                    node_name = event.get("name", "")
                    if node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]:
                        try:
                            event_data = event.get("data", {})
                            if isinstance(event_data, dict):
                                output = event_data.get("output")
                                if output is not None:
                                    # answer 필드 확인 (다양한 구조 지원)
                                    final_formatted_answer = None
                                    
                                    if isinstance(output, dict):
                                        # 최상위 레벨
                                        final_formatted_answer = output.get("answer", "")
                                        
                                        # common 그룹
                                        if not final_formatted_answer and "common" in output:
                                            common = output.get("common", {})
                                            if isinstance(common, dict):
                                                final_formatted_answer = common.get("answer", "")
                                        
                                        # generation 그룹
                                        if not final_formatted_answer and "generation" in output:
                                            generation = output.get("generation", {})
                                            if isinstance(generation, dict):
                                                final_formatted_answer = generation.get("answer", "")
                                    
                                    if final_formatted_answer and isinstance(final_formatted_answer, str) and len(final_formatted_answer) > 0:
                                        # 스트리밍이 없었을 때: 전체 답변 출력
                                        if not answer_found:
                                            full_answer = final_formatted_answer
                                            answer_found = True
                                            print(final_formatted_answer, end='', flush=True)
                                            test_logger.info("스트리밍 이벤트 없음, on_chain_end에서 폴백 출력")
                                        else:
                                            # 스트리밍이 있었을 때: 최종 포맷팅된 답변과 비교하여 누락된 부분 보완
                                            if final_formatted_answer != full_answer:
                                                if len(final_formatted_answer) > len(full_answer):
                                                    # 최종 답변이 더 긴 경우: 누락된 부분 출력
                                                    missing_part = final_formatted_answer[len(full_answer):]
                                                    if missing_part:
                                                        full_answer = final_formatted_answer
                                                        print(missing_part, end='', flush=True)
                                                        test_logger.info(f"최종 포맷팅된 답변에서 누락된 부분 출력: {len(missing_part)}자")
                                                # 최종 결과 저장 (포맷팅된 답변)
                                                final_result = output
                        except (AttributeError, TypeError, KeyError) as e:
                            test_logger.debug(f"on_chain_end 이벤트 처리 실패: {e}")
                            pass
            
            # 스트리밍 완료
            print()  # 줄바꿈
            test_logger.info(f"\n스트리밍 완료: 총 {event_count}개 이벤트, LLM 스트리밍 이벤트 {llm_stream_count}개, 토큰 수신 {tokens_received}개")
            test_logger.info(f"발생한 이벤트 타입: {sorted(event_types_seen)}")
            test_logger.info(f"발생한 노드 이름 (답변 생성 관련): {[n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]}")
            
            # 디버깅: 발생한 모든 이벤트 타입과 노드 이름 로깅
            if llm_stream_count == 0:
                test_logger.warning("⚠️ LLM 스트리밍 이벤트가 발생하지 않았습니다.")
                test_logger.debug(f"발생한 모든 이벤트 타입: {sorted(event_types_seen)}")
                test_logger.debug(f"발생한 모든 노드 이름: {sorted(node_names_seen)}")
                # 답변 생성 관련 노드가 실행되었는지 확인
                answer_nodes_executed = [n for n in sorted(node_names_seen) if 'answer' in n.lower() or 'generate' in n.lower()]
                if answer_nodes_executed:
                    test_logger.info(f"답변 생성 관련 노드 실행됨: {answer_nodes_executed}")
                else:
                    test_logger.warning("답변 생성 관련 노드가 실행되지 않았습니다.")
            
            # 최종 결과가 없으면 process_query()로 폴백
            if not final_result:
                test_logger.info("\n최종 결과를 가져오기 위해 process_query() 호출...")
                final_result = await service.process_query(
                    query=query,
                    session_id=session_id,
                    enable_checkpoint=False
                )
            
        except Exception as stream_error:
            test_logger.warning(f"스트리밍 실패, process_query()로 폴백: {stream_error}")
            final_result = await service.process_query(
                query=query,
                session_id=session_id,
                enable_checkpoint=False
            )
        
        # 결과 출력
        test_logger.info("\n4️⃣  결과:")
        test_logger.info("="*80)
        
        if final_result:
            # 답변은 이미 스트리밍으로 출력했으므로, 다른 정보만 출력
            answer = final_result.get("answer", full_answer or "")
            
            # 소스 정보
            sources = final_result.get("sources", [])
            if sources:
                test_logger.info(f"\n📚 소스 ({len(sources)}개):")
                test_logger.info("-" * 80)
                for i, source in enumerate(sources[:5], 1):
                    test_logger.info(f"   {i}. {source}")
                if len(sources) > 5:
                    test_logger.info(f"   ... (총 {len(sources)}개)")
            
            # 법률 참조
            legal_references = final_result.get("legal_references", [])
            if legal_references:
                test_logger.info(f"\n⚖️  법률 참조 ({len(legal_references)}개):")
                test_logger.info("-" * 80)
                for i, ref in enumerate(legal_references[:5], 1):
                    test_logger.info(f"   {i}. {ref}")
                if len(legal_references) > 5:
                    test_logger.info(f"   ... (총 {len(legal_references)}개)")
            
            # 신뢰도
            confidence = final_result.get("confidence", 0.0)
            if confidence:
                test_logger.info(f"\n🎯 신뢰도: {confidence:.2f}")
            
            # 처리 시간
            processing_time = final_result.get("processing_time", 0.0)
            if processing_time:
                test_logger.info(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
        test_logger.info("\n" + "="*80)
        test_logger.info("✅ 테스트 완료!")
        test_logger.info("="*80)
        
        if log_file:
            test_logger.info(f"\n📄 로그 파일: {log_file}")
        
        return final_result or {"answer": full_answer}
        
    except ImportError as e:
        test_logger.error(f"\n❌ Import 오류: {e}")
        test_logger.error("\n필요한 패키지가 설치되어 있는지 확인하세요:")
        test_logger.error("  - lawfirm_langgraph.config.langgraph_config")
        test_logger.error("  - lawfirm_langgraph.langgraph_core.workflow.workflow_service")
        raise
        
    except Exception as e:
        test_logger.error(f"\n❌ 오류 발생: {type(e).__name__}: {e}", exc_info=True)
        import traceback
        test_logger.error("\n상세 오류:")
        test_logger.error(traceback.format_exc())
        raise


async def run_single_query_test(query: str):
    """단일 질의 테스트 실행"""
    test_logger.info("\n" + "="*80)
    test_logger.info("LangGraph 단일 질의 테스트")
    test_logger.info("="*80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        
        test_logger.info(f"\n📋 질의: {query}")
        test_logger.info("-" * 80)
        
        # 설정 로드
        test_logger.info("\n1️⃣  설정 로드 중...")
        config = LangGraphConfig.from_env()
        # 테스트를 위해 체크포인트 비활성화
        config.enable_checkpoint = False
        test_logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        test_logger.info(f"   ✅ 체크포인트 사용: {config.enable_checkpoint} (테스트 모드: 비활성화)")
        
        # 서비스 초기화
        test_logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        test_logger.info("   ✅ 서비스 초기화 완료")
        
        # 질의 처리
        test_logger.info("\n3️⃣  질의 처리 중...")
        test_logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="single_query_test",
            enable_checkpoint=False  # 테스트이므로 체크포인트 비활성화
        )
        
        test_logger.info("\n4️⃣  결과:")
        test_logger.info("="*80)
        
        # 답변 추출
        answer = result.get("answer", "")
        answer_text = answer
        if isinstance(answer_text, dict):
            # 중첩된 딕셔너리에서 답변 추출 시도
            for key in ("answer", "content", "text"):
                if isinstance(answer_text, dict) and key in answer_text:
                    answer_text = answer_text[key]
            if isinstance(answer_text, dict):
                answer_text = str(answer_text)
        
        # 답변 출력 (개선: 전체 답변 출력)
        test_logger.info(f"\n📝 답변 (길이: {len(str(answer_text)) if answer_text else 0}자):")
        test_logger.info("-" * 80)
        if answer_text:
            # 개선: 전체 답변 출력 (1000자 제한 해제)
            full_answer = str(answer_text)
            test_logger.info(full_answer)
            if len(full_answer) > 5000:
                test_logger.info(f"\n... (총 {len(full_answer)}자, 전체 출력 완료)")
        else:
            test_logger.warning("<답변 없음>")
        
        # 소스 정보
        sources = result.get("sources", [])
        if sources:
            test_logger.info(f"\n📚 소스 ({len(sources)}개):")
            test_logger.info("-" * 80)
            for i, source in enumerate(sources[:5], 1):  # 최대 5개만 출력
                test_logger.info(f"   {i}. {source}")
            if len(sources) > 5:
                test_logger.info(f"   ... (총 {len(sources)}개)")
        
        # 법률 참조
        legal_references = result.get("legal_references", [])
        if legal_references:
            test_logger.info(f"\n⚖️  법률 참조 ({len(legal_references)}개):")
            test_logger.info("-" * 80)
            for i, ref in enumerate(legal_references[:5], 1):
                test_logger.info(f"   {i}. {ref}")
            if len(legal_references) > 5:
                test_logger.info(f"   ... (총 {len(legal_references)}개)")
        
        # 메타데이터
        metadata = result.get("metadata", {})
        if metadata:
            test_logger.info(f"\n📊 메타데이터:")
            test_logger.info("-" * 80)
            for key, value in list(metadata.items())[:10]:  # 최대 10개만 출력
                test_logger.info(f"   {key}: {value}")
        
        # 신뢰도
        confidence = result.get("confidence", 0.0)
        if confidence:
            test_logger.info(f"\n🎯 신뢰도: {confidence:.2f}")
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            test_logger.info(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
        test_logger.info("\n" + "="*80)
        test_logger.info("✅ 테스트 완료!")
        test_logger.info("="*80)
        
        if log_file:
            test_logger.info(f"\n📄 로그 파일: {log_file}")
        
        return result
        
    except ImportError as e:
        test_logger.error(f"\n❌ Import 오류: {e}")
        test_logger.error("\n필요한 패키지가 설치되어 있는지 확인하세요:")
        test_logger.error("  - lawfirm_langgraph.config.langgraph_config")
        test_logger.error("  - lawfirm_langgraph.langgraph_core.workflow.workflow_service")
        raise
        
    except Exception as e:
        test_logger.error(f"\n❌ 오류 발생: {type(e).__name__}: {e}", exc_info=True)
        import traceback
        test_logger.error("\n상세 오류:")
        test_logger.error(traceback.format_exc())
        raise


def main():
    """메인 실행 함수"""
    # 기본 질의 목록
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    default_query = default_queries[1]  # "민법 제750조 손해배상"을 기본값으로 변경
    
    # 질의 선택 방법 (우선순위):
    # 1. 환경 변수 TEST_QUERY (인코딩 문제 회피용)
    # 2. 파일에서 읽기 (-f 또는 --file 옵션)
    # 3. 명령줄 인자로 숫자 (0, 1, 2 등) - 기본 질의 목록에서 선택
    # 4. 명령줄 인자로 직접 질의 텍스트
    # 5. 인자가 없으면 첫 번째 기본 질의 사용
    
    query = None
    
    # 1. 환경 변수에서 질의 읽기 (인코딩 문제 회피용)
    test_query_env = os.getenv('TEST_QUERY')
    if test_query_env and test_query_env.strip():
        query = test_query_env.strip()
        original_query = query
        
        # 환경 변수에서 읽은 질의는 PowerShell 인코딩 문제로 깨질 수 있으므로
        # 먼저 복구 시도 (PowerShell에서 UTF-8이 CP949로 잘못 해석된 경우)
        recovered_query = _try_recover_garbled_text(query)
        if recovered_query and recovered_query != query:
            test_logger.info(f"\n💡 환경 변수 TEST_QUERY에서 질의를 읽었습니다 (복구됨).")
            test_logger.debug(f"   원본: '{query[:50]}...'")
            test_logger.debug(f"   복구: '{recovered_query[:50]}...'")
            query = recovered_query
        else:
            # 복구 실패 시 원본 사용
            query = original_query
        
        # 복구 후 검증
        query_validated = _validate_and_fix_query(query, default_query)
        
        # 검증 결과가 기본 질의와 다르면 정상적으로 검증된 것으로 간주
        if query_validated != default_query:
            query = query_validated
            test_logger.info(f"\n💡 환경 변수 TEST_QUERY에서 질의를 읽었습니다: '{query}'")
            test_logger.info(f"   사용법: $env:TEST_QUERY='계약서 작성 시 주의사항은 무엇인가요?'; python run_single_query_test.py")
            test_logger.info(f"   또는: set TEST_QUERY=계약서 작성 시 주의사항은 무엇인가요? && python run_single_query_test.py")
        else:
            # 검증 실패 시에도 복구된 질의가 있으면 사용
            if recovered_query and recovered_query != original_query:
                query = recovered_query
                test_logger.info(f"\n💡 환경 변수 TEST_QUERY에서 질의를 읽었습니다 (복구된 질의 사용): '{query}'")
            else:
                query = default_query
                test_logger.warning(f"\n⚠️  환경 변수 TEST_QUERY의 질의를 복구할 수 없습니다. 기본 질의를 사용합니다.")
                test_logger.warning(f"   원본 질의: {original_query[:100]}...")
                test_logger.warning(f"   💡 팁: PowerShell에서 환경 변수 설정 시 인코딩 문제가 발생할 수 있습니다.")
                test_logger.warning(f"   💡 대안: 파일로 질의를 저장하고 -f 옵션으로 읽기: python run_single_query_test.py -f query.txt")
    
    # 2. 파일에서 질의 읽기
    if not query and len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ['-f', '--file']:
            if len(sys.argv) > 2:
                file_path = sys.argv[2]
                try:
                    # 여러 인코딩 시도
                    for encoding in ['utf-8', 'cp949', 'euc-kr']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                query = f.read().strip()
                            if query:
                                test_logger.info(f"\n💡 파일에서 질의를 읽었습니다: {file_path} (인코딩: {encoding})")
                                break
                        except (UnicodeDecodeError, FileNotFoundError):
                            continue
                    if not query:
                        test_logger.error(f"\n❌ 파일을 읽을 수 없습니다: {file_path}")
                        return
                except Exception as e:
                    test_logger.error(f"\n❌ 파일 읽기 오류: {e}")
                    return
            else:
                test_logger.error(f"\n❌ 파일 경로를 지정해주세요: python run_single_query_test.py -f <파일경로>")
                return
    
    # 3. 명령줄 인자 처리
    if not query and len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        # 파일 옵션은 이미 처리했으므로 건너뛰기
        if arg in ['-f', '--file']:
            pass  # 이미 처리됨
        elif arg.isdigit():
            # 숫자로 시작하면 기본 질의 목록에서 선택
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                query = default_queries[idx]
                test_logger.info(f"\n💡 기본 질의 목록에서 선택: [{idx}]")
            else:
                test_logger.warning(f"\n⚠️  인덱스 {idx}가 범위를 벗어났습니다. 기본 질의를 사용합니다.")
                query = default_query
        else:
            # 직접 질의 텍스트로 간주
            # PowerShell 인코딩 문제 해결을 위해 여러 인자를 합침
            query_parts = sys.argv[1:]
            
            # 인코딩 문제 해결: 여러 인코딩 방식 시도
            decoded_parts = []
            for part in query_parts:
                if isinstance(part, bytes):
                    # bytes인 경우 여러 인코딩 시도
                    for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin1']:
                        try:
                            decoded = part.decode(encoding)
                            decoded_parts.append(decoded)
                            break
                        except (UnicodeDecodeError, AttributeError):
                            continue
                    else:
                        # 모든 인코딩 실패 시 errors='replace'로 디코딩
                        decoded_parts.append(part.decode('utf-8', errors='replace'))
                else:
                    # 이미 문자열인 경우
                    # Windows PowerShell에서 깨진 인코딩 복구 시도
                    if isinstance(part, str):
                        # 깨진 문자 패턴 감지
                        has_garbled = False
                        for c in part:
                            if len(c.encode('utf-8')) > 1:
                                char_code = ord(c)
                                if (char_code > 0x7F and char_code < 0xAC00) or char_code > 0xD7A3:
                                    has_garbled = True
                                    break
                        if has_garbled or '?' in part or any(ord(c) > 0xFF for c in part):
                            # 깨진 문자열인 경우 복구 시도
                            try:
                                # 여러 인코딩 방식으로 복구 시도
                                for encoding in ['cp949', 'euc-kr', 'latin1']:
                                    try:
                                        # 원본을 bytes로 인코딩 후 다시 디코딩
                                        fixed = part.encode(encoding, errors='ignore').decode('utf-8', errors='replace')
                                        if len(fixed) > 0 and not all(ord(c) < 0x20 or ord(c) > 0x7E for c in fixed[:10]):
                                            decoded_parts.append(fixed)
                                            break
                                    except Exception:
                                        continue
                                else:
                                    # 모든 복구 실패 시 원본 사용
                                    decoded_parts.append(part)
                            except Exception:
                                decoded_parts.append(part)
                        else:
                            # 정상적인 문자열인 경우 그대로 사용
                            decoded_parts.append(part)
                    else:
                        decoded_parts.append(str(part))
            
            query = " ".join(decoded_parts)
            
            # 최종 검증: 질의가 유효한지 확인 (공통 검증 함수 사용)
            query = _validate_and_fix_query(query, default_query)
            
            if query == default_query:
                test_logger.warning(f"\n⚠️  질의가 깨진 것으로 보입니다. 기본 질의를 사용합니다.")
                test_logger.warning(f"   깨진 질의: {query[:100]}...")
            else:
                # 질의 정규화 (공백 제거 등)
                query = query.strip()
                test_logger.info(f"\n💡 명령줄에서 질의를 받았습니다.")
    
    if query is None:
        query = default_query
        test_logger.info(f"\n💡 기본 질의를 사용합니다.")
        test_logger.info(f"   사용 가능한 기본 질의: 0='{default_queries[0]}', 1='{default_queries[1]}', 2='{default_queries[2]}'")
        test_logger.info(f"\n   다른 입력 방법:")
        test_logger.info(f"   - 환경 변수: $env:TEST_QUERY='질의내용'; python run_single_query_test.py")
        test_logger.info(f"   - 파일 입력: python run_single_query_test.py -f query.txt")
        test_logger.info(f"   - 숫자 선택: python run_single_query_test.py 0")
        test_logger.info(f"   사용법: python run_single_query_test.py 0  (또는 직접 질의 입력)")
    
    # 비동기 실행
    try:
        # 환경 변수로 스트리밍 모드 제어
        use_streaming = os.getenv("TEST_USE_STREAMING", "true").lower() == "true"
        
        if use_streaming:
            result = asyncio.run(run_single_query_test_streaming(query))
        else:
            result = asyncio.run(run_single_query_test(query))
        return 0
    except KeyboardInterrupt:
        test_logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        test_logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


