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
    logger.propagate = True
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
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
                    except (ValueError, AttributeError, OSError):
                        # 버퍼 분리 오류 발생 시 대체 방법 시도
                        try:
                            # 포맷된 메시지를 직접 출력 시도
                            msg = self.format(record) + self.terminator
                            # 원본 stdout에 직접 쓰기 시도
                            if self._original_stdout is not None:
                                try:
                                    self._original_stdout.write(msg)
                                    self._original_stdout.flush()
                                except (ValueError, AttributeError, OSError):
                                    # 원본 stdout 실패 시 현재 stdout 시도
                                    try:
                                        sys.stdout.write(msg)
                                        sys.stdout.flush()
                                    except (ValueError, AttributeError, OSError):
                                        # 모든 시도 실패 시 stderr 사용
                                        sys.stderr.write(msg)
                                        sys.stderr.flush()
                            else:
                                # 원본이 없는 경우 현재 stdout 사용
                                try:
                                    sys.stdout.write(msg)
                                    sys.stdout.flush()
                                except (ValueError, AttributeError, OSError):
                                    sys.stderr.write(msg)
                                    sys.stderr.flush()
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
        
        # 환경 변수 질의 검증 및 복구
        query = _validate_and_fix_query(query, default_query)
        
        if query == default_query:
            test_logger.warning(f"\n⚠️  환경 변수 TEST_QUERY의 질의가 깨진 것으로 보입니다. 기본 질의를 사용합니다.")
            test_logger.warning(f"   깨진 질의: {test_query_env[:100]}...")
        else:
            test_logger.info(f"\n💡 환경 변수 TEST_QUERY에서 질의를 읽었습니다.")
            test_logger.info(f"   사용법: $env:TEST_QUERY='민법 제750조 손해배상에 대해 설명해주세요'; python run_single_query_test.py")
            test_logger.info(f"   또는: set TEST_QUERY=민법 제750조 손해배상에 대해 설명해주세요 && python run_single_query_test.py")
    
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


