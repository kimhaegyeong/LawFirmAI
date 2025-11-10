# -*- coding: utf-8 -*-
"""
LangGraph 질의 테스트 스크립트 (개선 버전)

Usage:
    python lawfirm_langgraph/tests/scripts/run_query_test.py "질의 내용"
    python lawfirm_langgraph/tests/scripts/run_query_test.py 0  # 기본 질의 선택
    $env:TEST_QUERY='질의내용'; python run_query_test.py  # 환경 변수 사용
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# UTF-8 인코딩 설정 (Windows PowerShell 호환)
_original_stdout = sys.stdout
_original_stderr = sys.stderr

if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    if hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 경로 설정
# 스크립트 위치: lawfirm_langgraph/tests/scripts/run_query_test.py
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

# sys.path 설정 (순환 import 방지)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 로깅 설정 (SafeStreamHandler 사용)
def setup_logging(log_level: str = "INFO"):
    """로깅 설정 (Windows PowerShell 호환)"""
    logger = logging.getLogger("lawfirm_langgraph.tests")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    
    # SafeStreamHandler 클래스 정의
    class SafeStreamHandler(logging.StreamHandler):
        """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
        
        def __init__(self, stream, original_stdout_ref=None):
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
                        # 버퍼 분리 오류 방지를 위한 추가 검증
                        if hasattr(safe_stream, 'buffer'):
                            try:
                                buffer = safe_stream.buffer
                                if buffer is None:
                                    raise ValueError("Buffer is None")
                            except (ValueError, AttributeError):
                                # buffer가 분리된 경우, 직접 write 시도
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
                        # 버퍼 분리 오류인 경우 무시하고 계속 진행
                        if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                            pass
                        else:
                            pass
                
                # Fallback: stderr 사용
                try:
                    if sys.stderr and hasattr(sys.stderr, 'write'):
                        # stderr도 버퍼 분리 오류가 있을 수 있으므로 안전하게 처리
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
                # 모든 예외 무시 (로깅 실패가 전체 프로그램을 중단시키지 않도록)
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
    
    # 콘솔 핸들러 생성
    try:
        base_handler = logging.StreamHandler(_original_stdout)
    except (ValueError, AttributeError):
        try:
            base_handler = logging.StreamHandler(sys.stdout)
        except (ValueError, AttributeError):
            base_handler = logging.StreamHandler(sys.stderr)
    
    # SafeStreamHandler로 교체
    safe_handler = SafeStreamHandler(base_handler.stream, _original_stdout)
    safe_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    safe_handler.setFormatter(formatter)
    logger.addHandler(safe_handler)
    
    return logger

logger = setup_logging(os.getenv("TEST_LOG_LEVEL", "INFO"))


def get_query_from_args() -> str:
    """명령줄 인자에서 질의 추출"""
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    
    # 1. 환경 변수
    test_query = os.getenv('TEST_QUERY')
    if test_query and test_query.strip():
        return test_query.strip()
    
    # 2. 명령줄 인자
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        # 파일 옵션
        if arg in ['-f', '--file']:
            if len(sys.argv) > 2:
                file_path = sys.argv[2]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    logger.error(f"파일 읽기 오류: {e}")
                    return default_queries[1]
            else:
                logger.error("파일 경로를 지정해주세요")
                return default_queries[1]
        
        # 숫자 선택
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                return default_queries[idx]
        
        # 직접 질의
        return " ".join(sys.argv[1:])
    
    # 기본 질의
    return default_queries[1]


async def run_query_test(query: str):
    """질의 테스트 실행"""
    logger.info("\n" + "="*80)
    logger.info("LangGraph 질의 테스트")
    logger.info("="*80)
    logger.info(f"\n📋 질의: {query}\n")
    
    try:
        # Import (순환 import 방지를 위해 함수 내부에서 수행)
        # sys.path가 올바르게 설정되어 있으므로 직접 import 가능
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            # Fallback: 상대 경로
            sys.path.insert(0, str(lawfirm_langgraph_dir))
            from config.langgraph_config import LangGraphConfig
        
        try:
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        except ImportError:
            # Fallback: 상대 경로
            sys.path.insert(0, str(lawfirm_langgraph_dir))
            from core.workflow.workflow_service import LangGraphWorkflowService
        
        # 설정 로드
        logger.info("1️⃣  설정 로드 중...")
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False  # 테스트 모드
        logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   ✅ 체크포인트: {config.enable_checkpoint}")
        
        # 서비스 초기화
        logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        logger.info("   ✅ 서비스 초기화 완료")
        
        # 질의 처리
        logger.info("\n3️⃣  질의 처리 중...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="query_test",
            enable_checkpoint=False
        )
        
        # 결과 출력
        logger.info("\n4️⃣  결과:")
        logger.info("="*80)
        
        # 답변
        answer = result.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("content", answer.get("text", str(answer)))
        
        if answer:
            logger.info(f"\n📝 답변 ({len(str(answer))}자):")
            logger.info("-" * 80)
            logger.info(str(answer))
        else:
            logger.warning("<답변 없음>")
        
        # 소스
        sources = result.get("sources", [])
        if sources:
            logger.info(f"\n📚 소스 ({len(sources)}개):")
            for i, source in enumerate(sources[:5], 1):
                logger.info(f"   {i}. {source}")
            if len(sources) > 5:
                logger.info(f"   ... (총 {len(sources)}개)")
        
        # 법률 참조
        legal_references = result.get("legal_references", [])
        if legal_references:
            logger.info(f"\n⚖️  법률 참조 ({len(legal_references)}개):")
            for i, ref in enumerate(legal_references[:5], 1):
                logger.info(f"   {i}. {ref}")
            if len(legal_references) > 5:
                logger.info(f"   ... (총 {len(legal_references)}개)")
        
        # 메타데이터
        metadata = result.get("metadata", {})
        if metadata:
            logger.info(f"\n📊 메타데이터:")
            for key, value in list(metadata.items())[:10]:
                logger.info(f"   {key}: {value}")
        
        # 신뢰도
        confidence = result.get("confidence", 0.0)
        if confidence:
            logger.info(f"\n🎯 신뢰도: {confidence:.2f}")
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
        logger.info("\n" + "="*80)
        logger.info("✅ 테스트 완료!")
        logger.info("="*80)
        
        return result
        
    except ImportError as e:
        logger.error(f"\n❌ Import 오류: {e}")
        logger.error("\n필요한 패키지가 설치되어 있는지 확인하세요.")
        logger.error(f"   프로젝트 루트: {project_root}")
        logger.error(f"   lawfirm_langgraph 디렉토리: {lawfirm_langgraph_dir}")
        raise
        
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {type(e).__name__}: {e}", exc_info=True)
        raise


def main():
    """메인 실행 함수"""
    try:
        query = get_query_from_args()
        
        if not query:
            logger.error("질의를 입력해주세요.")
            logger.info("\n사용법:")
            logger.info("  python run_query_test.py \"질의 내용\"")
            logger.info("  python run_query_test.py 0  # 기본 질의 선택")
            logger.info("  $env:TEST_QUERY='질의내용'; python run_query_test.py")
            return 1
        
        result = asyncio.run(run_query_test(query))
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

