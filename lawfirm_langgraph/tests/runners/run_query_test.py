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
from pathlib import Path
from datetime import datetime
from typing import Optional

# UTF-8 인코딩 설정 (Windows 호환)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
runners_dir = script_dir.parent
tests_dir = runners_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

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


# 원본 stdout 저장
_original_stdout = sys.stdout

# 로깅 설정
def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """로깅 설정
    
    Args:
        log_level: 로그 레벨 (기본값: 환경 변수 LOG_LEVEL 또는 INFO)
    
    Returns:
        설정된 로거
    """
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
    
    # 로그 디렉토리 생성
    log_dir = project_root / "logs" / "langgraph"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_langgraph_query_{timestamp}.log"
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # 기존 핸들러 제거
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(log_level_value)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
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
    
    # lawfirm_langgraph 로거 설정
    langgraph_logger = logging.getLogger("lawfirm_langgraph")
    langgraph_logger.setLevel(log_level_value)
    langgraph_logger.propagate = True
    
    # 테스트 로거
    logger = logging.getLogger("lawfirm_langgraph.tests.scripts.test_langgraph_query")
    logger.setLevel(log_level_value)
    
    logger.info(f"로그 파일: {log_file}")
    logger.info(f"로그 레벨: {log_level}")
    
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
    logger.info(f"\n질의: {query}\n")
    
    try:
        # 설정 로드
        logger.info("1. 설정 로드 중...")
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        logger.info(f"   LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   체크포인트: {config.enable_checkpoint}")
        
        # 서비스 초기화
        logger.info("\n2. LangGraphWorkflowService 초기화 중...")
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        service = LangGraphWorkflowService(config)
        logger.info("   서비스 초기화 완료")
        
        # 질의 처리
        logger.info("\n3. 질의 처리 중...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="test_langgraph_query",
            enable_checkpoint=False,
            use_astream_events=True
        )
        
        # 결과 출력
        logger.info("\n4. 결과:")
        logger.info("=" * 80)
        
        # 답변
        answer = result.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("answer", "") or answer.get("content", "") or ""
        answer = str(answer).strip() if answer else ""
        
        if answer:
            logger.info(f"\n답변 ({len(answer)}자):")
            logger.info("-" * 80)
            logger.info(answer)
        else:
            logger.warning("\n답변이 없습니다!")
        
        # 검색 결과
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            logger.info(f"\n검색된 참고자료 ({len(retrieved_docs)}개):")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                if isinstance(doc, dict):
                    doc_type = doc.get("type") or doc.get("source_type", "unknown")
                    title = doc.get("title") or doc.get("name") or doc.get("content", "")[:50] or "제목 없음"
                    logger.info(f"   {i}. [{doc_type}] {title}")
                else:
                    logger.info(f"   {i}. {str(doc)[:100]}")
            if len(retrieved_docs) > 5:
                logger.info(f"   ... (총 {len(retrieved_docs)}개)")
        else:
            logger.warning("\n검색된 참고자료가 없습니다!")
        
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
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n처리 시간: {processing_time:.2f}초")
        
        # 오류 확인
        errors = result.get("errors", [])
        if errors:
            logger.warning(f"\n오류 발생 ({len(errors)}개):")
            for i, error in enumerate(errors[:5], 1):
                logger.warning(f"   {i}. {error}")
            if len(errors) > 5:
                logger.warning(f"   ... (총 {len(errors)}개)")
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 완료!")
        logger.info("=" * 80)
        
        return result
        
    except ImportError as e:
        logger.error(f"\nImport 오류: {e}")
        logger.error("필요한 패키지가 설치되어 있는지 확인하세요.")
        raise
    except Exception as e:
        logger.error(f"\n오류 발생: {type(e).__name__}: {e}", exc_info=True)
        raise


def main():
    """메인 실행 함수"""
    try:
        # 로깅 설정
        logger = setup_logging()
        
        # 질의 가져오기
        query = get_query_from_args()
        
        if not query:
            logger.error("질의를 입력해주세요.")
            logger.info("\n사용법:")
            logger.info("  python test_langgraph_query.py \"질의 내용\"")
            logger.info("  python test_langgraph_query.py 0  # 기본 질의 선택")
            logger.info("  $env:TEST_QUERY='질의내용'; python test_langgraph_query.py")
            return 1
        
        # 테스트 실행
        asyncio.run(test_langgraph_query(query, logger))
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"\n\n테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

