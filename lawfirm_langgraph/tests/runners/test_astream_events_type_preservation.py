# -*- coding: utf-8 -*-
"""
astream_events에서 type 정보 보존 테스트

Usage:
    python lawfirm_langgraph/tests/runners/test_astream_events_type_preservation.py "질의 내용"
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

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


class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 (버퍼 분리 오류 방지)"""
    
    def emit(self, record):
        """안전한 로그 출력 (버퍼 분리 오류 방지)"""
        try:
            msg = self.format(record) + self.terminator
            stream = self.stream
            if stream and hasattr(stream, 'write'):
                try:
                    stream.write(msg)
                    if hasattr(stream, 'flush'):
                        stream.flush()
                except (ValueError, AttributeError, OSError):
                    # 버퍼 분리 오류 등은 무시
                    pass
        except Exception:
            # 모든 예외는 무시 (로깅 실패가 프로그램 실패로 이어지지 않도록)
            self.handleError(record)
    
    def flush(self):
        """안전한 flush (오류 무시)"""
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        except (ValueError, AttributeError, OSError):
            pass


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """로깅 설정"""
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    log_level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    log_level_value = log_level_map.get(log_level, logging.INFO)
    
    # 로그 디렉토리 생성
    log_dir_env = os.getenv("TEST_LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env)
    else:
        log_dir = project_root / "logs" / "test"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 경로
    log_file_env = os.getenv("TEST_LOG_FILE")
    if log_file_env:
        log_file = Path(log_file_env)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"test_astream_events_type_{timestamp}.log"
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # 기존 핸들러 제거
    for handler in list(root_logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level_value)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 콘솔 핸들러 (안전한 핸들러 사용)
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(log_level_value)
    console_handler.setFormatter(file_formatter)
    root_logger.addHandler(console_handler)
    
    # 🔥 백그라운드 스레드 로깅 비활성화 (langsmith 등)
    try:
        langsmith_logger = logging.getLogger("langsmith")
        langsmith_logger.setLevel(logging.ERROR)  # ERROR 이상만 표시
        langsmith_logger.propagate = False  # 루트 로거로 전파하지 않음
    except Exception:
        pass
    
    logger = logging.getLogger("test_astream_events_type")
    logger.setLevel(log_level_value)
    
    logger.info(f"로그 파일: {log_file.absolute()}")
    
    return logger


def check_document_type(doc: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
    """문서의 type 정보 확인"""
    result = {
        "index": doc_index,
        "has_type": False,
        "type": None,
        "has_source_type": False,
        "source_type": None,
        "has_metadata_type": False,
        "metadata_type": None,
        "type_hints": {}
    }
    
    if not isinstance(doc, dict):
        return result
    
    # 최상위 레벨 type 확인
    if "type" in doc and doc["type"]:
        result["has_type"] = True
        result["type"] = doc["type"]
    
    # source_type 확인
    if "source_type" in doc and doc["source_type"]:
        result["has_source_type"] = True
        result["source_type"] = doc["source_type"]
    
    # metadata type 확인
    metadata = doc.get("metadata", {})
    if isinstance(metadata, dict):
        if "type" in metadata and metadata["type"]:
            result["has_metadata_type"] = True
            result["metadata_type"] = metadata["type"]
    
    # type hint 필드 확인
    type_hint_fields = [
        "statute_name", "law_name", "article_no", "case_id", "court",
        "doc_id", "casenames", "precedent_id"
    ]
    for field in type_hint_fields:
        if field in doc and doc[field]:
            result["type_hints"][field] = doc[field]
        elif isinstance(metadata, dict) and field in metadata and metadata[field]:
            result["type_hints"][field] = metadata[field]
    
    return result


async def test_astream_events_type_preservation(query: str, logger: logging.Logger):
    """astream_events에서 type 정보 보존 테스트"""
    logger.info("=" * 80)
    logger.info("astream_events Type 정보 보존 테스트")
    logger.info("=" * 80)
    logger.info(f"질의: {query}")
    
    try:
        # 설정 로드
        logger.info("1. 설정 로드 중...")
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.config.app_config import Config as AppConfig
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        app_config = AppConfig()
        logger.info(f"   LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   체크포인트: {config.enable_checkpoint}")
        
        # 서비스 초기화
        logger.info("\n2. LangGraphWorkflowService 초기화 중...")
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        service = LangGraphWorkflowService(config)
        logger.info("   서비스 초기화 완료")
        
        # 질의 처리 (astream_events 사용)
        logger.info("\n3. 질의 처리 중 (astream_events)...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="test_astream_events_type",
            enable_checkpoint=False,
            use_astream_events=True  # 🔥 CRITICAL: astream_events 사용
        )
        
        logger.info("   질의 처리 완료")
        
        # 결과 검증
        logger.info("\n4. Type 정보 검증:")
        logger.info("=" * 80)
        
        # 검색 결과 추출
        retrieved_docs = result.get("retrieved_docs", [])
        if not retrieved_docs:
            # search 그룹에서 확인
            search_group = result.get("search", {})
            if isinstance(search_group, dict):
                retrieved_docs = search_group.get("retrieved_docs", [])
        
        if not retrieved_docs:
            logger.error("   ❌ 검색 결과가 없습니다!")
            return False
        
        logger.info(f"   검색된 문서 수: {len(retrieved_docs)}개")
        
        # 각 문서의 type 정보 확인
        type_check_results = []
        type_missing_count = 0
        type_present_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            check_result = check_document_type(doc, i + 1)
            type_check_results.append(check_result)
            
            if check_result["has_type"] or check_result["has_source_type"] or check_result["has_metadata_type"]:
                type_present_count += 1
                logger.info(f"   ✅ 문서 {i+1}: type={check_result['type'] or check_result['source_type'] or check_result['metadata_type']}")
            else:
                type_missing_count += 1
                logger.warning(f"   ❌ 문서 {i+1}: type 정보 없음")
                logger.warning(f"      - type_hints: {check_result['type_hints']}")
        
        # 통계 출력
        logger.info("\n5. Type 정보 통계:")
        logger.info("=" * 80)
        logger.info(f"   총 문서 수: {len(retrieved_docs)}개")
        logger.info(f"   Type 정보 있음: {type_present_count}개 ({type_present_count/len(retrieved_docs)*100:.1f}%)")
        logger.info(f"   Type 정보 없음: {type_missing_count}개 ({type_missing_count/len(retrieved_docs)*100:.1f}%)")
        
        # Type 분포
        type_distribution = {}
        for check_result in type_check_results:
            doc_type = check_result["type"] or check_result["source_type"] or check_result["metadata_type"]
            if doc_type:
                type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
        
        if type_distribution:
            logger.info(f"\n   Type 분포:")
            for doc_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"      - {doc_type}: {count}개")
        
        # 테스트 결과 판정
        success_rate = type_present_count / len(retrieved_docs) if retrieved_docs else 0
        threshold = 0.8  # 80% 이상이면 성공
        
        logger.info("\n6. 테스트 결과:")
        logger.info("=" * 80)
        if success_rate >= threshold:
            logger.info(f"   ✅ 테스트 통과: {success_rate*100:.1f}% 문서에 type 정보가 있습니다 (임계값: {threshold*100:.0f}%)")
            return True
        else:
            logger.error(f"   ❌ 테스트 실패: {success_rate*100:.1f}% 문서에만 type 정보가 있습니다 (임계값: {threshold*100:.0f}%)")
            return False
        
    except Exception as e:
        logger.error(f"\n오류 발생: {type(e).__name__}: {e}")
        logger.debug("상세 스택 트레이스:", exc_info=True)
        return False


def get_query_from_args() -> str:
    """명령줄 인자에서 질의 추출"""
    default_queries = [
        "손해배상의 범위는 어떻게 결정되나요?",
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
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


def cleanup_logging():
    """로거 정리 (프로그램 종료 전)"""
    try:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)
        
        # 모든 로거의 핸들러 정리
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logger = logging.getLogger(logger_name)
                for handler in list(logger.handlers):
                    try:
                        handler.flush()
                        handler.close()
                    except Exception:
                        pass
                    logger.removeHandler(handler)
            except Exception:
                pass
    except Exception:
        pass


def main():
    """메인 실행 함수"""
    logger = None
    
    try:
        # 로깅 설정
        logger = setup_logging()
        
        # 질의 가져오기
        query = get_query_from_args()
        
        if not query:
            if logger:
                logger.error("질의를 입력해주세요.")
                logger.info("\n사용법:")
                logger.info("  python test_astream_events_type_preservation.py \"질의 내용\"")
                logger.info("  python test_astream_events_type_preservation.py 0  # 기본 질의 선택")
                logger.info("  $env:TEST_QUERY='질의내용'; python test_astream_events_type_preservation.py")
            return 1
        
        # 테스트 실행
        success = asyncio.run(test_astream_events_type_preservation(query, logger))
        
        if success:
            try:
                logger.info("\n" + "=" * 80)
                logger.info("테스트 완료: ✅ 통과")
                logger.info("=" * 80)
            except Exception:
                # 로깅 실패는 무시
                print("\n테스트 완료: ✅ 통과")
            return 0
        else:
            try:
                logger.error("\n" + "=" * 80)
                logger.error("테스트 완료: ❌ 실패")
                logger.error("=" * 80)
            except Exception:
                # 로깅 실패는 무시
                print("\n테스트 완료: ❌ 실패")
            return 1
        
    except KeyboardInterrupt:
        try:
            if logger:
                logger.warning("\n\n사용자에 의해 중단되었습니다.")
        except Exception:
            print("\n\n사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        try:
            if logger:
                logger.error(f"\n\n테스트 실패: {e}")
                logger.debug("상세 스택 트레이스:", exc_info=True)
            else:
                print(f"\n\n테스트 실패: {e}")
                import traceback
                traceback.print_exc()
        except Exception:
            print(f"\n\n테스트 실패: {e}")
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # 🔥 CRITICAL: 프로그램 종료 전 로거 정리
        cleanup_logging()


if __name__ == "__main__":
    sys.exit(main())

