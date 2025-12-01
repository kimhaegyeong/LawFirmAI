# -*- coding: utf-8 -*-
"""
LangGraph 질의 테스트 스크립트 (astream_events 사용)

Usage:
    python lawfirm_langgraph/tests/runners/run_query_test_astream_events.py "질의 내용"
    python lawfirm_langgraph/tests/runners/run_query_test_astream_events.py  # 기본 질의 사용
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

# run_query_test.py의 모든 유틸리티 함수들을 import
# (LineBufferedFileHandler, SafeStreamHandler, Tee, setup_logging 등)
# 파일이 너무 길어서 필요한 부분만 복사하거나 import 사용
# 여기서는 간단하게 run_query_test의 함수들을 재사용

# run_query_test.py의 setup_logging과 다른 유틸리티 함수들을 import
import importlib.util
run_query_test_path = script_dir / "run_query_test.py"
spec = importlib.util.spec_from_file_location("run_query_test", run_query_test_path)
run_query_test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_query_test_module)

# 필요한 함수들 가져오기
setup_logging = run_query_test_module.setup_logging
get_query_from_args = run_query_test_module.get_query_from_args
flush_all_log_handlers = run_query_test_module.flush_all_log_handlers


async def test_langgraph_query_astream_events(query: str, logger: logging.Logger):
    """LangGraph 질의 테스트 실행 (astream_events 사용)
    
    Args:
        query: 테스트할 질의
        logger: 로거
    """
    logger.info("=" * 80)
    logger.info("LangGraph 질의 테스트 (astream_events 사용)")
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
            raise ValueError("SQLite is no longer supported. Please configure PostgreSQL.")
        
        logger.info(f"   ✅ Database URL 설정됨 (PostgreSQL)")
        logger.info(f"   VECTOR_SEARCH_METHOD: {app_config.vector_search_method}")
        
        # DatabaseAdapter 확인
        try:
            from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
            if app_config.database_url:
                db_adapter = DatabaseAdapter(app_config.database_url)
                logger.info(f"   ✅ DatabaseAdapter 초기화 성공: type={db_adapter.db_type}")
        except Exception as e:
            logger.error(f"   ❌ DatabaseAdapter 초기화 실패: {e}")
            raise
        
        db_check_time = time.time() - db_check_start
        logger.info(f"   데이터베이스 확인 시간: {db_check_time:.3f}초")
        
        # 서비스 초기화
        logger.info("\n2. LangGraphWorkflowService 초기화 중...")
        service_start = time.time()
        
        try:
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
            
            service = LangGraphWorkflowService(config)
            service_time = time.time() - service_start
            logger.info(f"   서비스 초기화 완료 (총 시간: {service_time:.3f}초)")
            
            init_total_time = time.time() - total_start_time
            logger.info(f"\n초기화 완료 (총 시간: {init_total_time:.3f}초)")
                
        except Exception as e:
            logger.error(f"   ❌ 서비스 초기화 실패: {type(e).__name__}: {e}")
            logger.debug("상세 스택 트레이스:", exc_info=True)
            raise
        
        # 질의 처리 (astream_events 사용)
        logger.info("\n3. 질의 처리 중 (astream_events 사용)...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        query_start_time = time.time()
        
        try:
            logger.info("   🔄 process_query 실행 시작 (use_astream_events=True)...")
            
            # 🔥 CRITICAL: astream_events 사용 명시
            result = await service.process_query(
                query=query,
                session_id="test_langgraph_query_astream_events",
                enable_checkpoint=False,
                use_astream_events=True  # astream_events 사용
            )
            
            logger.info("   ✅ process_query 실행 완료")
                
        except Exception as query_error:
            logger.error(f"   ❌ 질의 처리 중 오류 발생: {type(query_error).__name__}: {query_error}")
            logger.debug("   상세 스택 트레이스:", exc_info=True)
            raise
        
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
        
        # 답변 추출
        answer = result.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("answer", "") or answer.get("content", "") or ""
        answer = str(answer).strip() if answer else ""
        
        if not answer:
            output = result.get("output", {})
            if isinstance(output, dict):
                answer = output.get("answer", "") or output.get("content", "")
        answer = str(answer).strip() if answer else ""
        
        if answer:
            logger.info(f"\n답변 ({len(answer)}자):")
            logger.info("-" * 80)
            logger.info(answer)
            flush_all_log_handlers()
        else:
            logger.warning("\n답변이 없습니다!")
            flush_all_log_handlers()
        
        # 검색 결과 (type 정보 확인)
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            logger.info(f"\n검색된 참고자료 ({len(retrieved_docs)}개):")
            
            # type 정보 통계
            type_stats = {}
            unknown_count = 0
            
            for i, doc in enumerate(retrieved_docs[:10], 1):
                if isinstance(doc, dict):
                    # type 정보 추출
                    doc_type = (
                        doc.get("type") or 
                        doc.get("source_type") or 
                        (doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None) or
                        (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None) or
                        "unknown"
                    )
                    
                    # 통계 수집
                    if doc_type == "unknown":
                        unknown_count += 1
                    type_stats[doc_type] = type_stats.get(doc_type, 0) + 1
                    
                    # 타입 이름 변환
                    type_names = {
                        "statute_article": "법령",
                        "precedent_content": "판례",
                        "unknown": "알 수 없음"
                    }
                    doc_type_display = type_names.get(doc_type, doc_type)
                    
                    # 제목 추출
                    title = (
                        doc.get("title") or 
                        doc.get("name") or 
                        doc.get("source") or
                        (doc.get("content", "")[:100] if doc.get("content") else "") or
                        (doc.get("text", "")[:100] if doc.get("text") else "") or
                        "제목 없음"
                    )
                    
                    # 점수 추출
                    score = (
                        doc.get("relevance_score") or 
                        doc.get("final_weighted_score") or
                        doc.get("score") or 
                        doc.get("similarity") or 
                        0.0
                    )
                    score_display = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                    
                    logger.info(f"   {i}. [{doc_type_display}] {title}")
                    logger.info(f"       점수: {score_display}, type={doc_type}")
                    
                    # type이 unknown인 경우 상세 정보 로깅
                    if doc_type == "unknown":
                        logger.warning(f"       ⚠️  type=unknown 감지!")
                        logger.debug(f"       - doc.type: {doc.get('type')}")
                        logger.debug(f"       - doc.source_type: {doc.get('source_type')}")
                        logger.debug(f"       - metadata.type: {doc.get('metadata', {}).get('type') if isinstance(doc.get('metadata'), dict) else 'N/A'}")
                        logger.debug(f"       - metadata.source_type: {doc.get('metadata', {}).get('source_type') if isinstance(doc.get('metadata'), dict) else 'N/A'}")
                        logger.debug(f"       - doc keys: {list(doc.keys())[:20]}")
                        logger.debug(f"       - metadata keys: {list(doc.get('metadata', {}).keys())[:20] if isinstance(doc.get('metadata'), dict) else 'N/A'}")
                else:
                    logger.info(f"   {i}. {str(doc)[:100]}")
            
            if len(retrieved_docs) > 10:
                logger.info(f"   ... (총 {len(retrieved_docs)}개)")
            
            # type 통계 출력
            logger.info(f"\n📊 Type 통계:")
            for doc_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
                type_names = {
                    "statute_article": "법령",
                    "precedent_content": "판례",
                    "unknown": "알 수 없음"
                }
                doc_type_display = type_names.get(doc_type, doc_type)
                logger.info(f"   - {doc_type_display}: {count}개")
            
            if unknown_count > 0:
                logger.warning(f"\n⚠️  type=unknown인 문서가 {unknown_count}개 발견되었습니다!")
            else:
                logger.info(f"\n✅ 모든 문서의 type 정보가 정상적으로 설정되었습니다!")
            
            flush_all_log_handlers()
        else:
            logger.warning("\n검색된 참고자료가 없습니다!")
            flush_all_log_handlers()
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n처리 시간 (결과): {processing_time:.2f}초")
        if 'query_elapsed_time' in locals():
            logger.info(f"처리 시간 (측정): {query_elapsed_time:.2f}초")
        flush_all_log_handlers()
        
        # 오류 확인
        errors = result.get("errors", [])
        if errors:
            logger.warning(f"\n오류 발생 ({len(errors)}개):")
            for i, error in enumerate(errors[:5], 1):
                logger.warning(f"   {i}. {error}")
            flush_all_log_handlers()
        
        # 결과 요약
        logger.info("\n5. 결과 요약:")
        logger.info("=" * 80)
        
        summary = {
            "질의": query,
            "답변 길이": len(answer) if answer else 0,
            "검색된 문서 수": len(retrieved_docs) if retrieved_docs else 0,
            "type=unknown 문서 수": unknown_count if 'unknown_count' in locals() else 0,
            "처리 시간": f"{processing_time:.2f}초" if processing_time else "N/A",
            "오류 수": len(errors) if errors else 0
        }
        
        logger.info("   요약 정보:")
        for key, value in summary.items():
            logger.info(f"   - {key}: {value}")
        
        flush_all_log_handlers()
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 완료!")
        logger.info("=" * 80)
        
        flush_all_log_handlers()
        
        # 리소스 정리
        try:
            if hasattr(service, 'cleanup'):
                service.cleanup()
        except Exception as e:
            logger.debug(f"리소스 정리 중 오류 (무시): {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"\n오류 발생: {type(e).__name__}: {e}")
        logger.debug("상세 스택 트레이스:", exc_info=True)
        flush_all_log_handlers()
        raise


def main():
    """메인 실행 함수"""
    logger = None
    log_file_path = None
    
    try:
        # 로그 파일 경로 결정
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
            log_file_path = str(log_dir / f"test_astream_events_{timestamp}.log")
        
        # 로깅 설정
        logger = setup_logging(log_file_path=log_file_path)
        
        if logger:
            logger.info("=" * 80)
            logger.info("테스트 시작 (astream_events 사용)")
            logger.info("=" * 80)
            flush_all_log_handlers()
        
        # 질의 가져오기
        query = get_query_from_args()
        
        if not query:
            if logger:
                logger.error("질의를 입력해주세요.")
                logger.info("\n사용법:")
                logger.info("  python run_query_test_astream_events.py \"질의 내용\"")
                logger.info("  python run_query_test_astream_events.py 0  # 기본 질의 선택")
                logger.info("  $env:TEST_QUERY='질의내용'; python run_query_test_astream_events.py")
            return 1
        
        # 테스트 실행
        flush_all_log_handlers()
        
        try:
            asyncio.run(test_langgraph_query_astream_events(query, logger))
        except KeyboardInterrupt:
            flush_all_log_handlers()
            if logger:
                logger.warning("\n\n사용자에 의해 중단되었습니다.")
            raise
        except Exception as async_error:
            flush_all_log_handlers()
            if logger:
                logger.error(f"\n\n비동기 작업 중 오류 발생: {type(async_error).__name__}: {async_error}")
                logger.debug("   전체 스택 트레이스:", exc_info=True)
            flush_all_log_handlers()
            raise
        finally:
            flush_all_log_handlers()
        
        # 테스트 완료 후 로그 파일 경로 출력
        if log_file_path:
            print(f"\n[테스트 완료]")
            print(f"  로그 파일: {log_file_path}")
            print(f"  로그 파일을 확인하여 type 정보 보존 여부를 검증하세요.")
        
        flush_all_log_handlers()
        
        return 0
        
    except KeyboardInterrupt:
        if logger:
            logger.warning("\n\n사용자에 의해 중단되었습니다.")
        flush_all_log_handlers()
        return 1
    except Exception as e:
        if logger:
            logger.error(f"\n\n테스트 실패: {e}")
            logger.debug("상세 스택 트레이스:", exc_info=True)
        flush_all_log_handlers()
        return 1
    finally:
        flush_all_log_handlers()


if __name__ == "__main__":
    sys.exit(main())

