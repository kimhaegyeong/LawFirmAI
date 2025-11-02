# -*- coding: utf-8 -*-
"""
LangGraph 동작 테스트 스크립트 (파일 로깅 포함)
리팩토링 후 LangGraph가 정상적으로 동작하는지 확인하고 상세 로그 저장
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 파일 로거 설정
log_file = log_dir / f"test_langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 루트 로거 설정
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Windows 비동기 환경에서 로깅 버퍼 에러 방지
class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 - detached 버퍼 에러 방지"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            pass

logging.raiseExceptions = False


async def test_langgraph_workflow_with_logging():
    """LangGraph 워크플로우 테스트 (상세 로깅 포함)"""
    try:
        # Import 경로 확인 및 조정
        # core/agents/workflow_service.py를 사용하도록 변경
        from core.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("LangGraph 워크플로우 테스트 시작 (파일 로깅 포함)")
        logger.info("=" * 80)
        logger.info(f"로그 파일: {log_file}")

        # 설정 로드
        logger.info("1. LangGraph 설정 로드 중...")
        config = LangGraphConfig.from_env()
        logger.info(f"   ✅ LangGraph 설정 로드 완료 (enabled={config.langgraph_enabled})")

        # 워크플로우 서비스 초기화 (파일 로깅 활성화)
        logger.info("2. 워크플로우 서비스 초기화 중...")
        start_time = time.time()
        workflow_service = LangGraphWorkflowService(config, enable_file_logging=True)
        init_time = time.time() - start_time
        logger.info(f"   ✅ 워크플로우 서비스 초기화 완료 ({init_time:.2f}초)")

        # 테스트 질의 (민사법 관련, 1개)
        test_queries = [
            "민사법에서 계약 해지 요건은 무엇인가요?"
        ]

        logger.info("3. 테스트 질의 실행 중...")

        results = []
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 질의 {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                # 세션 ID 생성
                session_id = f"test_session_{int(time.time())}_{i}"

                # 질의 처리
                start_time = time.time()
                result = await workflow_service.process_query(query, session_id, enable_checkpoint=False)
                processing_time = time.time() - start_time

                # 결과 검증
                answer = result.get("answer", "") if isinstance(result, dict) else ""

                # 중첩 딕셔너리에서 문자열 추출
                if isinstance(answer, dict):
                    depth = 0
                    max_depth = 20
                    while isinstance(answer, dict) and depth < max_depth:
                        if "answer" in answer:
                            answer = answer["answer"]
                        elif "content" in answer:
                            answer = answer["content"]
                        elif "text" in answer:
                            answer = answer["text"]
                        else:
                            answer = str(answer)
                            break
                        depth += 1
                    if isinstance(answer, dict):
                        answer = str(answer)

                answer = str(answer) if not isinstance(answer, str) else answer
                has_answer = bool(answer) and len(answer) > 0

                # Sources 확인
                sources = result.get("sources", []) if isinstance(result, dict) else []
                retrieved_docs = result.get("retrieved_docs", []) if isinstance(result, dict) else []

                has_sources = len(sources) > 0
                sources_count = len(sources)
                retrieved_docs_count = len(retrieved_docs)

                confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0
                errors = result.get("errors", []) if isinstance(result, dict) else []
                has_errors = len(errors) > 0 if isinstance(errors, list) else False

                # 성공 여부 판정
                is_success = has_answer and not has_errors
                result_status = "✅ 성공" if is_success else "❌ 실패"

                # 결과 출력
                logger.info(f"\n{result_status} 답변 생성 완료 (처리 시간: {processing_time:.2f}초)")
                logger.info(f"   - 답변 유무: {'있음' if has_answer else '없음'}")
                answer_length = len(answer) if isinstance(answer, str) else 0
                logger.info(f"   - 답변 길이: {answer_length}자")
                logger.info(f"   - 소스 유무: {'있음' if has_sources else '없음'} ({sources_count}개)")
                logger.info(f"   - 검색된 문서: {retrieved_docs_count}개")
                logger.info(f"   - 신뢰도: {confidence:.2%}")
                logger.info(f"   - 에러 유무: {'있음' if has_errors else '없음'}")

                # 로그 파일 경로 표시
                log_file_path = result.get("log_file", "")
                if log_file_path:
                    logger.info(f"   - 상세 로그: {log_file_path}")

                # stdout에도 출력
                print(f"\n{result_status} 질의 {i}/{len(test_queries)}: {query} (처리 시간: {processing_time:.2f}초)", flush=True)
                print(f"   - 답변 유무: {'있음' if has_answer else '없음'}", flush=True)
                print(f"   - 답변 길이: {answer_length}자", flush=True)
                print(f"   - 소스 유무: {'있음' if has_sources else '없음'} ({sources_count}개)", flush=True)
                print(f"   - 검색된 문서: {retrieved_docs_count}개", flush=True)
                print(f"   - 신뢰도: {confidence:.2%}", flush=True)
                print(f"   - 에러 유무: {'있음' if has_errors else '없음'}", flush=True)
                if log_file_path:
                    print(f"   - 상세 로그: {log_file_path}", flush=True)

                if has_answer:
                    logger.info(f"\n📝 답변 미리보기:")
                    if isinstance(answer, str):
                        answer_preview = answer[:200]
                        logger.info(f"   {answer_preview}{'...' if len(answer) > 200 else ''}")

                if has_errors:
                    logger.warning(f"\n⚠️ 에러 목록:")
                    error_list = errors if isinstance(errors, list) else []
                    for error in error_list[:5]:
                        logger.warning(f"   - {error}")

                # 결과 저장
                test_result = {
                    "query": query,
                    "success": is_success,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "answer_length": answer_length,
                    "has_answer": has_answer,
                    "has_sources": has_sources,
                    "sources_count": sources_count,
                    "retrieved_docs_count": retrieved_docs_count,
                    "has_errors": has_errors,
                    "errors": errors if isinstance(errors, list) else [],
                    "log_file": log_file_path
                }
                results.append(test_result)

            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()

                logger.error(f"\n❌ 테스트 질의 실패: {query}")
                logger.error(f"오류 유형: {type(e).__name__}")
                logger.error(f"오류 메시지: {str(e)}")
                logger.error(f"상세 스택 트레이스:\n{error_traceback}")

                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_traceback
                })

        # 최종 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("테스트 결과 요약")
        logger.info(f"{'='*80}")

        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("success", False))
        failed_queries = total_queries - successful_queries
        avg_time = sum(r.get("processing_time", 0) for r in results) / total_queries if total_queries > 0 else 0
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total_queries if total_queries > 0 else 0

        logger.info(f"   총 질의 수: {total_queries}")
        logger.info(f"   성공한 질의: {successful_queries}")
        logger.info(f"   실패한 질의: {failed_queries}")
        logger.info(f"   평균 처리 시간: {avg_time:.2f}초")
        logger.info(f"   평균 신뢰도: {avg_confidence:.2%}")
        logger.info(f"\n   상세 로그 파일: {log_file}")

        print(f"\n{'='*80}", flush=True)
        print("테스트 결과 요약", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"   총 질의 수: {total_queries}", flush=True)
        print(f"   성공한 질의: {successful_queries}", flush=True)
        print(f"   실패한 질의: {failed_queries}", flush=True)
        print(f"   평균 처리 시간: {avg_time:.2f}초", flush=True)
        print(f"   평균 신뢰도: {avg_confidence:.2%}", flush=True)
        print(f"\n   상세 로그 파일: {log_file}", flush=True)

        return successful_queries == total_queries

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        logger.error(f"{'='*80}")
        logger.error("테스트 실행 중 치명적 오류 발생")
        logger.error(f"{'='*80}")
        logger.error(f"오류 유형: {type(e).__name__}")
        logger.error(f"오류 메시지: {str(e)}")
        logger.error(f"상세 스택 트레이스:\n{error_traceback}")

        return False


def main():
    """메인 함수"""
    try:
        result = asyncio.run(test_langgraph_workflow_with_logging())

        print(f"\n{'='*80}")
        print(f"최종 테스트 결과: {'✅ 성공' if result else '❌ 실패'}")
        print(f"{'='*80}\n")

        sys.exit(0 if result else 1)

    except KeyboardInterrupt:
        logger.info("\n테스트가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"테스트 실행 중 치명적 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
