# -*- coding: utf-8 -*-
"""
LangGraph 동작 테스트 스크립트
리팩토링 후 LangGraph가 정상적으로 동작하는지 확인
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
# Windows 비동기 환경에서 로깅 버퍼 에러 방지
class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 - detached 버퍼 에러 방지"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            # detached buffer 에러나 기타 스트림 에러 무시
            pass

# 로깅 설정 (INFO 레벨로 변경하여 중요한 정보만 확인)
# DEBUG 레벨은 너무 많은 로그를 생성하므로 INFO로 조정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[SafeStreamHandler()],
    force=True  # 기존 설정을 강제로 재설정
)

# 특정 로거의 레벨을 INFO로 설정 (DEBUG는 선택적)
legal_workflow_logger = logging.getLogger('core.agents.legal_workflow_enhanced')
legal_workflow_logger.setLevel(logging.INFO)  # DEBUG에서 INFO로 변경
legal_workflow_logger.propagate = True

# 핸들러가 없으면 추가
if not legal_workflow_logger.handlers:
    legal_workflow_logger.addHandler(SafeStreamHandler())

# 다른 로거들도 동일하게 설정 (INFO 레벨)
semantic_search_logger = logging.getLogger('source.services.semantic_search_engine_v2')
semantic_search_logger.setLevel(logging.INFO)  # DEBUG에서 INFO로 변경
semantic_search_logger.propagate = True
if not semantic_search_logger.handlers:
    semantic_search_logger.addHandler(SafeStreamHandler())

data_connector_logger = logging.getLogger('core.agents.legal_data_connector_v2')
data_connector_logger.setLevel(logging.INFO)  # DEBUG에서 INFO로 변경
data_connector_logger.propagate = True
if not data_connector_logger.handlers:
    data_connector_logger.addHandler(SafeStreamHandler())

# workflow_service의 DEBUG 로그는 WARNING 레벨로 조정
workflow_service_logger = logging.getLogger('core.agents.workflow_service')
workflow_service_logger.setLevel(logging.WARNING)  # DEBUG 로그 억제

# 로깅 에러를 억제
logging.raiseExceptions = False

logger = logging.getLogger(__name__)

# pytest-asyncio 지원 (선택적)
try:
    import pytest
    pytest_available = True
except ImportError:
    pytest_available = False


async def test_langgraph_workflow():
    """LangGraph 워크플로우 테스트"""
    try:
        # core/agents/legal_workflow_enhanced.py를 사용하도록 변경
        from core.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("LangGraph 워크플로우 테스트 시작")
        logger.info("=" * 80)

        # 설정 로드
        logger.info("1. LangGraph 설정 로드 중...")
        config = LangGraphConfig.from_env()
        logger.info(f"   ✅ LangGraph 설정 로드 완료 (enabled={config.langgraph_enabled})")

        # 워크플로우 서비스 초기화
        logger.info("2. 워크플로우 서비스 초기화 중...")
        start_time = time.time()
        workflow_service = LangGraphWorkflowService(config)
        init_time = time.time() - start_time
        logger.info(f"   ✅ 워크플로우 서비스 초기화 완료 ({init_time:.2f}초)")

        # 테스트 질의 (다양한 법률 분야, 5개)
        test_queries = [
            "민사법에서 계약 해지 요건은 무엇인가요?",
            "형법에서 절도죄의 성립 요건과 처벌은 어떻게 되나요?",
            "가족법에서 협의이혼과 재판상 이혼의 차이는 무엇인가요?",
            "노동법에서 근로계약서 작성 시 포함해야 할 필수 사항은 무엇인가요?",
            "행정법에서 행정처분 취소 청구의 제소기간은 언제까지인가요?"
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

                # 결과 검증 (중첩 딕셔너리 안전하게 추출)
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

                # 최종 문자열 보장
                answer = str(answer) if not isinstance(answer, str) else answer
                has_answer = bool(answer) and len(answer) > 0

                # Sources 확인 로직 개선 (V2 최적화)
                sources = result.get("sources", []) if isinstance(result, dict) else []
                retrieved_docs = result.get("retrieved_docs", []) if isinstance(result, dict) else []

                # Sources 확인: 직접 sources 필드 또는 retrieved_docs에서 추출
                has_sources = False
                sources_count = 0
                sources_list = []

                if isinstance(sources, list) and len(sources) > 0:
                    # sources 필드가 있고 비어있지 않은 경우
                    has_sources = True
                    sources_count = len(sources)
                    sources_list = sources[:5]  # 상위 5개만
                elif isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                    # retrieved_docs에서 sources 추출 시도
                    extracted_sources = []
                    for doc in retrieved_docs:
                        if isinstance(doc, dict):
                            # 다양한 필드에서 source 추출 시도
                            source = (
                                doc.get("source") or
                                doc.get("source_name") or
                                doc.get("statute_name") or
                                None
                            )

                            # metadata에서도 추출 시도
                            if not source:
                                metadata = doc.get("metadata", {})
                                if isinstance(metadata, dict):
                                    source = (
                                        metadata.get("statute_name") or
                                        metadata.get("statute_abbrv") or
                                        metadata.get("court") or
                                        metadata.get("org") or
                                        None
                                    )

                            if source and source not in extracted_sources:
                                extracted_sources.append(source)

                    if len(extracted_sources) > 0:
                        has_sources = True
                        sources_count = len(extracted_sources)
                        sources_list = extracted_sources[:5]

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
                logger.info(f"   - 소스 유무: {'있음' if has_sources else '없음'}")
                if has_sources:
                    logger.info(f"   - 소스 개수: {sources_count}개")
                    if sources_list:
                        logger.info(f"   - 주요 소스: {', '.join(str(s) for s in sources_list)}")
                else:
                    logger.warning(f"   - 소스 없음 (retrieved_docs: {len(retrieved_docs)}개)")
                logger.info(f"   - 신뢰도: {confidence:.2%}")
                logger.info(f"   - 에러 유무: {'있음' if has_errors else '없음'}")

                # stdout에도 출력 (버퍼링 방지)
                print(f"\n{result_status} 질의 {i}/{len(test_queries)}: {query} (처리 시간: {processing_time:.2f}초)", flush=True)
                print(f"   - 답변 유무: {'있음' if has_answer else '없음'}", flush=True)
                print(f"   - 답변 길이: {answer_length}자", flush=True)
                print(f"   - 소스 유무: {'있음' if has_sources else '없음'}", flush=True)
                if has_sources:
                    print(f"   - 소스 개수: {sources_count}개", flush=True)
                    if sources_list:
                        print(f"   - 주요 소스: {', '.join(str(s) for s in sources_list)}", flush=True)
                else:
                    print(f"   - 소스 없음 (retrieved_docs: {len(retrieved_docs)}개)", flush=True)
                print(f"   - 신뢰도: {confidence:.2%}", flush=True)
                print(f"   - 에러 유무: {'있음' if has_errors else '없음'}", flush=True)

                if has_answer:
                    # 분리된 메타 정보 필드 확인
                    confidence_info = result.get("confidence_info", "") if isinstance(result, dict) else ""
                    reference_materials = result.get("reference_materials", "") if isinstance(result, dict) else ""
                    disclaimer = result.get("disclaimer", "") if isinstance(result, dict) else ""

                    logger.info(f"\n📝 답변 미리보기:")
                    # answer가 문자열인지 확인
                    if isinstance(answer, str):
                        answer_preview = answer[:500]
                        logger.info(f"   {answer_preview}{'...' if len(answer) > 500 else ''}")
                        print(f"\n📝 답변 전체 내용 (answer 필드만):", flush=True)
                        print(f"{'='*80}", flush=True)
                        print(answer, flush=True)
                        print(f"{'='*80}\n", flush=True)

                        # 분리된 메타 정보 표시
                        if confidence_info:
                            print(f"\n💡 신뢰도 정보 (confidence_info 필드):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(confidence_info, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        if reference_materials:
                            print(f"\n📚 참고 자료 (reference_materials 필드):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(reference_materials, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        if disclaimer:
                            print(f"\n💼 면책 조항 (disclaimer 필드):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(disclaimer, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        # 답변을 파일로 저장 (answer 필드만 저장)
                        import os
                        output_dir = "test_outputs"
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"answer_{i}_{int(time.time())}.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"질의: {query}\n")
                            f.write(f"{'='*80}\n")
                            f.write(f"답변 길이: {len(answer)}자\n")
                            f.write(f"신뢰도: {confidence:.2%}\n")
                            f.write(f"{'='*80}\n\n")
                            f.write("=== 답변 내용 (answer 필드) ===\n\n")
                            f.write(answer)

                            # 분리된 메타 정보도 함께 저장
                            if confidence_info:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== 신뢰도 정보 (confidence_info 필드) ===\n\n")
                                f.write(confidence_info)

                            if reference_materials:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== 참고 자료 (reference_materials 필드) ===\n\n")
                                f.write(reference_materials)

                            if disclaimer:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== 면책 조항 (disclaimer 필드) ===\n\n")
                                f.write(disclaimer)

                        logger.info(f"   답변이 {output_file}에 저장되었습니다.")
                        print(f"   답변이 {output_file}에 저장되었습니다.\n", flush=True)
                    else:
                        logger.info(f"   답변 타입: {type(answer).__name__}, 값: {answer}")

                if has_errors:
                    logger.warning(f"\n⚠️ 에러 목록:")
                    print(f"   ⚠️ 에러 목록:", flush=True)
                    error_list = errors if isinstance(errors, list) else []
                    for error in error_list[:5]:
                        logger.warning(f"   - {error}")
                        print(f"     - {error}", flush=True)

                if not is_success:
                    # 실패 원인 상세 분석
                    logger.warning(f"\n⚠️ 질의 실패 원인 분석:")
                    print(f"   ⚠️ 질의 실패 원인 분석:", flush=True)
                    if not has_answer:
                        logger.warning(f"   - 답변이 생성되지 않았습니다")
                        print(f"     - 답변이 생성되지 않았습니다", flush=True)
                        logger.warning(f"     - result.get('answer'): {result.get('answer')}")
                        print(f"       result.get('answer'): {result.get('answer')}", flush=True)
                    if has_errors:
                        logger.warning(f"   - 에러가 발생했습니다: {len(result.get('errors', []))}개")
                        print(f"     - 에러가 발생했습니다: {len(result.get('errors', []))}개", flush=True)

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
                    "sources_list": sources_list,
                    "retrieved_docs_count": len(retrieved_docs) if isinstance(retrieved_docs, list) else 0,
                    "has_errors": has_errors,
                    "errors": errors if isinstance(errors, list) else [],
                    "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                    "result_type": type(result).__name__
                }
                results.append(test_result)

            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()

                logger.error(f"\n❌ 테스트 질의 실패: {query}")
                logger.error(f"오류 유형: {type(e).__name__}")
                logger.error(f"오류 메시지: {str(e)}")
                logger.error(f"상세 스택 트레이스:\n{error_traceback}")

                # stdout에도 출력 (버퍼링 방지)
                print(f"\n❌ 테스트 질의 실패: {query}", flush=True)
                print(f"오류 유형: {type(e).__name__}", flush=True)
                print(f"오류 메시지: {str(e)}", flush=True)
                print(f"상세 스택 트레이스:\n{error_traceback[:500]}...", flush=True)

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

        # stdout에도 출력
        print(f"\n{'='*80}", flush=True)
        print("테스트 결과 요약", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"   총 질의 수: {total_queries}", flush=True)
        print(f"   성공한 질의: {successful_queries}", flush=True)
        print(f"   실패한 질의: {failed_queries}", flush=True)
        print(f"   평균 처리 시간: {avg_time:.2f}초", flush=True)
        print(f"   평균 신뢰도: {avg_confidence:.2%}", flush=True)

        # 실패한 질의 상세 정보
        if failed_queries > 0:
            logger.error(f"\n{'='*80}")
            logger.error("실패한 질의 상세 정보")
            logger.error(f"{'='*80}")
            print(f"\n{'='*80}", flush=True)
            print("실패한 질의 상세 정보", flush=True)
            print(f"{'='*80}", flush=True)

            for i, result in enumerate(results, 1):
                if not result.get("success", False):
                    logger.error(f"\n[{i}] 질의: {result.get('query', '알 수 없음')}")
                    print(f"\n[{i}] 질의: {result.get('query', '알 수 없음')}", flush=True)

                    # 답변 상태
                    if "has_answer" in result:
                        logger.error(f"   - 답변 생성: {'예' if result.get('has_answer') else '아니오'}")
                        print(f"   - 답변 생성: {'예' if result.get('has_answer') else '아니오'}", flush=True)

                    # Sources 상태
                    if "has_sources" in result:
                        sources_count = result.get('sources_count', 0)
                        sources_list = result.get('sources_list', [])
                        logger.error(f"   - 소스: {'있음' if result.get('has_sources') else '없음'} ({sources_count}개)")
                        print(f"   - 소스: {'있음' if result.get('has_sources') else '없음'} ({sources_count}개)", flush=True)
                        if sources_list:
                            logger.error(f"   - 소스 목록: {', '.join(str(s) for s in sources_list)}")
                            print(f"   - 소스 목록: {', '.join(str(s) for s in sources_list)}", flush=True)
                        retrieved_count = result.get('retrieved_docs_count', 0)
                        if retrieved_count > 0:
                            logger.error(f"   - 검색된 문서: {retrieved_count}개")
                            print(f"   - 검색된 문서: {retrieved_count}개", flush=True)

                    # 에러 상태
                    if result.get("has_errors"):
                        logger.error(f"   - 에러 발생: 예")
                        logger.error(f"   - 에러 목록: {result.get('errors', [])}")
                        print(f"   - 에러 발생: 예", flush=True)
                        print(f"   - 에러 목록: {result.get('errors', [])}", flush=True)

                    # 예외 발생
                    if "error" in result:
                        logger.error(f"   - 예외 발생: {result.get('error_type', 'Unknown')}")
                        logger.error(f"   - 오류 메시지: {result.get('error', '알 수 없음')}")
                        print(f"   - 예외 발생: {result.get('error_type', 'Unknown')}", flush=True)
                        print(f"   - 오류 메시지: {result.get('error', '알 수 없음')}", flush=True)

                        # 스택 트레이스 출력 (일부만)
                        if "traceback" in result:
                            traceback_lines = result["traceback"].split('\n')
                            logger.error(f"   - 스택 트레이스 (최근 5줄):")
                            print(f"   - 스택 트레이스 (최근 5줄):", flush=True)
                            for line in traceback_lines[-5:]:
                                if line.strip():
                                    logger.error(f"     {line}")
                                    print(f"     {line}", flush=True)

                    # 처리 시간
                    if "processing_time" in result:
                        logger.error(f"   - 처리 시간: {result.get('processing_time', 0):.2f}초")
                        print(f"   - 처리 시간: {result.get('processing_time', 0):.2f}초", flush=True)

        # 서비스 상태 확인
        logger.info("\n4. 서비스 상태 확인 중...")
        status = workflow_service.get_service_status()
        logger.info(f"   서비스 상태: {status.get('status')}")
        logger.info(f"   워크플로우 컴파일 여부: {status.get('workflow_compiled')}")

        # 최종 판정
        logger.info(f"\n{'='*80}")
        print(f"\n{'='*80}", flush=True)

        if successful_queries == total_queries:
            logger.info("✅ 모든 테스트 통과! LangGraph가 정상적으로 동작합니다.")
            print("✅ 모든 테스트 통과! LangGraph가 정상적으로 동작합니다.", flush=True)
        elif successful_queries > 0:
            logger.info(f"⚠️ 부분 성공: {successful_queries}/{total_queries} 질의 성공")
            print(f"⚠️ 부분 성공: {successful_queries}/{total_queries} 질의 성공", flush=True)

            # 부분 성공 원인 분석
            logger.warning(f"\n부분 성공 분석:")
            print(f"\n부분 성공 분석:", flush=True)
            for i, result in enumerate(results, 1):
                if not result.get("success"):
                    logger.warning(f"  질의 {i}: {result.get('query')}")
                    logger.warning(f"    - 답변: {'있음' if result.get('has_answer') else '없음'}")
                    logger.warning(f"    - 소스: {'있음' if result.get('has_sources') else '없음'} ({result.get('sources_count', 0)}개)")
                    logger.warning(f"    - 에러: {'있음' if result.get('has_errors') else '없음'}")
                    print(f"  질의 {i}: {result.get('query')}", flush=True)
                    print(f"    - 답변: {'있음' if result.get('has_answer') else '없음'}", flush=True)
                    print(f"    - 소스: {'있음' if result.get('has_sources') else '없음'} ({result.get('sources_count', 0)}개)", flush=True)
                    print(f"    - 에러: {'있음' if result.get('has_errors') else '없음'}", flush=True)
        else:
            logger.error("❌ 모든 테스트 실패: LangGraph에 문제가 있습니다.")
            print("❌ 모든 테스트 실패: LangGraph에 문제가 있습니다.", flush=True)

            # 전체 실패 원인 분석
            logger.error(f"\n전체 실패 원인 분석:")
            print(f"\n전체 실패 원인 분석:", flush=True)
            for i, result in enumerate(results, 1):
                logger.error(f"  질의 {i}: {result.get('query')}")
                print(f"  질의 {i}: {result.get('query')}", flush=True)

                if "error" in result:
                    logger.error(f"    - 예외: {result.get('error_type')} - {result.get('error')}")
                    print(f"    - 예외: {result.get('error_type')} - {result.get('error')}", flush=True)
                else:
                    logger.error(f"    - 답변: {'있음' if result.get('has_answer') else '없음'}")
                    logger.error(f"    - 에러: {'있음' if result.get('has_errors') else '없음'}")
                    if result.get('errors'):
                        logger.error(f"    - 에러 목록: {result.get('errors')}")
                    print(f"    - 답변: {'있음' if result.get('has_answer') else '없음'}", flush=True)
                    print(f"    - 에러: {'있음' if result.get('has_errors') else '없음'}", flush=True)
                    if result.get('errors'):
                        print(f"    - 에러 목록: {result.get('errors')}", flush=True)

        logger.info(f"{'='*80}\n")
        print(f"{'='*80}\n", flush=True)

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

        # stdout에도 출력
        print(f"\n{'='*80}", flush=True)
        print("테스트 실행 중 치명적 오류 발생", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"오류 유형: {type(e).__name__}", flush=True)
        print(f"오류 메시지: {str(e)}", flush=True)
        print(f"상세 스택 트레이스:\n{error_traceback}", flush=True)

        return False


def main():
    """메인 함수"""
    try:
        # 테스트 실행
        result = asyncio.run(test_langgraph_workflow())

        # 결과 출력 (버퍼링 방지)
        print(f"\n{'='*80}")
        print(f"최종 테스트 결과: {'✅ 성공' if result else '❌ 실패'}")
        print(f"{'='*80}\n")

        # stdout 버퍼 플러시
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        # 종료 코드
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
