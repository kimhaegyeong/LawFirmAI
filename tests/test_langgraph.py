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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_langgraph_workflow():
    """LangGraph 워크플로우 테스트"""
    try:
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

        # 테스트 질의
        test_queries = [
            "계약서 작성 시 주의사항은?",
            "이혼 소송 절차는 어떻게 되나요?",
            "손해배상 청구권의 성립 요건은?"
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
                has_answer = bool(result.get("answer"))
                has_sources = bool(result.get("sources")) or bool(result.get("retrieved_docs"))
                confidence = result.get("confidence", 0.0)
                has_errors = len(result.get("errors", [])) > 0

                # 결과 출력
                logger.info(f"\n✅ 답변 생성 완료 (처리 시간: {processing_time:.2f}초)")
                logger.info(f"   - 답변 유무: {'있음' if has_answer else '없음'}")
                logger.info(f"   - 답변 길이: {len(result.get('answer', ''))}자")
                logger.info(f"   - 소스 유무: {'있음' if has_sources else '없음'}")
                logger.info(f"   - 신뢰도: {confidence:.2%}")
                logger.info(f"   - 에러 유무: {'있음' if has_errors else '없음'}")

                if has_answer:
                    logger.info(f"\n📝 답변 미리보기:")
                    answer_preview = result.get("answer", "")[:200]
                    logger.info(f"   {answer_preview}{'...' if len(result.get('answer', '')) > 200 else ''}")

                if has_errors:
                    logger.warning(f"\n⚠️ 에러 목록:")
                    for error in result.get("errors", [])[:5]:
                        logger.warning(f"   - {error}")

                # 결과 저장
                test_result = {
                    "query": query,
                    "success": has_answer and not has_errors,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "answer_length": len(result.get("answer", "")),
                    "has_sources": has_sources,
                    "has_errors": has_errors,
                    "errors": result.get("errors", [])
                }
                results.append(test_result)

            except Exception as e:
                logger.error(f"\n❌ 테스트 질의 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })

        # 최종 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("테스트 결과 요약")
        logger.info(f"{'='*80}")

        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("success", False))
        avg_time = sum(r.get("processing_time", 0) for r in results) / total_queries if total_queries > 0 else 0
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total_queries if total_queries > 0 else 0

        logger.info(f"   총 질의 수: {total_queries}")
        logger.info(f"   성공한 질의: {successful_queries}")
        logger.info(f"   실패한 질의: {total_queries - successful_queries}")
        logger.info(f"   평균 처리 시간: {avg_time:.2f}초")
        logger.info(f"   평균 신뢰도: {avg_confidence:.2%}")

        # 서비스 상태 확인
        logger.info("\n4. 서비스 상태 확인 중...")
        status = workflow_service.get_service_status()
        logger.info(f"   서비스 상태: {status.get('status')}")
        logger.info(f"   워크플로우 컴파일 여부: {status.get('workflow_compiled')}")

        # 최종 판정
        logger.info(f"\n{'='*80}")
        if successful_queries == total_queries:
            logger.info("✅ 모든 테스트 통과! LangGraph가 정상적으로 동작합니다.")
        elif successful_queries > 0:
            logger.info(f"⚠️ 부분 성공: {successful_queries}/{total_queries} 질의 성공")
        else:
            logger.info("❌ 모든 테스트 실패: LangGraph에 문제가 있습니다.")
        logger.info(f"{'='*80}\n")

        return successful_queries == total_queries

    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """메인 함수"""
    try:
        # 테스트 실행
        result = asyncio.run(test_langgraph_workflow())

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
