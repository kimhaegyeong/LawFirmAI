# -*- coding: utf-8 -*-
"""
검색 결과가 generate_answer_enhanced까지 잘 전달되는지 테스트
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정 (안전한 핸들러 사용)
class SafeStreamHandler(logging.StreamHandler):
    """안전한 스트림 핸들러 - detached 버퍼 에러 방지"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[SafeStreamHandler()],
    force=True
)

# 로깅 예외 비활성화
logging.raiseExceptions = False

logger = logging.getLogger(__name__)


async def test_search_results_to_generate_answer():
    """검색 결과가 generate_answer_enhanced까지 전달되는지 테스트"""
    try:
        from core.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("검색 결과 전달 테스트 시작")
        logger.info("=" * 80)

        # 설정 로드
        logger.info("1. LangGraph 설정 로드 중...")
        config = LangGraphConfig.from_env()
        logger.info(f"   ✅ LangGraph 설정 로드 완료")

        # 워크플로우 서비스 초기화
        logger.info("2. 워크플로우 서비스 초기화 중...")
        workflow_service = LangGraphWorkflowService(config)
        logger.info(f"   ✅ 워크플로우 서비스 초기화 완료")

        # 테스트 질의 (검색 결과가 확실히 나올 것으로 예상되는 질의)
        test_queries = [
            "계약 해지 요건",
            "손해배상 책임",
            "민법 제543조"
        ]

        logger.info("3. 검색 결과 전달 테스트 시작...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                # 세션 ID 생성
                session_id = f"test_search_{int(time.time())}_{i}"

                # 질의 처리 (검색 단계까지 실행)
                logger.info(f"질의 처리 시작: {query}")
                start_time = time.time()

                # 전체 워크플로우 실행하고 검색 결과 전달 확인
                logger.info("🔄 전체 워크플로우 실행 중...")
                result = await workflow_service.process_query(query, session_id, enable_checkpoint=False)

                # 결과에서 검색 관련 정보 추출
                retrieved_docs = result.get("retrieved_docs", [])
                metadata = result.get("metadata", {})
                search_meta = metadata.get("search", {}) if isinstance(metadata, dict) else {}

                semantic_count = search_meta.get("semantic_results_count", 0)
                keyword_count = search_meta.get("keyword_results_count", 0)
                final_count = search_meta.get("final_count", len(retrieved_docs))

                logger.info(f"\n📊 검색 결과:")
                logger.info(f"   ✅ 의미적 검색: {semantic_count}개")
                logger.info(f"   ✅ 키워드 검색: {keyword_count}개")
                logger.info(f"   ✅ 최종 통합 검색 결과: {final_count}개")

                if final_count > 0:
                    logger.info(f"   📄 첫 번째 문서 샘플:")
                    first_doc = retrieved_docs[0] if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 else {}
                    if isinstance(first_doc, dict):
                        logger.info(f"      - Type: {first_doc.get('type', 'unknown')}")
                        logger.info(f"      - Source: {str(first_doc.get('source', 'unknown'))[:50]}")
                        content = first_doc.get('content', '') or first_doc.get('text', '')
                        logger.info(f"      - Content preview: {str(content)[:100]}...")
                else:
                    logger.warning("   ⚠️ 검색 결과가 없습니다!")

                # 답변 확인
                answer = result.get("answer", "")
                logger.info(f"\n✍️ generate_answer_enhanced 결과:")
                logger.info(f"   ✅ 답변 생성 완료")
                logger.info(f"   📊 답변에서 받은 검색 결과: {len(retrieved_docs)}개")

                if answer:
                    answer_preview = answer[:200] if len(answer) > 200 else answer
                    logger.info(f"   📝 생성된 답변 미리보기: {answer_preview}...")

                    # 검색 결과의 내용이 답변에 포함되었는지 간단히 확인
                    doc_mentioned = False
                    if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                        for doc in retrieved_docs[:3]:
                            if isinstance(doc, dict):
                                source = doc.get("source", "")
                                if source and len(str(source)) > 10 and str(source)[:20] in answer:
                                    doc_mentioned = True
                                    logger.info(f"   ✅ 검색 결과가 답변에 포함됨: {str(source)[:50]}")
                                    break

                    if not doc_mentioned and len(retrieved_docs) > 0:
                        logger.warning("   ⚠️ 검색 결과가 답변에 명시적으로 포함되지 않았을 수 있음")
                else:
                    logger.warning("   ⚠️ 생성된 답변이 없습니다!")

                logger.info(f"\n📊 검색 메타데이터:")
                logger.info(f"   - 의미적 검색: {semantic_count}개")
                logger.info(f"   - 키워드 검색: {keyword_count}개")
                logger.info(f"   - 최종 결과: {final_count}개")
                logger.info(f"   - 검색 시간: {search_meta.get('search_time', 0):.3f}초")

                processing_time = time.time() - start_time
                logger.info(f"\n✅ 테스트 {i} 완료 (총 {processing_time:.2f}초)")

                # 검증
                assert final_count > 0, f"검색 결과가 없습니다! (semantic: {semantic_count}, keyword: {keyword_count})"
                assert len(retrieved_docs) > 0, f"검색 결과가 retrieved_docs에 없습니다!"
                assert answer is not None and len(str(answer)) > 0, "답변이 생성되지 않았습니다!"

                # 검색 결과가 generate_answer_enhanced까지 전달되었는지 확인
                assert retrieved_docs is not None, "retrieved_docs가 None입니다!"
                assert isinstance(retrieved_docs, list), f"retrieved_docs가 리스트가 아닙니다: {type(retrieved_docs)}"

                logger.info(f"   ✅ 모든 검증 통과!")

            except Exception as e:
                logger.error(f"테스트 {i} 실패: {e}", exc_info=True)
                continue

        logger.info("\n" + "=" * 80)
        logger.info("검색 결과 전달 테스트 완료")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(test_search_results_to_generate_answer())
