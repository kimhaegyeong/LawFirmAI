# -*- coding: utf-8 -*-
"""
검색된 문서 결과가 프롬프트 작성에 제대로 사용되는지 검증하는 테스트
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)


class SearchResultsUsageValidator:
    """검색 결과 사용 여부 검증 클래스"""

    def __init__(self):
        self.verification_results = []

    def verify_search_results_usage(self, state: dict, prompt: str = None) -> dict:
        """
        검색 결과가 프롬프트에 사용되었는지 검증

        Args:
            state: 워크플로우 state
            prompt: 생성된 프롬프트 (선택적)

        Returns:
            검증 결과 딕셔너리
        """
        result = {
            "has_retrieved_docs": False,
            "retrieved_docs_count": 0,
            "retrieved_docs_sources": [],
            "has_structured_documents": False,
            "structured_documents_count": 0,
            "has_context_dict": False,
            "context_dict_has_documents": False,
            "prompt_has_documents": False,
            "prompt_has_sources": False,
            "sources_in_answer": False,
            "verification_score": 0.0,
            "warnings": [],
            "errors": []
        }

        # 1. retrieved_docs 확인
        retrieved_docs = state.get("retrieved_docs", [])
        if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
            result["has_retrieved_docs"] = True
            result["retrieved_docs_count"] = len(retrieved_docs)

            # 소스 추출
            sources = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    source = (
                        doc.get("source") or
                        doc.get("source_name") or
                        doc.get("title") or
                        None
                    )
                    if source:
                        sources.append(source)
            result["retrieved_docs_sources"] = sources[:5]  # 상위 5개
        else:
            result["warnings"].append("retrieved_docs가 없거나 비어있습니다")

        # 2. structured_documents 확인 (다양한 경로에서 확인)
        # state에서 직접 확인 또는 metadata/search에서 확인
        structured_docs = None

        # 경로 1: state에서 직접 확인
        if "structured_documents" in state:
            structured_docs = state.get("structured_documents", {})

        # 경로 2: metadata에서 확인
        if not structured_docs:
            metadata = state.get("metadata", {})
            if isinstance(metadata, dict):
                context_dict = metadata.get("context_dict", {})
                if context_dict:
                    result["has_context_dict"] = True
                    structured_docs = context_dict.get("structured_documents", {})

        # 경로 3: search 그룹에서 확인
        if not structured_docs:
            search = state.get("search", {})
            if isinstance(search, dict):
                structured_docs = search.get("structured_documents", {})

        # 경로 4: prompt_optimized_context에서 확인
        if not structured_docs:
            search = state.get("search", {})
            if isinstance(search, dict):
                prompt_optimized_context = search.get("prompt_optimized_context", {})
                if isinstance(prompt_optimized_context, dict):
                    structured_docs = prompt_optimized_context.get("structured_documents", {})

        if structured_docs and isinstance(structured_docs, dict):
            result["has_structured_documents"] = True
            documents = structured_docs.get("documents", [])
            result["structured_documents_count"] = len(documents)
            if len(documents) > 0:
                result["context_dict_has_documents"] = True
            else:
                result["warnings"].append("context_dict에 structured_documents가 있지만 documents가 비어있습니다")

        # 3. 프롬프트에 문서 포함 여부 확인
        # prompt가 없으면 답변에서 확인 (답변에 소스가 포함되어 있으면 프롬프트에도 포함되었을 가능성)
        if prompt:
            # 문서 섹션 확인
            has_doc_section = (
                "검색된 법률 문서" in prompt or
                "## 🔍" in prompt or
                "## 문서" in prompt or
                "structured_documents" in prompt.lower() or
                "참고 문서" in prompt or
                "관련 문서" in prompt
            )
            result["prompt_has_documents"] = has_doc_section

            # 소스 참조 확인
            if result["retrieved_docs_sources"]:
                sources_in_prompt = sum(
                    1 for source in result["retrieved_docs_sources"]
                    if source in prompt
                )
                result["prompt_has_sources"] = sources_in_prompt > 0
        else:
            # prompt가 없으면 답변에 소스가 포함되어 있는지로 판단
            answer = state.get("answer", "")
            if isinstance(answer, str) and answer:
                # 답변에 소스가 포함되어 있으면 프롬프트에도 포함되었을 가능성
                if result["retrieved_docs_sources"]:
                    sources_in_answer = sum(
                        1 for source in result["retrieved_docs_sources"]
                        if source in answer
                    )
                    if sources_in_answer > 0:
                        result["prompt_has_documents"] = True  # 간접 추정
                        result["prompt_has_sources"] = True

        # 4. 답변에 소스 포함 여부 확인
        answer = state.get("answer", "")
        if isinstance(answer, str) and answer:
            if result["retrieved_docs_sources"]:
                sources_in_answer = sum(
                    1 for source in result["retrieved_docs_sources"]
                    if source in answer
                )
                result["sources_in_answer"] = sources_in_answer > 0

        # 5. 검증 점수 계산 (가중치 적용)
        score = 0.0
        max_score = 10.0

        # 필수: retrieved_docs 존재 (가중치 2.0)
        if result["has_retrieved_docs"]:
            score += 2.0
        if result["retrieved_docs_count"] > 0:
            score += 1.0

        # 중요: 답변에 소스 포함 (가중치 2.0) - 실제 사용 여부를 나타냄
        if result["sources_in_answer"]:
            score += 2.0

        # 중요: structured_documents 존재 (가중치 1.5)
        if result["has_structured_documents"]:
            score += 1.5
        if result["structured_documents_count"] > 0:
            score += 1.0

        # 부가: context_dict 및 프롬프트 (가중치 1.0)
        if result["has_context_dict"]:
            score += 0.5
        if result["context_dict_has_documents"]:
            score += 0.5
        if result["prompt_has_documents"]:
            score += 0.5

        result["verification_score"] = score / max_score

        return result


async def test_search_results_usage():
    """검색 결과 사용 여부 테스트"""
    try:
        from core.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("검색 결과 사용 여부 검증 테스트 시작")
        logger.info("=" * 80)

        # 설정 로드
        config = LangGraphConfig.from_env()
        logger.info(f"✅ 설정 로드 완료")

        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        logger.info(f"✅ 워크플로우 서비스 초기화 완료")

        # 검증 클래스 초기화
        validator = SearchResultsUsageValidator()

        # 테스트 질의
        test_queries = [
            "민사법에서 계약 해지 요건은 무엇인가요?",
        ]

        results = []

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 질의 {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                session_id = f"test_search_validation_{int(time.time())}_{i}"

                # 워크플로우 실행
                start_time = time.time()
                result = await workflow_service.process_query(
                    query,
                    session_id,
                    enable_checkpoint=False
                )
                processing_time = time.time() - start_time

                logger.info(f"처리 시간: {processing_time:.2f}초")

                # 검증 실행
                verification = validator.verify_search_results_usage(result)

                # 결과 출력
                logger.info(f"\n📊 검증 결과:")
                logger.info(f"   - retrieved_docs 유무: {'✅ 있음' if verification['has_retrieved_docs'] else '❌ 없음'}")
                logger.info(f"   - retrieved_docs 개수: {verification['retrieved_docs_count']}개")
                if verification['retrieved_docs_sources']:
                    logger.info(f"   - 검색된 소스: {', '.join(verification['retrieved_docs_sources'])}")

                logger.info(f"   - structured_documents 유무: {'✅ 있음' if verification['has_structured_documents'] else '❌ 없음'}")
                logger.info(f"   - structured_documents 개수: {verification['structured_documents_count']}개")
                logger.info(f"   - context_dict 유무: {'✅ 있음' if verification['has_context_dict'] else '❌ 없음'}")
                logger.info(f"   - context_dict에 문서 포함: {'✅ 예' if verification['context_dict_has_documents'] else '❌ 아니오'}")
                logger.info(f"   - 프롬프트에 문서 섹션: {'✅ 있음' if verification['prompt_has_documents'] else '❌ 없음'}")
                logger.info(f"   - 답변에 소스 포함: {'✅ 예' if verification['sources_in_answer'] else '❌ 아니오'}")
                logger.info(f"   - 검증 점수: {verification['verification_score']:.2%}")

                if verification['warnings']:
                    logger.warning(f"\n⚠️ 경고:")
                    for warning in verification['warnings']:
                        logger.warning(f"   - {warning}")

                if verification['errors']:
                    logger.error(f"\n❌ 오류:")
                    for error in verification['errors']:
                        logger.error(f"   - {error}")

                # 상세 정보 출력
                logger.info(f"\n📝 상세 정보:")

                # retrieved_docs 상세
                retrieved_docs = result.get("retrieved_docs", [])
                if retrieved_docs:
                    logger.info(f"   retrieved_docs:")
                    for idx, doc in enumerate(retrieved_docs[:3], 1):
                        source = doc.get("source", "Unknown")
                        content_preview = (doc.get("content") or doc.get("text", ""))[:100]
                        logger.info(f"      [{idx}] {source}: {content_preview}...")

                # answer 확인
                answer = result.get("answer", "")
                if answer:
                    answer_preview = answer[:200] if isinstance(answer, str) else str(answer)[:200]
                    logger.info(f"\n   답변 미리보기:")
                    logger.info(f"      {answer_preview}...")

                    # 답변에 소스 포함 여부 확인
                    if retrieved_docs:
                        sources_found = []
                        for doc in retrieved_docs:
                            source = doc.get("source", "")
                            if source and source in answer:
                                sources_found.append(source)

                        if sources_found:
                            logger.info(f"\n   ✅ 답변에 포함된 소스: {', '.join(sources_found)}")
                        else:
                            logger.warning(f"\n   ⚠️ 답변에 검색된 소스가 포함되지 않았습니다")

                # 최종 판정
                is_valid = verification['verification_score'] >= 0.75
                status = "✅ 통과" if is_valid else "❌ 실패"

                logger.info(f"\n{status} 검증 완료 (점수: {verification['verification_score']:.2%})")

                results.append({
                    "query": query,
                    "verification": verification,
                    "is_valid": is_valid,
                    "processing_time": processing_time,
                    "result": result
                })

            except Exception as e:
                import traceback
                logger.error(f"\n❌ 테스트 질의 실패: {query}")
                logger.error(f"오류: {str(e)}")
                logger.error(f"스택 트레이스:\n{traceback.format_exc()}")

                results.append({
                    "query": query,
                    "error": str(e),
                    "is_valid": False
                })

        # 최종 요약
        logger.info(f"\n{'='*80}")
        logger.info("테스트 결과 요약")
        logger.info(f"{'='*80}")

        total = len(results)
        valid = sum(1 for r in results if r.get("is_valid", False))
        failed = total - valid

        logger.info(f"   총 테스트: {total}")
        logger.info(f"   통과: {valid}")
        logger.info(f"   실패: {failed}")

        if valid > 0:
            avg_score = sum(
                r.get("verification", {}).get("verification_score", 0.0)
                for r in results if r.get("is_valid", False)
            ) / valid
            logger.info(f"   평균 검증 점수: {avg_score:.2%}")

        # 실패한 테스트 분석
        if failed > 0:
            logger.warning(f"\n{'='*80}")
            logger.warning("실패한 테스트 분석")
            logger.warning(f"{'='*80}")

            for i, result in enumerate(results, 1):
                if not result.get("is_valid", False):
                    logger.warning(f"\n[{i}] 질의: {result.get('query')}")
                    verification = result.get("verification", {})

                    if not verification.get("has_retrieved_docs"):
                        logger.warning(f"   ❌ retrieved_docs 없음")
                    if not verification.get("has_structured_documents"):
                        logger.warning(f"   ❌ structured_documents 없음")
                    if not verification.get("context_dict_has_documents"):
                        logger.warning(f"   ❌ context_dict에 문서 없음")
                    if not verification.get("prompt_has_documents"):
                        logger.warning(f"   ❌ 프롬프트에 문서 섹션 없음")
                    if not verification.get("sources_in_answer"):
                        logger.warning(f"   ❌ 답변에 소스 포함 안 됨")

        # 최종 판정
        logger.info(f"\n{'='*80}")
        if valid == total:
            logger.info("✅ 모든 테스트 통과!")
            print("✅ 모든 테스트 통과!")
            return True
        elif valid > 0:
            logger.warning(f"⚠️ 부분 성공: {valid}/{total}")
            print(f"⚠️ 부분 성공: {valid}/{total}")
            return False
        else:
            logger.error("❌ 모든 테스트 실패")
            print("❌ 모든 테스트 실패")
            return False

    except Exception as e:
        import traceback
        logger.error(f"테스트 실행 중 오류: {e}")
        logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(test_search_results_usage())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        sys.exit(1)
