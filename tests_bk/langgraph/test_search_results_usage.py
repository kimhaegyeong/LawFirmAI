# -*- coding: utf-8 -*-
"""
ê²€?‰ëœ ë¬¸ì„œ ê²°ê³¼ê°€ ?„ë¡¬?„íŠ¸ ?‘ì„±???œë?ë¡??¬ìš©?˜ëŠ”ì§€ ê²€ì¦í•˜???ŒìŠ¤??
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)


class SearchResultsUsageValidator:
    """ê²€??ê²°ê³¼ ?¬ìš© ?¬ë? ê²€ì¦??´ë˜??""

    def __init__(self):
        self.verification_results = []

    def verify_search_results_usage(self, state: dict, prompt: str = None) -> dict:
        """
        ê²€??ê²°ê³¼ê°€ ?„ë¡¬?„íŠ¸???¬ìš©?˜ì—ˆ?”ì? ê²€ì¦?

        Args:
            state: ?Œí¬?Œë¡œ??state
            prompt: ?ì„±???„ë¡¬?„íŠ¸ (? íƒ??

        Returns:
            ê²€ì¦?ê²°ê³¼ ?•ì…”?ˆë¦¬
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

        # 1. retrieved_docs ?•ì¸
        retrieved_docs = state.get("retrieved_docs", [])
        if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
            result["has_retrieved_docs"] = True
            result["retrieved_docs_count"] = len(retrieved_docs)

            # ?ŒìŠ¤ ì¶”ì¶œ
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
            result["retrieved_docs_sources"] = sources[:5]  # ?ìœ„ 5ê°?
        else:
            result["warnings"].append("retrieved_docsê°€ ?†ê±°??ë¹„ì–´?ˆìŠµ?ˆë‹¤")

        # 2. structured_documents ?•ì¸ (?¤ì–‘??ê²½ë¡œ?ì„œ ?•ì¸)
        # state?ì„œ ì§ì ‘ ?•ì¸ ?ëŠ” metadata/search?ì„œ ?•ì¸
        structured_docs = None

        # ê²½ë¡œ 1: state?ì„œ ì§ì ‘ ?•ì¸
        if "structured_documents" in state:
            structured_docs = state.get("structured_documents", {})

        # ê²½ë¡œ 2: metadata?ì„œ ?•ì¸
        if not structured_docs:
            metadata = state.get("metadata", {})
            if isinstance(metadata, dict):
                context_dict = metadata.get("context_dict", {})
                if context_dict:
                    result["has_context_dict"] = True
                    structured_docs = context_dict.get("structured_documents", {})

        # ê²½ë¡œ 3: search ê·¸ë£¹?ì„œ ?•ì¸
        if not structured_docs:
            search = state.get("search", {})
            if isinstance(search, dict):
                structured_docs = search.get("structured_documents", {})

        # ê²½ë¡œ 4: prompt_optimized_context?ì„œ ?•ì¸
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
                result["warnings"].append("context_dict??structured_documentsê°€ ?ˆì?ë§?documentsê°€ ë¹„ì–´?ˆìŠµ?ˆë‹¤")

        # 3. ?„ë¡¬?„íŠ¸??ë¬¸ì„œ ?¬í•¨ ?¬ë? ?•ì¸
        # promptê°€ ?†ìœ¼ë©??µë??ì„œ ?•ì¸ (?µë????ŒìŠ¤ê°€ ?¬í•¨?˜ì–´ ?ˆìœ¼ë©??„ë¡¬?„íŠ¸?ë„ ?¬í•¨?˜ì—ˆ??ê°€?¥ì„±)
        if prompt:
            # ë¬¸ì„œ ?¹ì…˜ ?•ì¸
            has_doc_section = (
                "ê²€?‰ëœ ë²•ë¥  ë¬¸ì„œ" in prompt or
                "## ?”" in prompt or
                "## ë¬¸ì„œ" in prompt or
                "structured_documents" in prompt.lower() or
                "ì°¸ê³  ë¬¸ì„œ" in prompt or
                "ê´€??ë¬¸ì„œ" in prompt
            )
            result["prompt_has_documents"] = has_doc_section

            # ?ŒìŠ¤ ì°¸ì¡° ?•ì¸
            if result["retrieved_docs_sources"]:
                sources_in_prompt = sum(
                    1 for source in result["retrieved_docs_sources"]
                    if source in prompt
                )
                result["prompt_has_sources"] = sources_in_prompt > 0
        else:
            # promptê°€ ?†ìœ¼ë©??µë????ŒìŠ¤ê°€ ?¬í•¨?˜ì–´ ?ˆëŠ”ì§€ë¡??ë‹¨
            answer = state.get("answer", "")
            if isinstance(answer, str) and answer:
                # ?µë????ŒìŠ¤ê°€ ?¬í•¨?˜ì–´ ?ˆìœ¼ë©??„ë¡¬?„íŠ¸?ë„ ?¬í•¨?˜ì—ˆ??ê°€?¥ì„±
                if result["retrieved_docs_sources"]:
                    sources_in_answer = sum(
                        1 for source in result["retrieved_docs_sources"]
                        if source in answer
                    )
                    if sources_in_answer > 0:
                        result["prompt_has_documents"] = True  # ê°„ì ‘ ì¶”ì •
                        result["prompt_has_sources"] = True

        # 4. ?µë????ŒìŠ¤ ?¬í•¨ ?¬ë? ?•ì¸
        answer = state.get("answer", "")
        if isinstance(answer, str) and answer:
            if result["retrieved_docs_sources"]:
                sources_in_answer = sum(
                    1 for source in result["retrieved_docs_sources"]
                    if source in answer
                )
                result["sources_in_answer"] = sources_in_answer > 0

        # 5. ê²€ì¦??ìˆ˜ ê³„ì‚° (ê°€ì¤‘ì¹˜ ?ìš©)
        score = 0.0
        max_score = 10.0

        # ?„ìˆ˜: retrieved_docs ì¡´ì¬ (ê°€ì¤‘ì¹˜ 2.0)
        if result["has_retrieved_docs"]:
            score += 2.0
        if result["retrieved_docs_count"] > 0:
            score += 1.0

        # ì¤‘ìš”: ?µë????ŒìŠ¤ ?¬í•¨ (ê°€ì¤‘ì¹˜ 2.0) - ?¤ì œ ?¬ìš© ?¬ë?ë¥??˜í???
        if result["sources_in_answer"]:
            score += 2.0

        # ì¤‘ìš”: structured_documents ì¡´ì¬ (ê°€ì¤‘ì¹˜ 1.5)
        if result["has_structured_documents"]:
            score += 1.5
        if result["structured_documents_count"] > 0:
            score += 1.0

        # ë¶€ê°€: context_dict ë°??„ë¡¬?„íŠ¸ (ê°€ì¤‘ì¹˜ 1.0)
        if result["has_context_dict"]:
            score += 0.5
        if result["context_dict_has_documents"]:
            score += 0.5
        if result["prompt_has_documents"]:
            score += 0.5

        result["verification_score"] = score / max_score

        return result


async def test_search_results_usage():
    """ê²€??ê²°ê³¼ ?¬ìš© ?¬ë? ?ŒìŠ¤??""
    try:
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("ê²€??ê²°ê³¼ ?¬ìš© ?¬ë? ê²€ì¦??ŒìŠ¤???œì‘")
        logger.info("=" * 80)

        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        logger.info(f"???¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        workflow_service = LangGraphWorkflowService(config)
        logger.info(f"???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ")

        # ê²€ì¦??´ë˜??ì´ˆê¸°??
        validator = SearchResultsUsageValidator()

        # ?ŒìŠ¤??ì§ˆì˜
        test_queries = [
            "ë¯¼ì‚¬ë²•ì—??ê³„ì•½ ?´ì? ?”ê±´?€ ë¬´ì—‡?¸ê???",
        ]

        results = []

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"?ŒìŠ¤??ì§ˆì˜ {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                session_id = f"test_search_validation_{int(time.time())}_{i}"

                # ?Œí¬?Œë¡œ???¤í–‰
                start_time = time.time()
                result = await workflow_service.process_query(
                    query,
                    session_id,
                    enable_checkpoint=False
                )
                processing_time = time.time() - start_time

                logger.info(f"ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)

                # ê²€ì¦??¤í–‰
                verification = validator.verify_search_results_usage(result)

                # ê²°ê³¼ ì¶œë ¥
                logger.info(f"\n?“Š ê²€ì¦?ê²°ê³¼:")
                logger.info(f"   - retrieved_docs ? ë¬´: {'???ˆìŒ' if verification['has_retrieved_docs'] else '???†ìŒ'}")
                logger.info(f"   - retrieved_docs ê°œìˆ˜: {verification['retrieved_docs_count']}ê°?)
                if verification['retrieved_docs_sources']:
                    logger.info(f"   - ê²€?‰ëœ ?ŒìŠ¤: {', '.join(verification['retrieved_docs_sources'])}")

                logger.info(f"   - structured_documents ? ë¬´: {'???ˆìŒ' if verification['has_structured_documents'] else '???†ìŒ'}")
                logger.info(f"   - structured_documents ê°œìˆ˜: {verification['structured_documents_count']}ê°?)
                logger.info(f"   - context_dict ? ë¬´: {'???ˆìŒ' if verification['has_context_dict'] else '???†ìŒ'}")
                logger.info(f"   - context_dict??ë¬¸ì„œ ?¬í•¨: {'???? if verification['context_dict_has_documents'] else '???„ë‹ˆ??}")
                logger.info(f"   - ?„ë¡¬?„íŠ¸??ë¬¸ì„œ ?¹ì…˜: {'???ˆìŒ' if verification['prompt_has_documents'] else '???†ìŒ'}")
                logger.info(f"   - ?µë????ŒìŠ¤ ?¬í•¨: {'???? if verification['sources_in_answer'] else '???„ë‹ˆ??}")
                logger.info(f"   - ê²€ì¦??ìˆ˜: {verification['verification_score']:.2%}")

                if verification['warnings']:
                    logger.warning(f"\n? ï¸ ê²½ê³ :")
                    for warning in verification['warnings']:
                        logger.warning(f"   - {warning}")

                if verification['errors']:
                    logger.error(f"\n???¤ë¥˜:")
                    for error in verification['errors']:
                        logger.error(f"   - {error}")

                # ?ì„¸ ?•ë³´ ì¶œë ¥
                logger.info(f"\n?“ ?ì„¸ ?•ë³´:")

                # retrieved_docs ?ì„¸
                retrieved_docs = result.get("retrieved_docs", [])
                if retrieved_docs:
                    logger.info(f"   retrieved_docs:")
                    for idx, doc in enumerate(retrieved_docs[:3], 1):
                        source = doc.get("source", "Unknown")
                        content_preview = (doc.get("content") or doc.get("text", ""))[:100]
                        logger.info(f"      [{idx}] {source}: {content_preview}...")

                # answer ?•ì¸
                answer = result.get("answer", "")
                if answer:
                    answer_preview = answer[:200] if isinstance(answer, str) else str(answer)[:200]
                    logger.info(f"\n   ?µë? ë¯¸ë¦¬ë³´ê¸°:")
                    logger.info(f"      {answer_preview}...")

                    # ?µë????ŒìŠ¤ ?¬í•¨ ?¬ë? ?•ì¸
                    if retrieved_docs:
                        sources_found = []
                        for doc in retrieved_docs:
                            source = doc.get("source", "")
                            if source and source in answer:
                                sources_found.append(source)

                        if sources_found:
                            logger.info(f"\n   ???µë????¬í•¨???ŒìŠ¤: {', '.join(sources_found)}")
                        else:
                            logger.warning(f"\n   ? ï¸ ?µë???ê²€?‰ëœ ?ŒìŠ¤ê°€ ?¬í•¨?˜ì? ?Šì•˜?µë‹ˆ??)

                # ìµœì¢… ?ì •
                is_valid = verification['verification_score'] >= 0.75
                status = "???µê³¼" if is_valid else "???¤íŒ¨"

                logger.info(f"\n{status} ê²€ì¦??„ë£Œ (?ìˆ˜: {verification['verification_score']:.2%})")

                results.append({
                    "query": query,
                    "verification": verification,
                    "is_valid": is_valid,
                    "processing_time": processing_time,
                    "result": result
                })

            except Exception as e:
                import traceback
                logger.error(f"\n???ŒìŠ¤??ì§ˆì˜ ?¤íŒ¨: {query}")
                logger.error(f"?¤ë¥˜: {str(e)}")
                logger.error(f"?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{traceback.format_exc()}")

                results.append({
                    "query": query,
                    "error": str(e),
                    "is_valid": False
                })

        # ìµœì¢… ?”ì•½
        logger.info(f"\n{'='*80}")
        logger.info("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        logger.info(f"{'='*80}")

        total = len(results)
        valid = sum(1 for r in results if r.get("is_valid", False))
        failed = total - valid

        logger.info(f"   ì´??ŒìŠ¤?? {total}")
        logger.info(f"   ?µê³¼: {valid}")
        logger.info(f"   ?¤íŒ¨: {failed}")

        if valid > 0:
            avg_score = sum(
                r.get("verification", {}).get("verification_score", 0.0)
                for r in results if r.get("is_valid", False)
            ) / valid
            logger.info(f"   ?‰ê·  ê²€ì¦??ìˆ˜: {avg_score:.2%}")

        # ?¤íŒ¨???ŒìŠ¤??ë¶„ì„
        if failed > 0:
            logger.warning(f"\n{'='*80}")
            logger.warning("?¤íŒ¨???ŒìŠ¤??ë¶„ì„")
            logger.warning(f"{'='*80}")

            for i, result in enumerate(results, 1):
                if not result.get("is_valid", False):
                    logger.warning(f"\n[{i}] ì§ˆì˜: {result.get('query')}")
                    verification = result.get("verification", {})

                    if not verification.get("has_retrieved_docs"):
                        logger.warning(f"   ??retrieved_docs ?†ìŒ")
                    if not verification.get("has_structured_documents"):
                        logger.warning(f"   ??structured_documents ?†ìŒ")
                    if not verification.get("context_dict_has_documents"):
                        logger.warning(f"   ??context_dict??ë¬¸ì„œ ?†ìŒ")
                    if not verification.get("prompt_has_documents"):
                        logger.warning(f"   ???„ë¡¬?„íŠ¸??ë¬¸ì„œ ?¹ì…˜ ?†ìŒ")
                    if not verification.get("sources_in_answer"):
                        logger.warning(f"   ???µë????ŒìŠ¤ ?¬í•¨ ????)

        # ìµœì¢… ?ì •
        logger.info(f"\n{'='*80}")
        if valid == total:
            logger.info("??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
            print("??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
            return True
        elif valid > 0:
            logger.warning(f"? ï¸ ë¶€ë¶??±ê³µ: {valid}/{total}")
            print(f"? ï¸ ë¶€ë¶??±ê³µ: {valid}/{total}")
            return False
        else:
            logger.error("??ëª¨ë“  ?ŒìŠ¤???¤íŒ¨")
            print("??ëª¨ë“  ?ŒìŠ¤???¤íŒ¨")
            return False

    except Exception as e:
        import traceback
        logger.error(f"?ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜: {e}")
        logger.error(f"?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(test_search_results_usage())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n?ŒìŠ¤?¸ê? ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ì¹˜ëª…???¤ë¥˜: {e}")
        sys.exit(1)
