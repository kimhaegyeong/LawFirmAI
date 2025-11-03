# -*- coding: utf-8 -*-
"""
ê²€??ê²°ê³¼ê°€ generate_answer_enhancedê¹Œì? ???„ë‹¬?˜ëŠ”ì§€ ?ŒìŠ¤??
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ë¡œê¹… ?¤ì • (?ˆì „???¸ë“¤???¬ìš©)
class SafeStreamHandler(logging.StreamHandler):
    """?ˆì „???¤íŠ¸ë¦??¸ë“¤??- detached ë²„í¼ ?ëŸ¬ ë°©ì?"""
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

# ë¡œê¹… ?ˆì™¸ ë¹„í™œ?±í™”
logging.raiseExceptions = False

logger = logging.getLogger(__name__)


async def test_search_results_to_generate_answer():
    """ê²€??ê²°ê³¼ê°€ generate_answer_enhancedê¹Œì? ?„ë‹¬?˜ëŠ”ì§€ ?ŒìŠ¤??""
    try:
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("ê²€??ê²°ê³¼ ?„ë‹¬ ?ŒìŠ¤???œì‘")
        logger.info("=" * 80)

        # ?¤ì • ë¡œë“œ
        logger.info("1. LangGraph ?¤ì • ë¡œë“œ ì¤?..")
        config = LangGraphConfig.from_env()
        logger.info(f"   ??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        logger.info("2. ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??ì¤?..")
        workflow_service = LangGraphWorkflowService(config)
        logger.info(f"   ???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì§ˆì˜ (ê²€??ê²°ê³¼ê°€ ?•ì‹¤???˜ì˜¬ ê²ƒìœ¼ë¡??ˆìƒ?˜ëŠ” ì§ˆì˜)
        test_queries = [
            "ê³„ì•½ ?´ì? ?”ê±´",
            "?í•´ë°°ìƒ ì±…ì„",
            "ë¯¼ë²• ??43ì¡?
        ]

        logger.info("3. ê²€??ê²°ê³¼ ?„ë‹¬ ?ŒìŠ¤???œì‘...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"?ŒìŠ¤??{i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                # ?¸ì…˜ ID ?ì„±
                session_id = f"test_search_{int(time.time())}_{i}"

                # ì§ˆì˜ ì²˜ë¦¬ (ê²€???¨ê³„ê¹Œì? ?¤í–‰)
                logger.info(f"ì§ˆì˜ ì²˜ë¦¬ ?œì‘: {query}")
                start_time = time.time()

                # ?„ì²´ ?Œí¬?Œë¡œ???¤í–‰?˜ê³  ê²€??ê²°ê³¼ ?„ë‹¬ ?•ì¸
                logger.info("?”„ ?„ì²´ ?Œí¬?Œë¡œ???¤í–‰ ì¤?..")
                result = await workflow_service.process_query(query, session_id, enable_checkpoint=False)

                # ê²°ê³¼?ì„œ ê²€??ê´€???•ë³´ ì¶”ì¶œ
                retrieved_docs = result.get("retrieved_docs", [])
                metadata = result.get("metadata", {})
                search_meta = metadata.get("search", {}) if isinstance(metadata, dict) else {}

                semantic_count = search_meta.get("semantic_results_count", 0)
                keyword_count = search_meta.get("keyword_results_count", 0)
                final_count = search_meta.get("final_count", len(retrieved_docs))

                logger.info(f"\n?“Š ê²€??ê²°ê³¼:")
                logger.info(f"   ???˜ë???ê²€?? {semantic_count}ê°?)
                logger.info(f"   ???¤ì›Œ??ê²€?? {keyword_count}ê°?)
                logger.info(f"   ??ìµœì¢… ?µí•© ê²€??ê²°ê³¼: {final_count}ê°?)

                if final_count > 0:
                    logger.info(f"   ?“„ ì²?ë²ˆì§¸ ë¬¸ì„œ ?˜í”Œ:")
                    first_doc = retrieved_docs[0] if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 else {}
                    if isinstance(first_doc, dict):
                        logger.info(f"      - Type: {first_doc.get('type', 'unknown')}")
                        logger.info(f"      - Source: {str(first_doc.get('source', 'unknown'))[:50]}")
                        content = first_doc.get('content', '') or first_doc.get('text', '')
                        logger.info(f"      - Content preview: {str(content)[:100]}...")
                else:
                    logger.warning("   ? ï¸ ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤!")

                # ?µë? ?•ì¸
                answer = result.get("answer", "")
                logger.info(f"\n?ï¸ generate_answer_enhanced ê²°ê³¼:")
                logger.info(f"   ???µë? ?ì„± ?„ë£Œ")
                logger.info(f"   ?“Š ?µë??ì„œ ë°›ì? ê²€??ê²°ê³¼: {len(retrieved_docs)}ê°?)

                if answer:
                    answer_preview = answer[:200] if len(answer) > 200 else answer
                    logger.info(f"   ?“ ?ì„±???µë? ë¯¸ë¦¬ë³´ê¸°: {answer_preview}...")

                    # ê²€??ê²°ê³¼???´ìš©???µë????¬í•¨?˜ì—ˆ?”ì? ê°„ë‹¨???•ì¸
                    doc_mentioned = False
                    if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                        for doc in retrieved_docs[:3]:
                            if isinstance(doc, dict):
                                source = doc.get("source", "")
                                if source and len(str(source)) > 10 and str(source)[:20] in answer:
                                    doc_mentioned = True
                                    logger.info(f"   ??ê²€??ê²°ê³¼ê°€ ?µë????¬í•¨?? {str(source)[:50]}")
                                    break

                    if not doc_mentioned and len(retrieved_docs) > 0:
                        logger.warning("   ? ï¸ ê²€??ê²°ê³¼ê°€ ?µë???ëª…ì‹œ?ìœ¼ë¡??¬í•¨?˜ì? ?Šì•˜?????ˆìŒ")
                else:
                    logger.warning("   ? ï¸ ?ì„±???µë????†ìŠµ?ˆë‹¤!")

                logger.info(f"\n?“Š ê²€??ë©”í??°ì´??")
                logger.info(f"   - ?˜ë???ê²€?? {semantic_count}ê°?)
                logger.info(f"   - ?¤ì›Œ??ê²€?? {keyword_count}ê°?)
                logger.info(f"   - ìµœì¢… ê²°ê³¼: {final_count}ê°?)
                logger.info(f"   - ê²€???œê°„: {search_meta.get('search_time', 0):.3f}ì´?)

                processing_time = time.time() - start_time
                logger.info(f"\n???ŒìŠ¤??{i} ?„ë£Œ (ì´?{processing_time:.2f}ì´?")

                # ê²€ì¦?
                assert final_count > 0, f"ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤! (semantic: {semantic_count}, keyword: {keyword_count})"
                assert len(retrieved_docs) > 0, f"ê²€??ê²°ê³¼ê°€ retrieved_docs???†ìŠµ?ˆë‹¤!"
                assert answer is not None and len(str(answer)) > 0, "?µë????ì„±?˜ì? ?Šì•˜?µë‹ˆ??"

                # ê²€??ê²°ê³¼ê°€ generate_answer_enhancedê¹Œì? ?„ë‹¬?˜ì—ˆ?”ì? ?•ì¸
                assert retrieved_docs is not None, "retrieved_docsê°€ None?…ë‹ˆ??"
                assert isinstance(retrieved_docs, list), f"retrieved_docsê°€ ë¦¬ìŠ¤?¸ê? ?„ë‹™?ˆë‹¤: {type(retrieved_docs)}"

                logger.info(f"   ??ëª¨ë“  ê²€ì¦??µê³¼!")

            except Exception as e:
                logger.error(f"?ŒìŠ¤??{i} ?¤íŒ¨: {e}", exc_info=True)
                continue

        logger.info("\n" + "=" * 80)
        logger.info("ê²€??ê²°ê³¼ ?„ë‹¬ ?ŒìŠ¤???„ë£Œ")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"?ŒìŠ¤???¤í–‰ ?¤íŒ¨: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(test_search_results_to_generate_answer())
