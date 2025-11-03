# -*- coding: utf-8 -*-
"""
LangGraph ?™ì‘ ?ŒìŠ¤???¤í¬ë¦½íŠ¸
ë¦¬íŒ©? ë§ ??LangGraphê°€ ?•ìƒ?ìœ¼ë¡??™ì‘?˜ëŠ”ì§€ ?•ì¸
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
# Windows ë¹„ë™ê¸??˜ê²½?ì„œ ë¡œê¹… ë²„í¼ ?ëŸ¬ ë°©ì?
class SafeStreamHandler(logging.StreamHandler):
    """?ˆì „???¤íŠ¸ë¦??¸ë“¤??- detached ë²„í¼ ?ëŸ¬ ë°©ì?"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            # detached buffer ?ëŸ¬??ê¸°í? ?¤íŠ¸ë¦??ëŸ¬ ë¬´ì‹œ
            pass

# ë¡œê¹… ?¤ì • (INFO ?ˆë²¨ë¡?ë³€ê²½í•˜??ì¤‘ìš”???•ë³´ë§??•ì¸)
# DEBUG ?ˆë²¨?€ ?ˆë¬´ ë§ì? ë¡œê·¸ë¥??ì„±?˜ë?ë¡?INFOë¡?ì¡°ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[SafeStreamHandler()],
    force=True  # ê¸°ì¡´ ?¤ì •??ê°•ì œë¡??¬ì„¤??
)

# ?¹ì • ë¡œê±°???ˆë²¨??INFOë¡??¤ì • (DEBUG??? íƒ??
legal_workflow_logger = logging.getLogger('core.agents.legal_workflow_enhanced')
legal_workflow_logger.setLevel(logging.INFO)  # DEBUG?ì„œ INFOë¡?ë³€ê²?
legal_workflow_logger.propagate = True

# ?¸ë“¤?¬ê? ?†ìœ¼ë©?ì¶”ê?
if not legal_workflow_logger.handlers:
    legal_workflow_logger.addHandler(SafeStreamHandler())

# ?¤ë¥¸ ë¡œê±°?¤ë„ ?™ì¼?˜ê²Œ ?¤ì • (INFO ?ˆë²¨)
semantic_search_logger = logging.getLogger('source.services.semantic_search_engine_v2')
semantic_search_logger.setLevel(logging.INFO)  # DEBUG?ì„œ INFOë¡?ë³€ê²?
semantic_search_logger.propagate = True
if not semantic_search_logger.handlers:
    semantic_search_logger.addHandler(SafeStreamHandler())

data_connector_logger = logging.getLogger('core.agents.legal_data_connector_v2')
data_connector_logger.setLevel(logging.INFO)  # DEBUG?ì„œ INFOë¡?ë³€ê²?
data_connector_logger.propagate = True
if not data_connector_logger.handlers:
    data_connector_logger.addHandler(SafeStreamHandler())

# workflow_service??DEBUG ë¡œê·¸??WARNING ?ˆë²¨ë¡?ì¡°ì •
workflow_service_logger = logging.getLogger('core.agents.workflow_service')
workflow_service_logger.setLevel(logging.WARNING)  # DEBUG ë¡œê·¸ ?µì œ

# ë¡œê¹… ?ëŸ¬ë¥??µì œ
logging.raiseExceptions = False

logger = logging.getLogger(__name__)

# pytest-asyncio ì§€??(? íƒ??
try:
    import pytest
    pytest_available = True
except ImportError:
    pytest_available = False


async def test_langgraph_workflow():
    """LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??""
    try:
        # core/agents/legal_workflow_enhanced.pyë¥??¬ìš©?˜ë„ë¡?ë³€ê²?
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤???œì‘")
        logger.info("=" * 80)

        # ?¤ì • ë¡œë“œ
        logger.info("1. LangGraph ?¤ì • ë¡œë“œ ì¤?..")
        config = LangGraphConfig.from_env()
        logger.info(f"   ??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ (enabled={config.langgraph_enabled})")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        logger.info("2. ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??ì¤?..")
        start_time = time.time()
        workflow_service = LangGraphWorkflowService(config)
        init_time = time.time() - start_time
        logger.info(f"   ???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ ({init_time:.2f}ì´?")

        # ?ŒìŠ¤??ì§ˆì˜ (?¤ì–‘??ë²•ë¥  ë¶„ì•¼, 5ê°?
        test_queries = [
            "ë¯¼ì‚¬ë²•ì—??ê³„ì•½ ?´ì? ?”ê±´?€ ë¬´ì—‡?¸ê???",
            "?•ë²•?ì„œ ?ˆë„ì£„ì˜ ?±ë¦½ ?”ê±´ê³?ì²˜ë²Œ?€ ?´ë–»ê²??˜ë‚˜??",
            "ê°€ì¡±ë²•?ì„œ ?‘ì˜?´í˜¼ê³??¬íŒ???´í˜¼??ì°¨ì´??ë¬´ì—‡?¸ê???",
            "?¸ë™ë²•ì—??ê·¼ë¡œê³„ì•½???‘ì„± ???¬í•¨?´ì•¼ ???„ìˆ˜ ?¬í•­?€ ë¬´ì—‡?¸ê???",
            "?‰ì •ë²•ì—???‰ì •ì²˜ë¶„ ì·¨ì†Œ ì²?µ¬???œì†Œê¸°ê°„?€ ?¸ì œê¹Œì??¸ê???"
        ]

        logger.info("3. ?ŒìŠ¤??ì§ˆì˜ ?¤í–‰ ì¤?..")

        results = []
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"?ŒìŠ¤??ì§ˆì˜ {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*80}")

            try:
                # ?¸ì…˜ ID ?ì„±
                session_id = f"test_session_{int(time.time())}_{i}"

                # ì§ˆì˜ ì²˜ë¦¬
                start_time = time.time()
                result = await workflow_service.process_query(query, session_id, enable_checkpoint=False)
                processing_time = time.time() - start_time

                # ê²°ê³¼ ê²€ì¦?(ì¤‘ì²© ?•ì…”?ˆë¦¬ ?ˆì „?˜ê²Œ ì¶”ì¶œ)
                answer = result.get("answer", "") if isinstance(result, dict) else ""

                # ì¤‘ì²© ?•ì…”?ˆë¦¬?ì„œ ë¬¸ì??ì¶”ì¶œ
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

                # ìµœì¢… ë¬¸ì??ë³´ì¥
                answer = str(answer) if not isinstance(answer, str) else answer
                has_answer = bool(answer) and len(answer) > 0

                # Sources ?•ì¸ ë¡œì§ ê°œì„  (V2 ìµœì ??
                sources = result.get("sources", []) if isinstance(result, dict) else []
                retrieved_docs = result.get("retrieved_docs", []) if isinstance(result, dict) else []

                # Sources ?•ì¸: ì§ì ‘ sources ?„ë“œ ?ëŠ” retrieved_docs?ì„œ ì¶”ì¶œ
                has_sources = False
                sources_count = 0
                sources_list = []

                if isinstance(sources, list) and len(sources) > 0:
                    # sources ?„ë“œê°€ ?ˆê³  ë¹„ì–´?ˆì? ?Šì? ê²½ìš°
                    has_sources = True
                    sources_count = len(sources)
                    sources_list = sources[:5]  # ?ìœ„ 5ê°œë§Œ
                elif isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                    # retrieved_docs?ì„œ sources ì¶”ì¶œ ?œë„
                    extracted_sources = []
                    for doc in retrieved_docs:
                        if isinstance(doc, dict):
                            # ?¤ì–‘???„ë“œ?ì„œ source ì¶”ì¶œ ?œë„
                            source = (
                                doc.get("source") or
                                doc.get("source_name") or
                                doc.get("statute_name") or
                                None
                            )

                            # metadata?ì„œ??ì¶”ì¶œ ?œë„
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

                # ?±ê³µ ?¬ë? ?ì •
                is_success = has_answer and not has_errors
                result_status = "???±ê³µ" if is_success else "???¤íŒ¨"

                # ê²°ê³¼ ì¶œë ¥
                logger.info(f"\n{result_status} ?µë? ?ì„± ?„ë£Œ (ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?")
                logger.info(f"   - ?µë? ? ë¬´: {'?ˆìŒ' if has_answer else '?†ìŒ'}")
                answer_length = len(answer) if isinstance(answer, str) else 0
                logger.info(f"   - ?µë? ê¸¸ì´: {answer_length}??)
                logger.info(f"   - ?ŒìŠ¤ ? ë¬´: {'?ˆìŒ' if has_sources else '?†ìŒ'}")
                if has_sources:
                    logger.info(f"   - ?ŒìŠ¤ ê°œìˆ˜: {sources_count}ê°?)
                    if sources_list:
                        logger.info(f"   - ì£¼ìš” ?ŒìŠ¤: {', '.join(str(s) for s in sources_list)}")
                else:
                    logger.warning(f"   - ?ŒìŠ¤ ?†ìŒ (retrieved_docs: {len(retrieved_docs)}ê°?")
                logger.info(f"   - ? ë¢°?? {confidence:.2%}")
                logger.info(f"   - ?ëŸ¬ ? ë¬´: {'?ˆìŒ' if has_errors else '?†ìŒ'}")

                # stdout?ë„ ì¶œë ¥ (ë²„í¼ë§?ë°©ì?)
                print(f"\n{result_status} ì§ˆì˜ {i}/{len(test_queries)}: {query} (ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?", flush=True)
                print(f"   - ?µë? ? ë¬´: {'?ˆìŒ' if has_answer else '?†ìŒ'}", flush=True)
                print(f"   - ?µë? ê¸¸ì´: {answer_length}??, flush=True)
                print(f"   - ?ŒìŠ¤ ? ë¬´: {'?ˆìŒ' if has_sources else '?†ìŒ'}", flush=True)
                if has_sources:
                    print(f"   - ?ŒìŠ¤ ê°œìˆ˜: {sources_count}ê°?, flush=True)
                    if sources_list:
                        print(f"   - ì£¼ìš” ?ŒìŠ¤: {', '.join(str(s) for s in sources_list)}", flush=True)
                else:
                    print(f"   - ?ŒìŠ¤ ?†ìŒ (retrieved_docs: {len(retrieved_docs)}ê°?", flush=True)
                print(f"   - ? ë¢°?? {confidence:.2%}", flush=True)
                print(f"   - ?ëŸ¬ ? ë¬´: {'?ˆìŒ' if has_errors else '?†ìŒ'}", flush=True)

                if has_answer:
                    # ë¶„ë¦¬??ë©”í? ?•ë³´ ?„ë“œ ?•ì¸
                    confidence_info = result.get("confidence_info", "") if isinstance(result, dict) else ""
                    reference_materials = result.get("reference_materials", "") if isinstance(result, dict) else ""
                    disclaimer = result.get("disclaimer", "") if isinstance(result, dict) else ""

                    logger.info(f"\n?“ ?µë? ë¯¸ë¦¬ë³´ê¸°:")
                    # answerê°€ ë¬¸ì?´ì¸ì§€ ?•ì¸
                    if isinstance(answer, str):
                        answer_preview = answer[:500]
                        logger.info(f"   {answer_preview}{'...' if len(answer) > 500 else ''}")
                        print(f"\n?“ ?µë? ?„ì²´ ?´ìš© (answer ?„ë“œë§?:", flush=True)
                        print(f"{'='*80}", flush=True)
                        print(answer, flush=True)
                        print(f"{'='*80}\n", flush=True)

                        # ë¶„ë¦¬??ë©”í? ?•ë³´ ?œì‹œ
                        if confidence_info:
                            print(f"\n?’¡ ? ë¢°???•ë³´ (confidence_info ?„ë“œ):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(confidence_info, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        if reference_materials:
                            print(f"\n?“š ì°¸ê³  ?ë£Œ (reference_materials ?„ë“œ):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(reference_materials, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        if disclaimer:
                            print(f"\n?’¼ ë©´ì±… ì¡°í•­ (disclaimer ?„ë“œ):", flush=True)
                            print(f"{'='*80}", flush=True)
                            print(disclaimer, flush=True)
                            print(f"{'='*80}\n", flush=True)

                        # ?µë????Œì¼ë¡??€??(answer ?„ë“œë§??€??
                        import os
                        output_dir = "test_outputs"
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"answer_{i}_{int(time.time())}.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"ì§ˆì˜: {query}\n")
                            f.write(f"{'='*80}\n")
                            f.write(f"?µë? ê¸¸ì´: {len(answer)}??n")
                            f.write(f"? ë¢°?? {confidence:.2%}\n")
                            f.write(f"{'='*80}\n\n")
                            f.write("=== ?µë? ?´ìš© (answer ?„ë“œ) ===\n\n")
                            f.write(answer)

                            # ë¶„ë¦¬??ë©”í? ?•ë³´???¨ê»˜ ?€??
                            if confidence_info:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== ? ë¢°???•ë³´ (confidence_info ?„ë“œ) ===\n\n")
                                f.write(confidence_info)

                            if reference_materials:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== ì°¸ê³  ?ë£Œ (reference_materials ?„ë“œ) ===\n\n")
                                f.write(reference_materials)

                            if disclaimer:
                                f.write(f"\n\n{'='*80}\n")
                                f.write("=== ë©´ì±… ì¡°í•­ (disclaimer ?„ë“œ) ===\n\n")
                                f.write(disclaimer)

                        logger.info(f"   ?µë???{output_file}???€?¥ë˜?ˆìŠµ?ˆë‹¤.")
                        print(f"   ?µë???{output_file}???€?¥ë˜?ˆìŠµ?ˆë‹¤.\n", flush=True)
                    else:
                        logger.info(f"   ?µë? ?€?? {type(answer).__name__}, ê°? {answer}")

                if has_errors:
                    logger.warning(f"\n? ï¸ ?ëŸ¬ ëª©ë¡:")
                    print(f"   ? ï¸ ?ëŸ¬ ëª©ë¡:", flush=True)
                    error_list = errors if isinstance(errors, list) else []
                    for error in error_list[:5]:
                        logger.warning(f"   - {error}")
                        print(f"     - {error}", flush=True)

                if not is_success:
                    # ?¤íŒ¨ ?ì¸ ?ì„¸ ë¶„ì„
                    logger.warning(f"\n? ï¸ ì§ˆì˜ ?¤íŒ¨ ?ì¸ ë¶„ì„:")
                    print(f"   ? ï¸ ì§ˆì˜ ?¤íŒ¨ ?ì¸ ë¶„ì„:", flush=True)
                    if not has_answer:
                        logger.warning(f"   - ?µë????ì„±?˜ì? ?Šì•˜?µë‹ˆ??)
                        print(f"     - ?µë????ì„±?˜ì? ?Šì•˜?µë‹ˆ??, flush=True)
                        logger.warning(f"     - result.get('answer'): {result.get('answer')}")
                        print(f"       result.get('answer'): {result.get('answer')}", flush=True)
                    if has_errors:
                        logger.warning(f"   - ?ëŸ¬ê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤: {len(result.get('errors', []))}ê°?)
                        print(f"     - ?ëŸ¬ê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤: {len(result.get('errors', []))}ê°?, flush=True)

                # ê²°ê³¼ ?€??
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

                logger.error(f"\n???ŒìŠ¤??ì§ˆì˜ ?¤íŒ¨: {query}")
                logger.error(f"?¤ë¥˜ ? í˜•: {type(e).__name__}")
                logger.error(f"?¤ë¥˜ ë©”ì‹œì§€: {str(e)}")
                logger.error(f"?ì„¸ ?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{error_traceback}")

                # stdout?ë„ ì¶œë ¥ (ë²„í¼ë§?ë°©ì?)
                print(f"\n???ŒìŠ¤??ì§ˆì˜ ?¤íŒ¨: {query}", flush=True)
                print(f"?¤ë¥˜ ? í˜•: {type(e).__name__}", flush=True)
                print(f"?¤ë¥˜ ë©”ì‹œì§€: {str(e)}", flush=True)
                print(f"?ì„¸ ?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{error_traceback[:500]}...", flush=True)

                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_traceback
                })

        # ìµœì¢… ê²°ê³¼ ?”ì•½
        logger.info(f"\n{'='*80}")
        logger.info("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        logger.info(f"{'='*80}")

        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("success", False))
        failed_queries = total_queries - successful_queries
        avg_time = sum(r.get("processing_time", 0) for r in results) / total_queries if total_queries > 0 else 0
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total_queries if total_queries > 0 else 0

        logger.info(f"   ì´?ì§ˆì˜ ?? {total_queries}")
        logger.info(f"   ?±ê³µ??ì§ˆì˜: {successful_queries}")
        logger.info(f"   ?¤íŒ¨??ì§ˆì˜: {failed_queries}")
        logger.info(f"   ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.2f}ì´?)
        logger.info(f"   ?‰ê·  ? ë¢°?? {avg_confidence:.2%}")

        # stdout?ë„ ì¶œë ¥
        print(f"\n{'='*80}", flush=True)
        print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"   ì´?ì§ˆì˜ ?? {total_queries}", flush=True)
        print(f"   ?±ê³µ??ì§ˆì˜: {successful_queries}", flush=True)
        print(f"   ?¤íŒ¨??ì§ˆì˜: {failed_queries}", flush=True)
        print(f"   ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.2f}ì´?, flush=True)
        print(f"   ?‰ê·  ? ë¢°?? {avg_confidence:.2%}", flush=True)

        # ?¤íŒ¨??ì§ˆì˜ ?ì„¸ ?•ë³´
        if failed_queries > 0:
            logger.error(f"\n{'='*80}")
            logger.error("?¤íŒ¨??ì§ˆì˜ ?ì„¸ ?•ë³´")
            logger.error(f"{'='*80}")
            print(f"\n{'='*80}", flush=True)
            print("?¤íŒ¨??ì§ˆì˜ ?ì„¸ ?•ë³´", flush=True)
            print(f"{'='*80}", flush=True)

            for i, result in enumerate(results, 1):
                if not result.get("success", False):
                    logger.error(f"\n[{i}] ì§ˆì˜: {result.get('query', '?????†ìŒ')}")
                    print(f"\n[{i}] ì§ˆì˜: {result.get('query', '?????†ìŒ')}", flush=True)

                    # ?µë? ?íƒœ
                    if "has_answer" in result:
                        logger.error(f"   - ?µë? ?ì„±: {'?? if result.get('has_answer') else '?„ë‹ˆ??}")
                        print(f"   - ?µë? ?ì„±: {'?? if result.get('has_answer') else '?„ë‹ˆ??}", flush=True)

                    # Sources ?íƒœ
                    if "has_sources" in result:
                        sources_count = result.get('sources_count', 0)
                        sources_list = result.get('sources_list', [])
                        logger.error(f"   - ?ŒìŠ¤: {'?ˆìŒ' if result.get('has_sources') else '?†ìŒ'} ({sources_count}ê°?")
                        print(f"   - ?ŒìŠ¤: {'?ˆìŒ' if result.get('has_sources') else '?†ìŒ'} ({sources_count}ê°?", flush=True)
                        if sources_list:
                            logger.error(f"   - ?ŒìŠ¤ ëª©ë¡: {', '.join(str(s) for s in sources_list)}")
                            print(f"   - ?ŒìŠ¤ ëª©ë¡: {', '.join(str(s) for s in sources_list)}", flush=True)
                        retrieved_count = result.get('retrieved_docs_count', 0)
                        if retrieved_count > 0:
                            logger.error(f"   - ê²€?‰ëœ ë¬¸ì„œ: {retrieved_count}ê°?)
                            print(f"   - ê²€?‰ëœ ë¬¸ì„œ: {retrieved_count}ê°?, flush=True)

                    # ?ëŸ¬ ?íƒœ
                    if result.get("has_errors"):
                        logger.error(f"   - ?ëŸ¬ ë°œìƒ: ??)
                        logger.error(f"   - ?ëŸ¬ ëª©ë¡: {result.get('errors', [])}")
                        print(f"   - ?ëŸ¬ ë°œìƒ: ??, flush=True)
                        print(f"   - ?ëŸ¬ ëª©ë¡: {result.get('errors', [])}", flush=True)

                    # ?ˆì™¸ ë°œìƒ
                    if "error" in result:
                        logger.error(f"   - ?ˆì™¸ ë°œìƒ: {result.get('error_type', 'Unknown')}")
                        logger.error(f"   - ?¤ë¥˜ ë©”ì‹œì§€: {result.get('error', '?????†ìŒ')}")
                        print(f"   - ?ˆì™¸ ë°œìƒ: {result.get('error_type', 'Unknown')}", flush=True)
                        print(f"   - ?¤ë¥˜ ë©”ì‹œì§€: {result.get('error', '?????†ìŒ')}", flush=True)

                        # ?¤íƒ ?¸ë ˆ?´ìŠ¤ ì¶œë ¥ (?¼ë?ë§?
                        if "traceback" in result:
                            traceback_lines = result["traceback"].split('\n')
                            logger.error(f"   - ?¤íƒ ?¸ë ˆ?´ìŠ¤ (ìµœê·¼ 5ì¤?:")
                            print(f"   - ?¤íƒ ?¸ë ˆ?´ìŠ¤ (ìµœê·¼ 5ì¤?:", flush=True)
                            for line in traceback_lines[-5:]:
                                if line.strip():
                                    logger.error(f"     {line}")
                                    print(f"     {line}", flush=True)

                    # ì²˜ë¦¬ ?œê°„
                    if "processing_time" in result:
                        logger.error(f"   - ì²˜ë¦¬ ?œê°„: {result.get('processing_time', 0):.2f}ì´?)
                        print(f"   - ì²˜ë¦¬ ?œê°„: {result.get('processing_time', 0):.2f}ì´?, flush=True)

        # ?œë¹„???íƒœ ?•ì¸
        logger.info("\n4. ?œë¹„???íƒœ ?•ì¸ ì¤?..")
        status = workflow_service.get_service_status()
        logger.info(f"   ?œë¹„???íƒœ: {status.get('status')}")
        logger.info(f"   ?Œí¬?Œë¡œ??ì»´íŒŒ???¬ë?: {status.get('workflow_compiled')}")

        # ìµœì¢… ?ì •
        logger.info(f"\n{'='*80}")
        print(f"\n{'='*80}", flush=True)

        if successful_queries == total_queries:
            logger.info("??ëª¨ë“  ?ŒìŠ¤???µê³¼! LangGraphê°€ ?•ìƒ?ìœ¼ë¡??™ì‘?©ë‹ˆ??")
            print("??ëª¨ë“  ?ŒìŠ¤???µê³¼! LangGraphê°€ ?•ìƒ?ìœ¼ë¡??™ì‘?©ë‹ˆ??", flush=True)
        elif successful_queries > 0:
            logger.info(f"? ï¸ ë¶€ë¶??±ê³µ: {successful_queries}/{total_queries} ì§ˆì˜ ?±ê³µ")
            print(f"? ï¸ ë¶€ë¶??±ê³µ: {successful_queries}/{total_queries} ì§ˆì˜ ?±ê³µ", flush=True)

            # ë¶€ë¶??±ê³µ ?ì¸ ë¶„ì„
            logger.warning(f"\në¶€ë¶??±ê³µ ë¶„ì„:")
            print(f"\në¶€ë¶??±ê³µ ë¶„ì„:", flush=True)
            for i, result in enumerate(results, 1):
                if not result.get("success"):
                    logger.warning(f"  ì§ˆì˜ {i}: {result.get('query')}")
                    logger.warning(f"    - ?µë?: {'?ˆìŒ' if result.get('has_answer') else '?†ìŒ'}")
                    logger.warning(f"    - ?ŒìŠ¤: {'?ˆìŒ' if result.get('has_sources') else '?†ìŒ'} ({result.get('sources_count', 0)}ê°?")
                    logger.warning(f"    - ?ëŸ¬: {'?ˆìŒ' if result.get('has_errors') else '?†ìŒ'}")
                    print(f"  ì§ˆì˜ {i}: {result.get('query')}", flush=True)
                    print(f"    - ?µë?: {'?ˆìŒ' if result.get('has_answer') else '?†ìŒ'}", flush=True)
                    print(f"    - ?ŒìŠ¤: {'?ˆìŒ' if result.get('has_sources') else '?†ìŒ'} ({result.get('sources_count', 0)}ê°?", flush=True)
                    print(f"    - ?ëŸ¬: {'?ˆìŒ' if result.get('has_errors') else '?†ìŒ'}", flush=True)
        else:
            logger.error("??ëª¨ë“  ?ŒìŠ¤???¤íŒ¨: LangGraph??ë¬¸ì œê°€ ?ˆìŠµ?ˆë‹¤.")
            print("??ëª¨ë“  ?ŒìŠ¤???¤íŒ¨: LangGraph??ë¬¸ì œê°€ ?ˆìŠµ?ˆë‹¤.", flush=True)

            # ?„ì²´ ?¤íŒ¨ ?ì¸ ë¶„ì„
            logger.error(f"\n?„ì²´ ?¤íŒ¨ ?ì¸ ë¶„ì„:")
            print(f"\n?„ì²´ ?¤íŒ¨ ?ì¸ ë¶„ì„:", flush=True)
            for i, result in enumerate(results, 1):
                logger.error(f"  ì§ˆì˜ {i}: {result.get('query')}")
                print(f"  ì§ˆì˜ {i}: {result.get('query')}", flush=True)

                if "error" in result:
                    logger.error(f"    - ?ˆì™¸: {result.get('error_type')} - {result.get('error')}")
                    print(f"    - ?ˆì™¸: {result.get('error_type')} - {result.get('error')}", flush=True)
                else:
                    logger.error(f"    - ?µë?: {'?ˆìŒ' if result.get('has_answer') else '?†ìŒ'}")
                    logger.error(f"    - ?ëŸ¬: {'?ˆìŒ' if result.get('has_errors') else '?†ìŒ'}")
                    if result.get('errors'):
                        logger.error(f"    - ?ëŸ¬ ëª©ë¡: {result.get('errors')}")
                    print(f"    - ?µë?: {'?ˆìŒ' if result.get('has_answer') else '?†ìŒ'}", flush=True)
                    print(f"    - ?ëŸ¬: {'?ˆìŒ' if result.get('has_errors') else '?†ìŒ'}", flush=True)
                    if result.get('errors'):
                        print(f"    - ?ëŸ¬ ëª©ë¡: {result.get('errors')}", flush=True)

        logger.info(f"{'='*80}\n")
        print(f"{'='*80}\n", flush=True)

        return successful_queries == total_queries

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        logger.error(f"{'='*80}")
        logger.error("?ŒìŠ¤???¤í–‰ ì¤?ì¹˜ëª…???¤ë¥˜ ë°œìƒ")
        logger.error(f"{'='*80}")
        logger.error(f"?¤ë¥˜ ? í˜•: {type(e).__name__}")
        logger.error(f"?¤ë¥˜ ë©”ì‹œì§€: {str(e)}")
        logger.error(f"?ì„¸ ?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{error_traceback}")

        # stdout?ë„ ì¶œë ¥
        print(f"\n{'='*80}", flush=True)
        print("?ŒìŠ¤???¤í–‰ ì¤?ì¹˜ëª…???¤ë¥˜ ë°œìƒ", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"?¤ë¥˜ ? í˜•: {type(e).__name__}", flush=True)
        print(f"?¤ë¥˜ ë©”ì‹œì§€: {str(e)}", flush=True)
        print(f"?ì„¸ ?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{error_traceback}", flush=True)

        return False


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # ?ŒìŠ¤???¤í–‰
        result = asyncio.run(test_langgraph_workflow())

        # ê²°ê³¼ ì¶œë ¥ (ë²„í¼ë§?ë°©ì?)
        print(f"\n{'='*80}")
        print(f"ìµœì¢… ?ŒìŠ¤??ê²°ê³¼: {'???±ê³µ' if result else '???¤íŒ¨'}")
        print(f"{'='*80}\n")

        # stdout ë²„í¼ ?ŒëŸ¬??
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        # ì¢…ë£Œ ì½”ë“œ
        sys.exit(0 if result else 1)

    except KeyboardInterrupt:
        logger.info("\n?ŒìŠ¤?¸ê? ?¬ìš©?ì— ?˜í•´ ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        sys.exit(1)
    except Exception as e:
        logger.error(f"?ŒìŠ¤???¤í–‰ ì¤?ì¹˜ëª…???¤ë¥˜: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
