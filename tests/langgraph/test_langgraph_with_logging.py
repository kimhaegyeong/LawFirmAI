# -*- coding: utf-8 -*-
"""
LangGraph ?™ì‘ ?ŒìŠ¤???¤í¬ë¦½íŠ¸ (?Œì¼ ë¡œê¹… ?¬í•¨)
ë¦¬íŒ©? ë§ ??LangGraphê°€ ?•ìƒ?ìœ¼ë¡??™ì‘?˜ëŠ”ì§€ ?•ì¸?˜ê³  ?ì„¸ ë¡œê·¸ ?€??
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# ?Œì¼ ë¡œê±° ?¤ì •
log_file = log_dir / f"test_langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# ì½˜ì†” ?¸ë“¤???¤ì •
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# ë£¨íŠ¸ ë¡œê±° ?¤ì •
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Windows ë¹„ë™ê¸??˜ê²½?ì„œ ë¡œê¹… ë²„í¼ ?ëŸ¬ ë°©ì?
class SafeStreamHandler(logging.StreamHandler):
    """?ˆì „???¤íŠ¸ë¦??¸ë“¤??- detached ë²„í¼ ?ëŸ¬ ë°©ì?"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            pass

logging.raiseExceptions = False


async def test_langgraph_workflow_with_logging():
    """LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??(?ì„¸ ë¡œê¹… ?¬í•¨)"""
    try:
        # Import ê²½ë¡œ ?•ì¸ ë°?ì¡°ì •
        # core/agents/workflow_service.pyë¥??¬ìš©?˜ë„ë¡?ë³€ê²?
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        logger.info("=" * 80)
        logger.info("LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤???œì‘ (?Œì¼ ë¡œê¹… ?¬í•¨)")
        logger.info("=" * 80)
        logger.info(f"ë¡œê·¸ ?Œì¼: {log_file}")

        # ?¤ì • ë¡œë“œ
        logger.info("1. LangGraph ?¤ì • ë¡œë“œ ì¤?..")
        config = LangGraphConfig.from_env()
        logger.info(f"   ??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ (enabled={config.langgraph_enabled})")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??(?Œì¼ ë¡œê¹… ?œì„±??
        logger.info("2. ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??ì¤?..")
        start_time = time.time()
        workflow_service = LangGraphWorkflowService(config, enable_file_logging=True)
        init_time = time.time() - start_time
        logger.info(f"   ???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ ({init_time:.2f}ì´?")

        # ?ŒìŠ¤??ì§ˆì˜ (ë¯¼ì‚¬ë²?ê´€?? 1ê°?
        test_queries = [
            "ë¯¼ì‚¬ë²•ì—??ê³„ì•½ ?´ì? ?”ê±´?€ ë¬´ì—‡?¸ê???"
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

                # ê²°ê³¼ ê²€ì¦?
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

                answer = str(answer) if not isinstance(answer, str) else answer
                has_answer = bool(answer) and len(answer) > 0

                # Sources ?•ì¸
                sources = result.get("sources", []) if isinstance(result, dict) else []
                retrieved_docs = result.get("retrieved_docs", []) if isinstance(result, dict) else []

                has_sources = len(sources) > 0
                sources_count = len(sources)
                retrieved_docs_count = len(retrieved_docs)

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
                logger.info(f"   - ?ŒìŠ¤ ? ë¬´: {'?ˆìŒ' if has_sources else '?†ìŒ'} ({sources_count}ê°?")
                logger.info(f"   - ê²€?‰ëœ ë¬¸ì„œ: {retrieved_docs_count}ê°?)
                logger.info(f"   - ? ë¢°?? {confidence:.2%}")
                logger.info(f"   - ?ëŸ¬ ? ë¬´: {'?ˆìŒ' if has_errors else '?†ìŒ'}")

                # ë¡œê·¸ ?Œì¼ ê²½ë¡œ ?œì‹œ
                log_file_path = result.get("log_file", "")
                if log_file_path:
                    logger.info(f"   - ?ì„¸ ë¡œê·¸: {log_file_path}")

                # stdout?ë„ ì¶œë ¥
                print(f"\n{result_status} ì§ˆì˜ {i}/{len(test_queries)}: {query} (ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?", flush=True)
                print(f"   - ?µë? ? ë¬´: {'?ˆìŒ' if has_answer else '?†ìŒ'}", flush=True)
                print(f"   - ?µë? ê¸¸ì´: {answer_length}??, flush=True)
                print(f"   - ?ŒìŠ¤ ? ë¬´: {'?ˆìŒ' if has_sources else '?†ìŒ'} ({sources_count}ê°?", flush=True)
                print(f"   - ê²€?‰ëœ ë¬¸ì„œ: {retrieved_docs_count}ê°?, flush=True)
                print(f"   - ? ë¢°?? {confidence:.2%}", flush=True)
                print(f"   - ?ëŸ¬ ? ë¬´: {'?ˆìŒ' if has_errors else '?†ìŒ'}", flush=True)
                if log_file_path:
                    print(f"   - ?ì„¸ ë¡œê·¸: {log_file_path}", flush=True)

                if has_answer:
                    logger.info(f"\n?“ ?µë? ë¯¸ë¦¬ë³´ê¸°:")
                    if isinstance(answer, str):
                        answer_preview = answer[:200]
                        logger.info(f"   {answer_preview}{'...' if len(answer) > 200 else ''}")

                if has_errors:
                    logger.warning(f"\n? ï¸ ?ëŸ¬ ëª©ë¡:")
                    error_list = errors if isinstance(errors, list) else []
                    for error in error_list[:5]:
                        logger.warning(f"   - {error}")

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
                    "retrieved_docs_count": retrieved_docs_count,
                    "has_errors": has_errors,
                    "errors": errors if isinstance(errors, list) else [],
                    "log_file": log_file_path
                }
                results.append(test_result)

            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()

                logger.error(f"\n???ŒìŠ¤??ì§ˆì˜ ?¤íŒ¨: {query}")
                logger.error(f"?¤ë¥˜ ? í˜•: {type(e).__name__}")
                logger.error(f"?¤ë¥˜ ë©”ì‹œì§€: {str(e)}")
                logger.error(f"?ì„¸ ?¤íƒ ?¸ë ˆ?´ìŠ¤:\n{error_traceback}")

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
        logger.info(f"\n   ?ì„¸ ë¡œê·¸ ?Œì¼: {log_file}")

        print(f"\n{'='*80}", flush=True)
        print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"   ì´?ì§ˆì˜ ?? {total_queries}", flush=True)
        print(f"   ?±ê³µ??ì§ˆì˜: {successful_queries}", flush=True)
        print(f"   ?¤íŒ¨??ì§ˆì˜: {failed_queries}", flush=True)
        print(f"   ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.2f}ì´?, flush=True)
        print(f"   ?‰ê·  ? ë¢°?? {avg_confidence:.2%}", flush=True)
        print(f"\n   ?ì„¸ ë¡œê·¸ ?Œì¼: {log_file}", flush=True)

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

        return False


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        result = asyncio.run(test_langgraph_workflow_with_logging())

        print(f"\n{'='*80}")
        print(f"ìµœì¢… ?ŒìŠ¤??ê²°ê³¼: {'???±ê³µ' if result else '???¤íŒ¨'}")
        print(f"{'='*80}\n")

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
