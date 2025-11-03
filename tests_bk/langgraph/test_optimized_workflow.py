# -*- coding: utf-8 -*-
"""
ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤??(Adaptive RAG + ê·¸ë˜???¨ìˆœ??+ ë³‘ë ¬ ?¤í–‰)
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


class WorkflowPerformanceTester:
    """?Œí¬?Œë¡œ???±ëŠ¥ ?ŒìŠ¤??""

    def __init__(self):
        """?ŒìŠ¤??ì´ˆê¸°??""
        config = LangGraphConfig.from_env()
        self.workflow_service = LangGraphWorkflowService(config)
        self.results = []

    async def test_simple_query(self):
        """ê°„ë‹¨??ì§ˆë¬¸ ?ŒìŠ¤??(Adaptive RAG - ê²€???¤í‚µ)"""
        print("\n" + "=" * 80)
        print("?ŒìŠ¤??1: ê°„ë‹¨??ì§ˆë¬¸ (Adaptive RAG - ê²€???¤í‚µ)")
        print("=" * 80)

        test_queries = [
            "?ˆë…•?˜ì„¸??,
            "ê³ ë§ˆ?Œìš”",
            "ê³„ì•½?´ë? ë¬´ì—‡?¸ê???",
            "ë²•ë¥  ?©ì–´ '?Œì†¡'???˜ë?ë¥??Œë ¤ì£¼ì„¸??
        ]

        for query in test_queries:
            print(f"\n?“ ì§ˆë¬¸: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # ê²°ê³¼ ë¶„ì„
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)

                print(f"  ?±ï¸  ?‘ë‹µ ?œê°„: {elapsed_time:.2f}ì´?)
                print(f"  ?“Š ë³µì¡?? {query_complexity}")
                print(f"  ?” ê²€???„ìš”: {needs_search}")
                print(f"  ?“„ ?µë? ê¸¸ì´: {len(answer)}??)
                print(f"  ?“š ?ŒìŠ¤ ?? {len(sources)}ê°?)
                print(f"  ?”„ ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)

                # ê²€???¤í‚µ ?•ì¸
                if query_complexity == "simple":
                    if needs_search == False:
                        print("  ??ê²€???¤í‚µ ?•ì¸??(Adaptive RAG ?‘ë™)")
                    else:
                        print("  ? ï¸  ê²€???¤í‚µ ?ˆìƒ?˜ì—ˆ?¼ë‚˜ ?¤í–‰??)
                else:
                    print(f"  ? ï¸  ê°„ë‹¨??ì§ˆë¬¸?¼ë¡œ ë¶„ë¥˜?˜ì? ?ŠìŒ (ë³µì¡?? {query_complexity})")

                # ?µë? ?´ìš© ë¯¸ë¦¬ë³´ê¸°
                if answer:
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"  ?’¬ ?µë? ë¯¸ë¦¬ë³´ê¸°: {preview}")

                self.results.append({
                    "query": query,
                    "type": "simple",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
                logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")

    async def test_complexity_classification(self):
        """ë³µì¡??ë¶„ë¥˜ ?ŒìŠ¤??""
        print("\n" + "=" * 80)
        print("?ŒìŠ¤?? ë³µì¡??ë¶„ë¥˜ ?•ì¸")
        print("=" * 80)

        test_cases = [
            ("?ˆë…•?˜ì„¸??, "simple"),
            ("ê³„ì•½?´ë? ë¬´ì—‡?¸ê???", "simple"),
            ("ë¯¼ë²• ??11ì¡°ì˜ ?´ìš©???Œë ¤ì£¼ì„¸??, "moderate"),
        ]

        passed = 0
        failed = 0

        for query, expected_complexity in test_cases:
            print(f"\n?“ ì§ˆë¬¸: {query}")
            print(f"  ?ˆìƒ ë³µì¡?? {expected_complexity}")

            try:
                result = await self.workflow_service.process_query(query)
                actual_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)

                print(f"  ?¤ì œ ë³µì¡?? {actual_complexity}")
                print(f"  ê²€???„ìš”: {needs_search}")

                if actual_complexity == expected_complexity:
                    print("  ???¬ë°”ë¥´ê²Œ ë¶„ë¥˜??)
                    passed += 1
                else:
                    print(f"  ? ï¸  ë³µì¡??ë¶ˆì¼ì¹?(?ˆìƒ: {expected_complexity}, ?¤ì œ: {actual_complexity})")
                    failed += 1

                self.results.append({
                    "query": query,
                    "type": "classification",
                    "expected": expected_complexity,
                    "actual": actual_complexity,
                    "passed": actual_complexity == expected_complexity
                })

            except Exception as e:
                print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
                logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")
                failed += 1

        print(f"\n?“Š ê²°ê³¼: {passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")
        return failed == 0

    async def test_moderate_query(self):
        """ì¤‘ê°„ ë³µì¡??ì§ˆë¬¸ ?ŒìŠ¤??(ê²€???˜í–‰)"""
        print("\n" + "=" * 80)
        print("?ŒìŠ¤??2: ì¤‘ê°„ ë³µì¡??ì§ˆë¬¸ (ê²€???˜í–‰)")
        print("=" * 80)

        test_queries = [
            "ë¯¼ë²• ??11ì¡°ì˜ ?´ìš©???Œë ¤ì£¼ì„¸??,
            "ê³„ì•½ ?´ì? ì¡°ê±´?€ ë¬´ì—‡?¸ê???",
            "?´í˜¼ ?Œì†¡ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??
        ]

        for query in test_queries:
            print(f"\n?“ ì§ˆë¬¸: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # ê²°ê³¼ ë¶„ì„
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)
                retrieved_docs = result.get("retrieved_docs", [])

                print(f"  ?±ï¸  ?‘ë‹µ ?œê°„: {elapsed_time:.2f}ì´?)
                print(f"  ?“Š ë³µì¡?? {query_complexity}")
                print(f"  ?” ê²€???„ìš”: {needs_search}")
                print(f"  ?“„ ?µë? ê¸¸ì´: {len(answer)}??)
                print(f"  ?“š ?ŒìŠ¤ ?? {len(sources)}ê°?)
                print(f"  ?“– ê²€??ë¬¸ì„œ ?? {len(retrieved_docs)}ê°?)
                print(f"  ?”„ ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)

                # ê²€???˜í–‰ ?•ì¸
                if needs_search == True and len(retrieved_docs) > 0:
                    print("  ??ê²€???˜í–‰ ?•ì¸??)
                elif needs_search == True and len(retrieved_docs) == 0:
                    print("  ? ï¸  ê²€???„ìš”?ˆìœ¼??ë¬¸ì„œë¥?ì°¾ì? ëª»í•¨")

                self.results.append({
                    "query": query,
                    "type": "moderate",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "retrieved_docs_count": len(retrieved_docs),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
                logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")

    async def test_complex_query(self):
        """ë³µì¡??ì§ˆë¬¸ ?ŒìŠ¤??(?„ì²´ ?Œë¡œ??"""
        print("\n" + "=" * 80)
        print("?ŒìŠ¤??3: ë³µì¡??ì§ˆë¬¸ (?„ì²´ ?Œë¡œ??")
        print("=" * 80)

        test_queries = [
            "?´í˜¼ê³??¬í˜¼??ì°¨ì´?ê³¼ ê°ê°??ë²•ì  ?ˆì°¨ë¥?ë¹„êµ?´ì£¼?¸ìš”",
            "ê³„ì•½ ?´ì??€ ?´ì œ??ì°¨ì´??ë¬´ì—‡?¸ê???",
            "ìµœê·¼ ?ë?ë¥?ë°”íƒ•?¼ë¡œ ?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???¤ëª…?´ì£¼?¸ìš”"
        ]

        for query in test_queries:
            print(f"\n?“ ì§ˆë¬¸: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # ê²°ê³¼ ë¶„ì„
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)
                retrieved_docs = result.get("retrieved_docs", [])

                print(f"  ?±ï¸  ?‘ë‹µ ?œê°„: {elapsed_time:.2f}ì´?)
                print(f"  ?“Š ë³µì¡?? {query_complexity}")
                print(f"  ?” ê²€???„ìš”: {needs_search}")
                print(f"  ?“„ ?µë? ê¸¸ì´: {len(answer)}??)
                print(f"  ?“š ?ŒìŠ¤ ?? {len(sources)}ê°?)
                print(f"  ?“– ê²€??ë¬¸ì„œ ?? {len(retrieved_docs)}ê°?)
                print(f"  ?”„ ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)

                # ì²˜ë¦¬ ?¨ê³„ ?ì„¸
                if processing_steps:
                    print(f"  ?“‹ ì²˜ë¦¬ ?¨ê³„:")
                    for idx, step in enumerate(processing_steps[:10], 1):  # ìµœë? 10ê°œë§Œ
                        print(f"     {idx}. {step}")

                self.results.append({
                    "query": query,
                    "type": "complex",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "retrieved_docs_count": len(retrieved_docs),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
                logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")

    def print_summary(self):
        """?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
        print("\n" + "=" * 80)
        print("?“Š ?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        print("=" * 80)

        if not self.results:
            print("???ŒìŠ¤??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤.")
            return

        # ? í˜•ë³??µê³„
        simple_results = [r for r in self.results if r["type"] == "simple"]
        moderate_results = [r for r in self.results if r["type"] == "moderate"]
        complex_results = [r for r in self.results if r["type"] == "complex"]

        print(f"\n?“ˆ ?„ì²´ ?µê³„:")
        print(f"  - ì´??ŒìŠ¤?? {len(self.results)}ê°?)
        print(f"  - ê°„ë‹¨??ì§ˆë¬¸: {len(simple_results)}ê°?)
        print(f"  - ì¤‘ê°„ ì§ˆë¬¸: {len(moderate_results)}ê°?)
        print(f"  - ë³µì¡??ì§ˆë¬¸: {len(complex_results)}ê°?)

        # ?±ëŠ¥ ?µê³„
        if simple_results:
            avg_time = sum(r["elapsed_time"] for r in simple_results) / len(simple_results)
            min_time = min(r["elapsed_time"] for r in simple_results)
            max_time = max(r["elapsed_time"] for r in simple_results)
            print(f"\n??ê°„ë‹¨??ì§ˆë¬¸ ?±ëŠ¥:")
            print(f"  - ?‰ê·  ?‘ë‹µ ?œê°„: {avg_time:.2f}ì´?)
            print(f"  - ìµœì†Œ ?œê°„: {min_time:.2f}ì´?)
            print(f"  - ìµœë? ?œê°„: {max_time:.2f}ì´?)
            search_skipped = sum(1 for r in simple_results if r.get("needs_search") == False)
            print(f"  - ê²€???¤í‚µë¥? {search_skipped}/{len(simple_results)} ({search_skipped/len(simple_results)*100:.1f}%)")

        if moderate_results:
            avg_time = sum(r["elapsed_time"] for r in moderate_results) / len(moderate_results)
            min_time = min(r["elapsed_time"] for r in moderate_results)
            max_time = max(r["elapsed_time"] for r in moderate_results)
            print(f"\n??ì¤‘ê°„ ì§ˆë¬¸ ?±ëŠ¥:")
            print(f"  - ?‰ê·  ?‘ë‹µ ?œê°„: {avg_time:.2f}ì´?)
            print(f"  - ìµœì†Œ ?œê°„: {min_time:.2f}ì´?)
            print(f"  - ìµœë? ?œê°„: {max_time:.2f}ì´?)
            avg_docs = sum(r.get("retrieved_docs_count", 0) for r in moderate_results) / len(moderate_results)
            print(f"  - ?‰ê·  ê²€??ë¬¸ì„œ ?? {avg_docs:.1f}ê°?)

        if complex_results:
            avg_time = sum(r["elapsed_time"] for r in complex_results) / len(complex_results)
            min_time = min(r["elapsed_time"] for r in complex_results)
            max_time = max(r["elapsed_time"] for r in complex_results)
            print(f"\n??ë³µì¡??ì§ˆë¬¸ ?±ëŠ¥:")
            print(f"  - ?‰ê·  ?‘ë‹µ ?œê°„: {avg_time:.2f}ì´?)
            print(f"  - ìµœì†Œ ?œê°„: {min_time:.2f}ì´?)
            print(f"  - ìµœë? ?œê°„: {max_time:.2f}ì´?)
            avg_docs = sum(r.get("retrieved_docs_count", 0) for r in complex_results) / len(complex_results)
            print(f"  - ?‰ê·  ê²€??ë¬¸ì„œ ?? {avg_docs:.1f}ê°?)
            avg_steps = sum(r.get("steps_count", 0) for r in complex_results) / len(complex_results)
            print(f"  - ?‰ê·  ì²˜ë¦¬ ?¨ê³„ ?? {avg_steps:.1f}ê°?)

        # ?„ì²´ ?‰ê· 
        if self.results:
            avg_time_all = sum(r["elapsed_time"] for r in self.results if "elapsed_time" in r) / len([r for r in self.results if "elapsed_time" in r])
            print(f"\n?“Š ?„ì²´ ?‰ê·  ?‘ë‹µ ?œê°„: {avg_time_all:.2f}ì´?)

        print("\n" + "=" * 80)


async def main():
    """ë©”ì¸ ?ŒìŠ¤???¤í–‰"""
    print("=" * 80)
    print("?? ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤???œì‘")
    print("=" * 80)
    print("\n?ŒìŠ¤????ª©:")
    print("  1. ê°„ë‹¨??ì§ˆë¬¸ (Adaptive RAG - ê²€???¤í‚µ)")
    print("  2. ì¤‘ê°„ ë³µì¡??ì§ˆë¬¸ (ê²€???˜í–‰)")
    print("  3. ë³µì¡??ì§ˆë¬¸ (?„ì²´ ?Œë¡œ??")
    print("  4. ë³µì¡??ë¶„ë¥˜ ?ŒìŠ¤??)
    print("  5. ?±ëŠ¥ ì¸¡ì • ë°??”ì•½")

    tester = WorkflowPerformanceTester()

    try:
        # ?ŒìŠ¤???¤í–‰
        await tester.test_simple_query()
        await tester.test_moderate_query()
        await tester.test_complex_query()
        await tester.test_complexity_classification()

        # ê²°ê³¼ ?”ì•½
        tester.print_summary()

        print("\n??ëª¨ë“  ?ŒìŠ¤???„ë£Œ!")

    except KeyboardInterrupt:
        print("\n? ï¸  ?ŒìŠ¤?¸ê? ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
    except Exception as e:
        print(f"\n???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
        logger.exception("?ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜")


if __name__ == "__main__":
    asyncio.run(main())
