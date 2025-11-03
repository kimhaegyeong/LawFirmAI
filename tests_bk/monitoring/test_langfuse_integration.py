# -*- coding: utf-8 -*-
"""
Langfuse ?µí•© ?ŒìŠ¤??
ê°œì„ ??LangGraph ?Œí¬?Œë¡œ?°ë? Langfuseë¡?ëª¨ë‹ˆ?°ë§?˜ëŠ” ?ŒìŠ¤??
"""

import asyncio
import os
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ?ŒìŠ¤???˜ê²½ ?¤ì •
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGGRAPH_CHECKPOINT_STORAGE"] = "memory"  # ë¹ ë¥¸ ?ŒìŠ¤?¸ë? ?„í•´ ë©”ëª¨ë¦??¬ìš©

from source.agents.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


class LangfuseIntegrationTest:
    """Langfuse ?µí•© ?ŒìŠ¤???´ë˜??""

    def __init__(self):
        self.config = LangGraphConfig()
        self.service = LangGraphWorkflowService(self.config)

    async def run_single_query(self, query: str, session_id: str, query_number: int):
        """?¨ì¼ ì§ˆì˜ ?¤í–‰"""
        print(f"\n{'='*80}")
        print(f"ì§ˆì˜ #{query_number}: {query}")
        print(f"{'='*80}")

        try:
            # LangGraph ?Œí¬?Œë¡œ???¤í–‰
            result = await self.service.process_query(
                query=query,
                session_id=session_id
            )

            # ê²°ê³¼ ì¶œë ¥
            print(f"\n??ì²˜ë¦¬ ?„ë£Œ")
            print(f"  ?µë? ê¸¸ì´: {len(result.get('answer', ''))}??)
            print(f"  ? ë¢°?? {result.get('confidence', 0):.2%}")
            print(f"  ê²€?‰ëœ ë¬¸ì„œ ?? {len(result.get('retrieved_docs', []))}ê°?)
            print(f"  ì²˜ë¦¬ ?¨ê³„ ?? {len(result.get('processing_steps', []))}ê°?)

            # ?¤ì›Œ???•ì¥ ?•ë³´
            if result.get('ai_keyword_expansion'):
                expansion = result['ai_keyword_expansion']
                print(f"  AI ?¤ì›Œ???•ì¥: {expansion.get('method')}")
                print(f"    - ?ë³¸ ?¤ì›Œ?? {len(expansion.get('original_keywords', []))}ê°?)
                print(f"    - ?•ì¥ ?¤ì›Œ?? {len(expansion.get('expanded_keywords', []))}ê°?)
                print(f"    - ? ë¢°?? {expansion.get('confidence', 0):.2%}")

            # ì²˜ë¦¬ ?¨ê³„ ì¶œë ¥
            steps = result.get('processing_steps', [])
            if steps:
                print(f"\n  ì²˜ë¦¬ ?¨ê³„:")
                for i, step in enumerate(steps[-5:], 1):  # ë§ˆì?ë§?5ê°œë§Œ
                    print(f"    {i}. {step}")

            return result

        except Exception as e:
            print(f"\n??ì²˜ë¦¬ ?¤íŒ¨: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_all_tests(self):
        """?„ì²´ ?ŒìŠ¤???¤í–‰"""
        print("=" * 80)
        print("Langfuse ?µí•© ?ŒìŠ¤???œì‘")
        print("ê°œì„ ??LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??)
        print("=" * 80)

        # ?ŒìŠ¤??ì¼€?´ìŠ¤ ?•ì˜
        test_cases = [
            {
                "query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                "session_id": "test_session_001",
                "description": "ê¸°ë³¸ ë²•ë¥  ì¡°ì–¸ ì§ˆë¬¸"
            },
            {
                "query": "ê³„ì•½ ?„ë°˜ ??ë²•ì  ì¡°ì¹˜ ë°©ë²•",
                "session_id": "test_session_002",
                "description": "ê³„ì•½ ê´€??ì§ˆë¬¸"
            },
            {
                "query": "ë¯¼ì‚¬?Œì†¡?ì„œ ?¹ì†Œ?˜ê¸° ?„í•œ ì¦ê±° ?˜ì§‘ ë°©ë²•",
                "session_id": "test_session_003",
                "description": "ë¯¼ì‚¬?Œì†¡ ?ˆì°¨ ì§ˆë¬¸"
            },
            {
                "query": "ê³„ì•½?œì— ?°ë¥´ë©?ë°°ì†¡ ì§€?????´ë–»ê²??€?‘í•´???˜ë‚˜??",
                "session_id": "test_session_004",
                "description": "êµ¬ì²´???¬ì•ˆ ì§ˆë¬¸"
            },
            {
                "query": "?´ì „???Œê°œ?´ì£¼???í•´ë°°ìƒ ì²?µ¬?ì„œ ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²?ê²°ì •?˜ë‚˜??",
                "session_id": "test_session_005",
                "description": "ë©€?°í„´ ì§ˆë¬¸ (?´ì „ ì§ˆë¬¸ ì°¸ì¡°)"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n\n{'#'*80}")
            print(f"?ŒìŠ¤??ì¼€?´ìŠ¤ #{i}/{len(test_cases)}")
            print(f"?¤ëª…: {test_case['description']}")
            print(f"{'#'*80}")

            result = await self.run_single_query(
                query=test_case['query'],
                session_id=test_case['session_id'],
                query_number=i
            )

            if result:
                results.append({
                    'case': i,
                    'query': test_case['query'],
                    'result': result
                })

            # ê°??ŒìŠ¤???¬ì´??ì§§ì? ?€ê¸?
            await asyncio.sleep(1)

        # ?µê³„ ì¶œë ¥
        print("\n" + "=" * 80)
        print("?ŒìŠ¤??ê²°ê³¼ ?µê³„")
        print("=" * 80)

        total_queries = len(test_cases)
        successful_queries = len(results)

        print(f"\nì´?ì§ˆì˜ ?? {total_queries}")
        print(f"?±ê³µ??ì§ˆì˜: {successful_queries}")
        print(f"?¤íŒ¨??ì§ˆì˜: {total_queries - successful_queries}")

        if results:
            # ?‰ê·  ê°?ê³„ì‚°
            total_confidence = sum(r['result'].get('confidence', 0) for r in results)
            total_docs = sum(len(r['result'].get('retrieved_docs', [])) for r in results)
            total_steps = sum(len(r['result'].get('processing_steps', [])) for r in results)

            print(f"\n?‰ê·  ? ë¢°?? {total_confidence/successful_queries:.2%}")
            print(f"?‰ê·  ê²€??ë¬¸ì„œ ?? {total_docs/successful_queries:.1f}ê°?)
            print(f"?‰ê·  ì²˜ë¦¬ ?¨ê³„ ?? {total_steps/successful_queries:.1f}ê°?)

            # AI ?¤ì›Œ???•ì¥ ?µê³„
            ai_expansions = [r for r in results if r['result'].get('ai_keyword_expansion')]
            if ai_expansions:
                print(f"\nAI ?¤ì›Œ???•ì¥ ?¤í–‰: {len(ai_expansions)}??)
                gemini_count = len([r for r in ai_expansions
                                     if r['result']['ai_keyword_expansion'].get('method') == 'gemini_ai'])
                fallback_count = len([r for r in ai_expansions
                                       if r['result']['ai_keyword_expansion'].get('method') == 'fallback'])
                print(f"  - Gemini AI: {gemini_count}??)
                print(f"  - Fallback: {fallback_count}??)

        print("\n" + "=" * 80)
        print("Langfuse ëª¨ë‹ˆ?°ë§ ?•ì¸")
        print("=" * 80)
        print("\nLangfuse ?€?œë³´?œì—???¤ìŒ ?•ë³´ë¥??•ì¸?????ˆìŠµ?ˆë‹¤:")
        print("  - ê°??¸ë“œ???¤í–‰ ?œê°„")
        print("  - ?¸ë“œ ê°??°ì´???ë¦„")
        print("  - AI ?¤ì›Œ???•ì¥ ê³¼ì •")
        print("  - ?ëŸ¬ ë°?ê²½ê³  ë©”ì‹œì§€")
        print("\nLangfuse URL: http://localhost:3000 (ë¡œì»¬ ?¤ì •??ê²½ìš°)")

        return results


async def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    test_runner = LangfuseIntegrationTest()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 80)
    print("?ŒìŠ¤???„ë£Œ!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Langfuse ?¤ì • ?•ì¸
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse()
        print("??Langfuse ì´ˆê¸°???±ê³µ")
    except Exception as e:
        print(f"??Langfuse ì´ˆê¸°???¤íŒ¨: {e}")
        print("  Langfuse ?†ì´???ŒìŠ¤?¸ëŠ” ì§„í–‰?©ë‹ˆ??")

    # ?ŒìŠ¤???¤í–‰
    asyncio.run(main())
