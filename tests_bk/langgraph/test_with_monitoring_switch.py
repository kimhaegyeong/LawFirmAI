# -*- coding: utf-8 -*-
"""
LangGraph ëª¨ë‹ˆ?°ë§ ?„í™˜ ?ŒìŠ¤??
LangSmith?€ Langfuseë¥?ë²ˆê°ˆ?„ê?ë©??¬ìš©?˜ëŠ” ?µí•© ?ŒìŠ¤??
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env ?Œì¼ ë¡œë“œ
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(project_root / ".env"))
except ImportError:
    pass

from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch

# WorkflowFactory??? íƒ?ìœ¼ë¡?import
try:
    from tests.langgraph.fixtures.workflow_factory import WorkflowFactory
    WORKFLOW_FACTORY_AVAILABLE = True
except ImportError as e:
    WORKFLOW_FACTORY_AVAILABLE = False
    print(f"??WorkflowFactoryë¥??¬ìš©?????†ìŠµ?ˆë‹¤: {e}")
    print("  langgraph ?¨í‚¤ì§€ê°€ ?¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ??")
    print("  ?Œí¬?Œë¡œ???ŒìŠ¤?¸ë? ê±´ë„ˆ?ë‹ˆ??")


class MonitoringSwitchTest:
    """ëª¨ë‹ˆ?°ë§ ?„í™˜ ?ŒìŠ¤???´ë˜??""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_queries = [
            "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­?€ ë¬´ì—‡?¸ê???",
            "?ë? ê²€?‰ì„ ?„ì?ì£¼ì„¸??,
            "ë²•ë ¹ ?´ì„¤???„ìš”?©ë‹ˆ??
        ]

    async def test_single_mode(self, mode: MonitoringMode, query: str) -> Dict[str, Any]:
        """
        ?¨ì¼ ëª¨ë‹ˆ?°ë§ ëª¨ë“œë¡??ŒìŠ¤???¤í–‰

        Args:
            mode: ëª¨ë‹ˆ?°ë§ ëª¨ë“œ
            query: ?ŒìŠ¤??ì¿¼ë¦¬

        Returns:
            Dict: ?ŒìŠ¤??ê²°ê³¼
        """
        print(f"\n{'='*80}")
        print(f"?ŒìŠ¤??ëª¨ë“œ: {mode.value}")
        print(f"ì¿¼ë¦¬: {query}")
        print(f"{'='*80}")

        result = {
            "mode": mode.value,
            "query": query,
            "success": False,
            "response": None,
            "error": None,
            "verification": None
        }

        try:
            # ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ?¤ì •
            with MonitoringSwitch.set_mode(mode):
                # ?Œí¬?Œë¡œ???œë¹„???ì„±
                if not WORKFLOW_FACTORY_AVAILABLE or not WorkflowFactory.is_available():
                    result["error"] = "WorkflowFactoryë¥??¬ìš©?????†ìŠµ?ˆë‹¤. langgraph ?¨í‚¤ì§€ë¥??¤ì¹˜?˜ì„¸??"
                    result["success"] = False
                    print(f"??{result['error']}")
                    return result

                service = WorkflowFactory.get_workflow(mode, force_recreate=True)

                # ê²€ì¦?
                verification = MonitoringSwitch.verify_mode(service, mode)
                result["verification"] = verification

                if verification.get("warnings"):
                    print(f"??ê²½ê³ : {', '.join(verification['warnings'])}")

                # ì¿¼ë¦¬ ?¤í–‰
                print(f"\nì¿¼ë¦¬ ?¤í–‰ ì¤?..")
                response = await service.process_query(
                    query=query,
                    session_id=f"test_{mode.value}_{hash(query) % 10000}"
                )

                result["response"] = response
                result["success"] = True

                print(f"???ŒìŠ¤???±ê³µ")
                if response.get("answer"):
                    print(f"?µë? ê¸¸ì´: {len(response.get('answer', ''))} ë¬¸ì")

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"???ŒìŠ¤???¤íŒ¨: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def test_switch_between_modes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        ?¬ëŸ¬ ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ê°??„í™˜ ?ŒìŠ¤??

        Returns:
            Dict: ê°?ëª¨ë“œë³??ŒìŠ¤??ê²°ê³¼
        """
        print("\n" + "="*80)
        print("ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ?„í™˜ ?ŒìŠ¤???œì‘")
        print("="*80)

        results_by_mode: Dict[str, List[Dict[str, Any]]] = {}

        # ?ŒìŠ¤?¸í•  ëª¨ë“œ ëª©ë¡
        test_modes = [
            MonitoringMode.LANGSMITH,
            MonitoringMode.LANGFUSE,
            MonitoringMode.BOTH,
            MonitoringMode.NONE
        ]

        for mode in test_modes:
            mode_results = []

            print(f"\n{'='*80}")
            print(f"ëª¨ë‹ˆ?°ë§ ëª¨ë“œ: {mode.value.upper()}")
            print(f"{'='*80}")

            # ê°?ì¿¼ë¦¬ë¡??ŒìŠ¤??
            for query in self.test_queries[:1]:  # ì²?ë²ˆì§¸ ì¿¼ë¦¬ë§?ë¹ ë¥¸ ?ŒìŠ¤??
                result = await self.test_single_mode(mode, query)
                mode_results.append(result)

            results_by_mode[mode.value] = mode_results

            # ìºì‹œ ?•ë¦¬ (?¤ìŒ ëª¨ë“œë¥??„í•´)
            if WORKFLOW_FACTORY_AVAILABLE and WorkflowFactory.is_available():
                WorkflowFactory.clear_cache(mode)

        return results_by_mode

    async def test_sequential_switch(self, query: str) -> List[Dict[str, Any]]:
        """
        ?œì°¨?ìœ¼ë¡?ëª¨ë“œë¥??„í™˜?˜ë©° ?™ì¼ ì¿¼ë¦¬ ?ŒìŠ¤??

        Args:
            query: ?ŒìŠ¤??ì¿¼ë¦¬

        Returns:
            List: ê°?ëª¨ë“œë³??ŒìŠ¤??ê²°ê³¼
        """
        print(f"\n{'='*80}")
        print("?œì°¨ ëª¨ë“œ ?„í™˜ ?ŒìŠ¤??)
        print(f"ì¿¼ë¦¬: {query}")
        print(f"{'='*80}")

        results = []
        modes = [MonitoringMode.LANGSMITH, MonitoringMode.LANGFUSE]

        for mode in modes:
            result = await self.test_single_mode(mode, query)
            results.append(result)

            # ìºì‹œ ?•ë¦¬
            if WORKFLOW_FACTORY_AVAILABLE and WorkflowFactory.is_available():
                WorkflowFactory.clear_cache(mode)

        return results

    def print_results_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
        print("\n" + "="*80)
        print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        print("="*80)

        for mode, mode_results in results.items():
            success_count = sum(1 for r in mode_results if r.get("success"))
            total_count = len(mode_results)

            print(f"\n{mode.upper()}:")
            print(f"  ?±ê³µ: {success_count}/{total_count}")

            for i, result in enumerate(mode_results, 1):
                status = "?? if result.get("success") else "??
                print(f"  {status} ?ŒìŠ¤??{i}: {result.get('query', 'N/A')[:50]}...")
                if result.get("error"):
                    print(f"     ?¤ë¥˜: {result['error']}")
                if result.get("verification", {}).get("warnings"):
                    for warning in result["verification"]["warnings"]:
                        print(f"     ??{warning}")


async def main():
    """ë©”ì¸ ?ŒìŠ¤???¤í–‰"""
    test_runner = MonitoringSwitchTest()

    print("\n" + "="*80)
    print("LangGraph ëª¨ë‹ˆ?°ë§ ?„í™˜ ?ŒìŠ¤??)
    print("="*80)

    # ?„ì¬ ?˜ê²½ë³€???•ì¸
    current_mode = MonitoringSwitch.get_current_mode()
    print(f"\n?„ì¬ ëª¨ë‹ˆ?°ë§ ëª¨ë“œ: {current_mode.value}")

    # ëª¨ë“œ ?„í™˜ ?ŒìŠ¤???¤í–‰
    results = await test_runner.test_switch_between_modes()

    # ê²°ê³¼ ?”ì•½
    test_runner.print_results_summary(results)

    # ?±ê³µë¥?ê³„ì‚°
    total_tests = sum(len(mode_results) for mode_results in results.values())
    total_success = sum(
        sum(1 for r in mode_results if r.get("success"))
        for mode_results in results.values()
    )

    print(f"\n{'='*80}")
    print(f"?„ì²´ ?±ê³µë¥? {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    # ëª…ë ¹ì¤??¸ì ì²˜ë¦¬
    if len(sys.argv) > 1:
        mode_str = sys.argv[1].lower()
        try:
            mode = MonitoringMode.from_string(mode_str)

            # ?¨ì¼ ëª¨ë“œ ?ŒìŠ¤??
            async def test_single():
                test_runner = MonitoringSwitchTest()
                query = test_runner.test_queries[0]
                result = await test_runner.test_single_mode(mode, query)
                print(f"\n?ŒìŠ¤???„ë£Œ: {result.get('success', False)}")
                return result

            asyncio.run(test_single())
        except ValueError:
            print(f"?????†ëŠ” ëª¨ë‹ˆ?°ë§ ëª¨ë“œ: {mode_str}")
            print(f"?¬ìš© ê°€?¥í•œ ëª¨ë“œ: {[m.value for m in MonitoringMode]}")
            sys.exit(1)
    else:
        # ?„ì²´ ?ŒìŠ¤???¤í–‰
        asyncio.run(main())
