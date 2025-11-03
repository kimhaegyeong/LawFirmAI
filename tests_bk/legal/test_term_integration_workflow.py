# -*- coding: utf-8 -*-
"""
?©ì–´ ?µí•© ê¸°ëŠ¥ ?¬í•¨ LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??
TermIntegrationSystem ?µí•© ???Œí¬?Œë¡œ??ê²€ì¦?
"""

import asyncio
import os
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ?ŒìŠ¤???˜ê²½ ?¤ì •
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from source.agents.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


async def test_term_integration_workflow():
    """?©ì–´ ?µí•© ?Œí¬?Œë¡œ???ŒìŠ¤??""
    print("\n" + "="*80)
    print("?©ì–´ ?µí•© LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??)
    print("="*80 + "\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        print("??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        workflow_service = LangGraphWorkflowService(config)
        print("??LangGraphWorkflowService ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì¿¼ë¦¬??- ë²•ë¥  ?©ì–´ê°€ ë§ì? ì§ˆë¬¸??
        test_queries = [
            {
                "query": "?´í˜¼ ?ˆì°¨?€ ?‘ìœ¡ê¶?ë¶„ìŸ???€???Œë ¤ì£¼ì„¸??,
                "description": "ê°€ì¡±ë²• - ?©ì–´ ?µí•© ?ŒìŠ¤??
            },
            {
                "query": "ê³„ì•½???‘ì„± ???í•´ë°°ìƒ ì¡°í•­ê³??„ì•½ê¸?ì¡°í•­??ì°¨ì´?ì??",
                "description": "ë¯¼ì‚¬ë²?- ì¤‘ë³µ ?©ì–´ ?•ë¦¬ ?ŒìŠ¤??
            },
            {
                "query": "?´ê³  ?œí•œ ì¡°ê±´ê³?ë¶€?¹í•´ê³?ë°©ì????€???¤ëª…?´ì£¼?¸ìš”",
                "description": "?¸ë™ë²?- ? ì‚¬ ?©ì–´ ê·¸ë£¹???ŒìŠ¤??
            },
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]

            print(f"\n{'='*80}")
            print(f"?ŒìŠ¤??{i}: {description}")
            print(f"{'='*80}")
            print(f"ì§ˆë¬¸: {query}\n")

            try:
                # ?Œí¬?Œë¡œ???¤í–‰
                result = await workflow_service.process_query(query)

                # ê²°ê³¼ ê²€ì¦?
                assert "answer" in result, "?µë????†ìŠµ?ˆë‹¤"
                assert len(result["answer"]) > 0, "?µë???ë¹„ì–´?ˆìŠµ?ˆë‹¤"
                assert "confidence" in result, "? ë¢°???•ë³´ê°€ ?†ìŠµ?ˆë‹¤"

                # ê²°ê³¼ ì¶œë ¥
                answer_length = len(result["answer"])
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time", 0.0)
                sources_count = len(result.get("sources", []))

                print(f"?“Š ì²˜ë¦¬ ê²°ê³¼:")
                print(f"   - ?µë? ê¸¸ì´: {answer_length}??)
                print(f"   - ? ë¢°?? {confidence:.2f}")
                print(f"   - ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)
                print(f"   - ì¶œì²˜: {sources_count}ê°?)

                # ?©ì–´ ?µí•© ê²°ê³¼ ?•ì¸
                metadata = result.get("metadata", {})
                extracted_terms = metadata.get("extracted_terms", [])
                unique_terms = metadata.get("unique_terms", 0)
                total_terms = metadata.get("total_terms_extracted", 0)

                print(f"\n?“ ?©ì–´ ?µí•© ê²°ê³¼:")
                print(f"   - ì´?ì¶”ì¶œ???©ì–´: {total_terms}ê°?)
                print(f"   - ê³ ìœ  ?©ì–´ (?µí•© ??: {unique_terms}ê°?)
                print(f"   - ì¤‘ë³µ ?œê±°?? {(1 - unique_terms/total_terms)*100:.1f}%" if total_terms > 0 else "   - ì¤‘ë³µ ?œê±°?? N/A")

                if extracted_terms:
                    print(f"\n   ?”‘ ì£¼ìš” ë²•ë¥  ?©ì–´ (ìµœë? 10ê°?:")
                    for j, term in enumerate(extracted_terms[:10], 1):
                        print(f"   {j}. {term}")
                    if len(extracted_terms) > 10:
                        print(f"   ... ??{len(extracted_terms) - 10}ê°?)

                # ì²˜ë¦¬ ?¨ê³„ ?•ì¸
                processing_steps = result.get("processing_steps", [])
                if processing_steps:
                    print(f"\n   ?”„ ì²˜ë¦¬ ?¨ê³„:")
                    for step in processing_steps:
                        if "?©ì–´" in step:
                            print(f"      ??{step}")
                        else:
                            print(f"      ??{step}")

                # ?¤ë¥˜ ?•ì¸
                errors = result.get("errors", [])
                if errors:
                    print(f"\n   ? ï¸ ?¤ë¥˜ ë°œìƒ:")
                    for error in errors:
                        print(f"      - {error}")

                results.append({
                    "success": True,
                    "description": description,
                    "query": query,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "extracted_terms_count": unique_terms,
                    "total_terms": total_terms
                })

                print(f"\n   ???ŒìŠ¤??{i} ?±ê³µ")

            except Exception as e:
                print(f"\n   ???ŒìŠ¤??{i} ?¤íŒ¨: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "success": False,
                    "description": description,
                    "query": query,
                    "error": str(e)
                })

        # ê²°ê³¼ ?”ì•½
        print("\n" + "="*80)
        print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        print("="*80)

        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]

        print(f"\n???±ê³µ: {len(successful_tests)}/{len(results)}")
        print(f"???¤íŒ¨: {len(failed_tests)}/{len(results)}")

        if successful_tests:
            print(f"\n?“Š ?±ê³µ???ŒìŠ¤???µê³„:")
            avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
            avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
            avg_unique_terms = sum(r["extracted_terms_count"] for r in successful_tests) / len(successful_tests)
            avg_total_terms = sum(r["total_terms"] for r in successful_tests) / len(successful_tests)

            print(f"   - ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.2f}ì´?)
            print(f"   - ?‰ê·  ? ë¢°?? {avg_confidence:.2f}")
            print(f"   - ?‰ê·  ê³ ìœ  ?©ì–´: {avg_unique_terms:.0f}ê°?)
            print(f"   - ?‰ê·  ì´??©ì–´: {avg_total_terms:.0f}ê°?)

        if failed_tests:
            print(f"\n???¤íŒ¨???ŒìŠ¤??")
            for test in failed_tests:
                print(f"   - {test['description']}: {test.get('error', 'Unknown error')}")

        if all(r["success"] for r in results):
            print("\n" + "="*80)
            print("??ëª¨ë“  ?ŒìŠ¤?¸ê? ?±ê³µ?ˆìŠµ?ˆë‹¤!")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("? ï¸ ?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤.")
            print("="*80 + "\n")
            return False

    except Exception as e:
        print(f"\n???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_single_query_detailed():
    """?¨ì¼ ì¿¼ë¦¬???€???ì„¸ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?ì„¸ ?¨ì¼ ì¿¼ë¦¬ ?ŒìŠ¤??)
    print("="*80 + "\n")

    try:
        config = LangGraphConfig.from_env()
        workflow_service = LangGraphWorkflowService(config)

        query = "?´í˜¼ ?ˆì°¨?€ ?‘ìœ¡ê¶?ë¶„ìŸ, ?ì† ë¬¸ì œ???€???ì„¸???Œë ¤ì£¼ì„¸??

        print(f"?ŒìŠ¤??ì§ˆë¬¸: {query}\n")

        result = await workflow_service.process_query(query)

        print("\n" + "="*80)
        print("?Œí¬?Œë¡œ??ì²˜ë¦¬ ê²°ê³¼")
        print("="*80)
        print(f"\n?µë?:\n{result['answer']}\n")

        # ë©”í??°ì´???ì„¸ ì¶œë ¥
        metadata = result.get("metadata", {})
        print(f"ë©”í??°ì´??")
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  - {key}: {len(value)}ê°???ª©")
            else:
                print(f"  - {key}: {value}")

        print(f"\nì²˜ë¦¬ ?¨ê³„:")
        for step in result.get("processing_steps", []):
            print(f"  ??{step}")

        if result.get("errors"):
            print(f"\n?¤ë¥˜:")
            for error in result["errors"]:
                print(f"  ? ï¸ {error}")

        return True

    except Exception as e:
        print(f"\n???ŒìŠ¤??ì¤??¤ë¥˜: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """?„ì²´ ?ŒìŠ¤???¤í–‰"""
    print("\n" + "="*80)
    print("?©ì–´ ?µí•© ?Œí¬?Œë¡œ???ŒìŠ¤???œì‘")
    print("="*80)

    # ë¹„ë™ê¸??ŒìŠ¤???¤í–‰
    result1 = asyncio.run(test_term_integration_workflow())
    result2 = asyncio.run(test_single_query_detailed())

    success = result1 and result2

    print("\n" + "="*80)
    if success:
        print("??ëª¨ë“  ?ŒìŠ¤???„ë£Œ!")
    else:
        print("? ï¸ ?¼ë? ?ŒìŠ¤???¤íŒ¨")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
