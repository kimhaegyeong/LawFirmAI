# -*- coding: utf-8 -*-
"""
LangGraph ?Œí¬?Œë¡œ???¤í–‰ ?ŒìŠ¤??
UnifiedPromptManager ?µí•© ???¤ì œ ?Œí¬?Œë¡œ???¤í–‰ ê²€ì¦?
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


async def test_workflow_execution():
    """?Œí¬?Œë¡œ???¤í–‰ ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("LangGraph ?Œí¬?Œë¡œ???¤í–‰ ?ŒìŠ¤??)
    print("="*80 + "\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        print("??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        workflow_service = LangGraphWorkflowService(config)
        print("??LangGraphWorkflowService ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì¿¼ë¦¬??
        test_queries = [
            ("?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??, "ê°€ì¡±ë²•"),
            ("ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­?€?", "ë¯¼ì‚¬ë²?),
            ("?´ê³  ?œí•œ ì¡°ê±´?€ ë¬´ì—‡?¸ê???", "?¸ë™ë²?),
            ("?ˆë„ì£„ì˜ ì²˜ë²Œ?€?", "?•ì‚¬ë²?),
            ("ë¯¼ë²• ??50ì¡°ì— ?€???Œë ¤ì£¼ì„¸??, "ë¯¼ì‚¬ë²?ì¡°ë¬¸"),
        ]

        results = []

        for query, description in test_queries:
            print(f"\n?“‹ ?ŒìŠ¤?? {description}")
            print(f"   ì§ˆë¬¸: {query}")

            try:
                # ?Œí¬?Œë¡œ???¤í–‰
                result = await workflow_service.process_query(query)

                # ê²°ê³¼ ê²€ì¦?
                assert "answer" in result
                assert len(result["answer"]) > 0
                assert "confidence" in result
                assert "sources" in result

                # ê²°ê³¼ ì¶œë ¥
                answer_length = len(result["answer"])
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time", 0.0)

                print(f"   ??ì²˜ë¦¬ ?„ë£Œ")
                print(f"   - ?µë? ê¸¸ì´: {answer_length}??)
                print(f"   - ? ë¢°?? {confidence:.2f}")
                print(f"   - ì²˜ë¦¬ ?œê°„: {processing_time:.2f}ì´?)
                print(f"   - ì¶œì²˜: {len(result['sources'])}ê°?)

                # ì²˜ë¦¬ ?¨ê³„ ?•ì¸
                if result.get("processing_steps"):
                    steps = result["processing_steps"]
                    if any("UnifiedPromptManager" in step for step in steps):
                        print(f"   - UnifiedPromptManager ?¬ìš© ?•ì¸??)

                results.append(True)

            except Exception as e:
                print(f"   ??ì²˜ë¦¬ ?¤íŒ¨: {e}")
                results.append(False)

        # ê²°ê³¼ ?”ì•½
        print("\n" + "="*80)
        print("ê²°ê³¼ ?”ì•½")
        print("="*80)

        passed = sum(results)
        total = len(results)
        print(f"\n???±ê³µ: {passed}/{total}")
        print(f"???¤íŒ¨: {total - passed}/{total}")

        if all(results):
            print("\n??ëª¨ë“  ?Œí¬?Œë¡œ???ŒìŠ¤?¸ê? ?±ê³µ?ˆìŠµ?ˆë‹¤!")
            return True
        else:
            print("\n? ï¸ ?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤.")
            return False

    except Exception as e:
        print(f"\n???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """ë¹ ë¥¸ ?ŒìŠ¤???¤í–‰"""
    print("\n" + "="*80)
    print("ë¹ ë¥¸ ?Œí¬?Œë¡œ???ŒìŠ¤??)
    print("="*80 + "\n")

    # ë¹„ë™ê¸??ŒìŠ¤???¤í–‰
    result = asyncio.run(test_workflow_execution())

    return result


if __name__ == "__main__":
    success = run_quick_test()

    if success:
        print("\n" + "="*80)
        print("??ëª¨ë“  ?ŒìŠ¤???„ë£Œ!")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("? ï¸ ?¼ë? ?ŒìŠ¤???¤íŒ¨")
        print("="*80 + "\n")
