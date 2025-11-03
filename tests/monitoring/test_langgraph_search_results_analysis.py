#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph ?Œí¬?Œë¡œ?°ì—??ê²€??ê²°ê³¼ ?¬í•¨ ë¶„ì„
?¤ì œ ?Œí¬?Œë¡œ???¤í–‰ ë°?ë¡œê·¸ ë¶„ì„
"""

import logging
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_langgraph_search_results_flow():
    """LangGraph ?Œí¬?Œë¡œ?°ì—??ê²€??ê²°ê³¼ ?„ë‹¬ ?•ì¸"""
    print("\n" + "="*80)
    print("LangGraph ?Œí¬?Œë¡œ??ê²€??ê²°ê³¼ ?¬í•¨ ë¶„ì„")
    print("="*80 + "\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        workflow_service = LangGraphWorkflowService(config)

        print("???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ\n")

        # ?ŒìŠ¤??ì¼€?´ìŠ¤
        test_query = "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??
        print(f"?“‹ ?ŒìŠ¤??ì§ˆë¬¸: {test_query}\n")
        print("?”„ ?Œí¬?Œë¡œ???¤í–‰ ì¤?..\n")

        # ?Œí¬?Œë¡œ???¤í–‰
        result = await workflow_service.process_query(test_query)

        # ê²°ê³¼ ë¶„ì„
        print("\n" + "="*80)
        print("ê²°ê³¼ ë¶„ì„")
        print("="*80 + "\n")

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        processing_steps = result.get("processing_steps", [])

        print(f"?“ ?µë? ê¸¸ì´: {len(answer)}??)
        print(f"?“š ì¶œì²˜ ?? {len(sources)}ê°?)
        print(f"?¯ ? ë¢°?? {confidence:.2f}")
        print(f"?±ï¸ ì²˜ë¦¬ ?¨ê³„: {len(processing_steps)}ê°?)

        # ê²€??ê´€???¨ê³„ ?•ì¸
        search_steps = [step for step in processing_steps if "ê²€?? in step or "search" in step.lower()]
        print(f"\n?” ê²€??ê´€???¨ê³„: {len(search_steps)}ê°?)
        for step in search_steps[:10]:
            print(f"   - {step}")

        # sources ?•ì¸
        if sources:
            print(f"\n?“š ì¶œì²˜ ?ì„¸:")
            for i, source in enumerate(sources[:5], 1):
                print(f"   {i}. {source}")
        else:
            print("\n? ï¸ ì¶œì²˜ê°€ ?†ìŠµ?ˆë‹¤!")

        # ?µë???ê²€??ê²°ê³¼ ?¸ìš© ?•ì¸
        answer_lower = answer.lower()
        citation_keywords = ["??, "ì¡?, "ë²?, "?ë?", "?€ë²•ì›"]
        has_citations = any(kw in answer_lower for kw in citation_keywords)

        print(f"\n?“– ?µë? ?¸ìš© ?•ì¸:")
        print(f"   - ë²•ë¥  ì¡°ë¬¸/?ë? ?¤ì›Œ???¬í•¨: {'?? if has_citations else '??}")

        # ?µë? ë¯¸ë¦¬ë³´ê¸°
        print(f"\n?“‹ ?µë? ë¯¸ë¦¬ë³´ê¸° (ì²?500??:")
        print("-" * 80)
        print(answer[:500])
        print("-" * 80)

        return {
            "answer_length": len(answer),
            "sources_count": len(sources),
            "confidence": confidence,
            "has_citations": has_citations,
            "processing_steps_count": len(processing_steps)
        }

    except Exception as e:
        print(f"???ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(test_langgraph_search_results_flow())

    if result:
        print("\n" + "="*80)
        print("ë¶„ì„ ê²°ê³¼ ?”ì•½")
        print("="*80)
        print(f"?µë? ê¸¸ì´: {result['answer_length']}??)
        print(f"ì¶œì²˜ ?? {result['sources_count']}ê°?)
        print(f"? ë¢°?? {result['confidence']:.2f}")
        print(f"ë²•ë¥  ?¸ìš©: {'?? if result['has_citations'] else '??}")
        print(f"ì²˜ë¦¬ ?¨ê³„: {result['processing_steps_count']}ê°?)
        print("="*80 + "\n")
