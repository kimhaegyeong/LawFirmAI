# -*- coding: utf-8 -*-
"""
ê²€??ê²°ê³¼ê°€ generate_answer_enhancedê¹Œì? ?„ë‹¬?˜ëŠ”ì§€ ê°„ë‹¨???ŒìŠ¤??
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_search_results_to_answer():
    """ê²€??ê²°ê³¼ ?„ë‹¬ ?ŒìŠ¤??""
    try:
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        print("=" * 80)
        print("ê²€??ê²°ê³¼ ?„ë‹¬ ?ŒìŠ¤???œì‘")
        print("=" * 80)

        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        print(f"???¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        workflow_service = LangGraphWorkflowService(config)
        print(f"???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì§ˆì˜
        test_query = "ê³„ì•½ ?´ì? ?”ê±´"
        print(f"\n?ŒìŠ¤??ì§ˆì˜: {test_query}")

        # ?„ì²´ ?Œí¬?Œë¡œ???¤í–‰
        print("?”„ ?„ì²´ ?Œí¬?Œë¡œ???¤í–‰ ì¤?..")
        result = await workflow_service.process_query(test_query, "test_session", enable_checkpoint=False)

        # ê²€??ê²°ê³¼ ?•ì¸
        retrieved_docs = result.get("retrieved_docs", [])
        metadata = result.get("metadata", {})
        search_meta = metadata.get("search", {}) if isinstance(metadata, dict) else {}

        semantic_count = search_meta.get("semantic_results_count", 0)
        keyword_count = search_meta.get("keyword_results_count", 0)
        final_count = len(retrieved_docs)

        print(f"\n?“Š ê²€??ê²°ê³¼:")
        print(f"   ?˜ë???ê²€?? {semantic_count}ê°?)
        print(f"   ?¤ì›Œ??ê²€?? {keyword_count}ê°?)
        print(f"   ìµœì¢… ?µí•© ê²°ê³¼: {final_count}ê°?)

        # ê²€??ê²°ê³¼ ?ì„¸
        if final_count > 0:
            print(f"\n?“„ ê²€??ê²°ê³¼ ?˜í”Œ:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                print(f"   {i}. Type: {doc.get('type', 'unknown')}")
                print(f"      Source: {str(doc.get('source', 'unknown'))[:60]}")
                content = str(doc.get('content', '') or doc.get('text', ''))[:100]
                print(f"      Content: {content}...")
        else:
            print("   ? ï¸ ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤!")

        # ?µë? ?•ì¸
        answer = result.get("answer", "")
        print(f"\n?ï¸ generate_answer_enhanced ê²°ê³¼:")
        print(f"   ?µë? ê¸¸ì´: {len(answer)}??)
        print(f"   retrieved_docs ê°œìˆ˜: {len(retrieved_docs)}ê°?)

        if answer:
            print(f"   ?µë? ë¯¸ë¦¬ë³´ê¸°: {answer[:150]}...")

            # ê²€??ê²°ê³¼ê°€ ?µë????¬í•¨?˜ì—ˆ?”ì? ?•ì¸
            if retrieved_docs:
                doc_found = False
                for doc in retrieved_docs[:3]:
                    source = str(doc.get("source", ""))
                    if source and len(source) > 10:
                        # ?ŒìŠ¤ ?´ë¦„???¼ë?ê°€ ?µë????ˆëŠ”ì§€ ?•ì¸
                        source_words = source.split()[:3]  # ì²˜ìŒ 3?¨ì–´
                        for word in source_words:
                            if word and len(word) > 5 and word in answer:
                                doc_found = True
                                print(f"   ??ê²€??ê²°ê³¼ê°€ ?µë????¬í•¨?? {word}")
                                break
                        if doc_found:
                            break

                if not doc_found:
                    print("   ? ï¸ ê²€??ê²°ê³¼ê°€ ?µë???ëª…ì‹œ?ìœ¼ë¡??¬í•¨?˜ì? ?Šì•˜?????ˆìŒ")

        # ìµœì¢… ê²€ì¦?
        print(f"\n??ìµœì¢… ê²€ì¦?")
        print(f"   - ê²€??ê²°ê³¼ ?ˆìŒ: {'?? if final_count > 0 else '??}")
        print(f"   - retrieved_docs ?„ë‹¬?? {'?? if len(retrieved_docs) > 0 else '??}")
        print(f"   - ?µë? ?ì„±?? {'?? if answer else '??}")

        # ê²€ì¦?
        assert final_count > 0, f"ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤! (semantic: {semantic_count}, keyword: {keyword_count})"
        assert len(retrieved_docs) > 0, "retrieved_docsê°€ ë¹„ì–´?ˆìŠµ?ˆë‹¤!"
        assert answer and len(answer) > 0, "?µë????ì„±?˜ì? ?Šì•˜?µë‹ˆ??"

        print(f"\n??ëª¨ë“  ê²€ì¦??µê³¼! ê²€??ê²°ê³¼ê°€ generate_answer_enhancedê¹Œì? ???„ë‹¬?˜ì—ˆ?µë‹ˆ??")

        return True

    except Exception as e:
        print(f"\n???ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_search_results_to_answer())
    sys.exit(0 if result else 1)


