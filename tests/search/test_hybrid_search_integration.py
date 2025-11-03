# -*- coding: utf-8 -*-
"""
?˜ì´ë¸Œë¦¬??ê²€???µí•© ?ŒìŠ¤??
SemanticSearchEngine + LangGraph ?Œí¬?Œë¡œ???µí•© ê²€ì¦?
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


async def test_hybrid_search_integration():
    """?˜ì´ë¸Œë¦¬??ê²€???µí•© ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("?˜ì´ë¸Œë¦¬??ê²€??LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??)
    print("="*80 + "\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        print("??LangGraph ?¤ì • ë¡œë“œ ?„ë£Œ")

        # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
        workflow_service = LangGraphWorkflowService(config)
        print("??LangGraphWorkflowService ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì¿¼ë¦¬??- ?¤ì–‘??ê²€???œë‚˜ë¦¬ì˜¤
        test_queries = [
            {
                "query": "?´í˜¼ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
                "description": "ê°€ì¡±ë²• - ?˜ì´ë¸Œë¦¬??ê²€???ŒìŠ¤??,
                "expected_sources": ["family_law", "civil_law"]
            },
            {
                "query": "ê³„ì•½?œì—???í•´ë°°ìƒ ì¡°í•­?€ ?´ë–»ê²??‘ì„±?´ì•¼ ?˜ë‚˜??",
                "description": "ë¯¼ì‚¬ë²?- ?˜ë???ê²€??ê°•ì  ?ŒìŠ¤??,
                "expected_sources": ["contract_review", "civil_law"]
            },
            {
                "query": "ë¶€?¹í•´ê³ ì˜ êµ¬ì œ ?ˆì°¨??",
                "description": "?¸ë™ë²?- ?¤ì›Œ??ê²€???ŒìŠ¤??,
                "expected_sources": ["labor_law"]
            },
            {
                "query": "?¹í—ˆê¶?ì¹¨í•´ ???´ë–»ê²??€?‘í•˜?˜ìš”?",
                "description": "ì§€?ì¬?°ê¶Œë²?- ?¼í•© ê²€???ŒìŠ¤??,
                "expected_sources": ["intellectual_property"]
            }
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"?ŒìŠ¤??{i}/{len(test_queries)}: {test_case['description']}")
            print(f"ì§ˆë¬¸: {test_case['query']}")
            print(f"{'='*80}\n")

            try:
                # ?Œí¬?Œë¡œ???¤í–‰
                result = await workflow_service.process_query(
                    query=test_case['query'],
                    enable_checkpoint=False
                )

                # ê²°ê³¼ ë¶„ì„
                print(f"??ì²˜ë¦¬ ?„ë£Œ: {result.get('answer', '')[:100]}...")

                # ê²€??ë©”í??°ì´???•ì¸
                metadata = result.get('metadata', {})
                search_metadata = result.get('search_metadata', {})

                print(f"\n?“Š ê²€??ë©”í??°ì´??")
                print(f"  - ?˜ë???ê²€??ê²°ê³¼: {search_metadata.get('semantic_results_count', 0)}ê°?)
                print(f"  - ?¤ì›Œ??ê²€??ê²°ê³¼: {search_metadata.get('keyword_results_count', 0)}ê°?)
                print(f"  - ìµœì¢… ë¬¸ì„œ ?? {search_metadata.get('final_count', 0)}ê°?)
                print(f"  - ê²€??ëª¨ë“œ: {search_metadata.get('search_mode', 'N/A')}")
                print(f"  - ê²€???œê°„: {search_metadata.get('search_time', 0):.3f}ì´?)

                # ì²˜ë¦¬ ?¨ê³„ ?•ì¸
                steps = result.get('processing_steps', [])
                print(f"\n?” ì²˜ë¦¬ ?¨ê³„:")
                for step in steps:
                    if 'ê²€?? in step or '?˜ì´ë¸Œë¦¬?? in step:
                        print(f"  ??{step}")

                # ?ŒìŠ¤ ?•ì¸
                sources = result.get('sources', [])
                print(f"\n?“š ê²€?‰ëœ ?ŒìŠ¤ ({len(sources)}ê°?:")
                for j, source in enumerate(sources[:5], 1):
                    print(f"  {j}. {source}")

                results.append({
                    'test_case': test_case,
                    'success': True,
                    'result': result,
                    'search_metadata': search_metadata
                })

            except Exception as e:
                print(f"???ŒìŠ¤???¤íŒ¨: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'test_case': test_case,
                    'success': False,
                    'error': str(e)
                })

        # ì¢…í•© ê²°ê³¼ ì¶œë ¥
        print(f"\n{'='*80}")
        print("ì¢…í•© ?ŒìŠ¤??ê²°ê³¼")
        print(f"{'='*80}\n")

        success_count = sum(1 for r in results if r.get('success'))
        total_count = len(results)

        print(f"???±ê³µ: {success_count}/{total_count}")
        print(f"???¤íŒ¨: {total_count - success_count}/{total_count}\n")

        # ?˜ì´ë¸Œë¦¬??ê²€??ë¶„ì„
        hybrid_search_used = sum(1 for r in results
                                 if r.get('success') and
                                 r.get('search_metadata', {}).get('search_mode') == 'hybrid')

        print(f"?” ?˜ì´ë¸Œë¦¬??ê²€??ëª¨ë“œ ?¬ìš©: {hybrid_search_used}/{success_count}??)

        # ê²€???±ëŠ¥ ë¶„ì„
        if success_count > 0:
            avg_semantic = sum(r.get('search_metadata', {}).get('semantic_results_count', 0)
                              for r in results if r.get('success')) / success_count
            avg_keyword = sum(r.get('search_metadata', {}).get('keyword_results_count', 0)
                             for r in results if r.get('success')) / success_count
            avg_final = sum(r.get('search_metadata', {}).get('final_count', 0)
                           for r in results if r.get('success')) / success_count

            print(f"\n?“ˆ ?‰ê·  ê²€???±ëŠ¥:")
            print(f"  - ?˜ë???ê²€??ê²°ê³¼: {avg_semantic:.1f}ê°?)
            print(f"  - ?¤ì›Œ??ê²€??ê²°ê³¼: {avg_keyword:.1f}ê°?)
            print(f"  - ìµœì¢… ? íƒ ë¬¸ì„œ: {avg_final:.1f}ê°?)

        # ?ì„¸ ê²°ê³¼ ì¶œë ¥
        print(f"\n{'='*80}")
        print("?ì„¸ ê²°ê³¼")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            if result.get('success'):
                test_case = result['test_case']
                metadata = result['search_metadata']
                print(f"\n{i}. {test_case['description']}")
                print(f"   ê²€?? ?˜ë???{metadata.get('semantic_results_count', 0)}ê°?+ "
                      f"?¤ì›Œ??{metadata.get('keyword_results_count', 0)}ê°???"
                      f"ìµœì¢… {metadata.get('final_count', 0)}ê°?)
            else:
                print(f"\n{i}. ???¤íŒ¨: {result['test_case']['description']}")
                print(f"   ?¤ë¥˜: {result.get('error', 'Unknown error')}")

        print(f"\n{'='*80}")
        print("???˜ì´ë¸Œë¦¬??ê²€???µí•© ?ŒìŠ¤???„ë£Œ")
        print(f"{'='*80}\n")

        return results

    except Exception as e:
        print(f"\n???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_semantic_search_only():
    """?˜ë???ê²€???”ì§„ ?¨ë… ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("SemanticSearchEngine ?¨ë… ?ŒìŠ¤??)
    print("="*80 + "\n")

    try:
        from source.services.semantic_search_engine import SemanticSearchEngine

        # SemanticSearchEngine ì´ˆê¸°??
        print("SemanticSearchEngine ì´ˆê¸°??ì¤?..")
        search_engine = SemanticSearchEngine()
        print("??SemanticSearchEngine ì´ˆê¸°???„ë£Œ")

        # ?ŒìŠ¤??ì¿¼ë¦¬
        test_queries = [
            "?´í˜¼ ?ˆì°¨",
            "ê³„ì•½???í•´ë°°ìƒ",
            "ë¶€?¹í•´ê³?êµ¬ì œ",
            "?¹í—ˆê¶?ì¹¨í•´"
        ]

        for query in test_queries:
            print(f"\nê²€??ì¿¼ë¦¬: '{query}'")
            print("-" * 80)

            results = search_engine.search(query, k=5)

            if results:
                print(f"ê²€??ê²°ê³¼: {len(results)}ê°?n")
                for i, result in enumerate(results[:3], 1):
                    print(f"{i}. [Score: {result.get('score', 0):.3f}]")
                    print(f"   ?ìŠ¤?? {result.get('text', '')[:100]}...")
                    print(f"   ?ŒìŠ¤: {result.get('source', 'Unknown')}")
                    print()
            else:
                print("ê²€??ê²°ê³¼ ?†ìŒ\n")

        print("??SemanticSearchEngine ?¨ë… ?ŒìŠ¤???„ë£Œ")

    except Exception as e:
        print(f"??SemanticSearchEngine ?ŒìŠ¤???¤íŒ¨: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("?˜ì´ë¸Œë¦¬??ê²€???µí•© ?ŒìŠ¤???¤í–‰\n")

    # 1. SemanticSearchEngine ?¨ë… ?ŒìŠ¤??
    asyncio.run(test_semantic_search_only())

    # 2. ?„ì²´ ?Œí¬?Œë¡œ???µí•© ?ŒìŠ¤??
    asyncio.run(test_hybrid_search_integration())
