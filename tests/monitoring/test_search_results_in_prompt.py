#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ê²€??ê²°ê³¼ ?¬í•¨ ?ŒìŠ¤??
ê²€??ê²°ê³¼ê°€ ?„ë¡¬?„íŠ¸???¬í•¨?˜ì–´ ?ì ˆ???µë????ì„±?˜ëŠ”ì§€ ?•ì¸
"""

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from infrastructure.utils.langgraph_config import LangGraphConfig


def create_initial_legal_state(query: str, session_id: str) -> dict:
    """ì´ˆê¸° ë²•ë¥  ?íƒœ ?ì„±"""
    return {
        "query": query,
        "session_id": session_id,
        "query_type": "",
        "retrieved_docs": [],
        "processing_steps": [],
        "metadata": {}
    }


def test_with_search_results():
    """ê²€??ê²°ê³¼ê°€ ?ˆëŠ” ê²½ìš° ?ŒìŠ¤??""
    print("\n" + "="*80)
    print("ê²€??ê²°ê³¼ ?¬í•¨ ?ŒìŠ¤??- ê²€??ê²°ê³¼ê°€ ?ˆì„ ??)
    print("="*80 + "\n")

    try:
        # ?¤ì • ë¡œë“œ
        config = LangGraphConfig.from_env()
        workflow = EnhancedLegalQuestionWorkflow(config)

        print("???Œí¬?Œë¡œ??ì´ˆê¸°???„ë£Œ\n")

        # ?ŒìŠ¤??ì¼€?´ìŠ¤: ê²€??ê²°ê³¼ê°€ ?ˆëŠ” ê²½ìš°
        test_cases = [
            {
                "query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                "query_type": "legal_advice",
                "description": "ë¯¼ì‚¬ë²?- ?í•´ë°°ìƒ",
                "retrieved_docs": [
                    {
                        "content": "ë¯¼ë²• ??50ì¡?(ë¶ˆë²•?‰ìœ„???´ìš©) ?€?¸ì˜ ê³ ì˜ ?ëŠ” ê³¼ì‹¤ë¡??¸í•œ ë¶ˆë²•?‰ìœ„ë¡??¸í•˜???í•´ë¥?ë°›ì? ?ëŠ” ê·??í•´ë¥?ë°°ìƒë°›ì„ ???ˆë‹¤. ?í•´ë°°ìƒ??ì²?µ¬?˜ë ¤ë©?ê°€?´ì??ê³ ì˜ ?ëŠ” ê³¼ì‹¤, ?í•´??ë°œìƒ, ?¸ê³¼ê´€ê³„ë? ?…ì¦?´ì•¼ ?œë‹¤.",
                        "source": "ë¯¼ë²• ??50ì¡?,
                        "relevance_score": 0.95,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "750"}
                    },
                    {
                        "content": "ë¯¼ë²• ??51ì¡?(?¬ì‚°???í•´ë°°ìƒ) ë¶ˆë²•?‰ìœ„ë¡??¸í•˜???¬ì‚°???í•´ê°€ ë°œìƒ???Œì—??ê·??í•´ë¥?ë°°ìƒ?˜ì—¬???œë‹¤. ?í•´ë°°ìƒ?€ ?ì¹™?ìœ¼ë¡?ê¸ˆì „?¼ë¡œ ?´ë£¨?´ì?ë©? ?í•´??ë²”ìœ„???µìƒ???í•´?€ ?¹ë³„ ?í•´ë¥??¬í•¨?œë‹¤.",
                        "source": "ë¯¼ë²• ??51ì¡?,
                        "relevance_score": 0.92,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "751"}
                    },
                    {
                        "content": "?€ë²•ì› 2020??2345 ?ê²°???°ë¥´ë©? ?í•´ë°°ìƒ ì²?µ¬ê¶Œì? ?í•´ ë°œìƒ ?¬ì‹¤ê³??¸ê³¼ê´€ê³„ê? ?…ì¦?˜ì–´???±ë¦½?œë‹¤. ê°€?´ì??ê³¼ì‹¤ê³??í•´ ?¬ì´???¸ê³¼ê´€ê³„ëŠ” ?¼ë°˜?ì¸ ?¬íšŒ?µë…???°ë¼ ?ë‹¨?œë‹¤.",
                        "source": "?€ë²•ì› 2020??2345",
                        "relevance_score": 0.88,
                        "metadata": {"case_number": "2020??2345", "court": "?€ë²•ì›"}
                    }
                ]
            },
            {
                "query": "ê³„ì•½ ?´ì? ?”ê±´?€ ë¬´ì—‡?¸ê???",
                "query_type": "law_inquiry",
                "description": "ë¯¼ì‚¬ë²?- ê³„ì•½ ?´ì?",
                "retrieved_docs": [
                    {
                        "content": "ë¯¼ë²• ??43ì¡?(?´ì?ê¶Œì˜ ?‰ì‚¬) ê³„ì•½ ?¹ì‚¬?ì˜ ?¼ë°©?€ ê³„ì•½ ?ëŠ” ë²•ë¥ ??ê·œì •???˜í•œ ?´ì?ê¶Œì„ ?‰ì‚¬?????ˆë‹¤. ?´ì?ê¶Œì˜ ?‰ì‚¬???ë?ë°©ì— ?€???˜ì‚¬?œì‹œë¡??œë‹¤.",
                        "source": "ë¯¼ë²• ??43ì¡?,
                        "relevance_score": 0.96,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "543"}
                    },
                    {
                        "content": "ë¯¼ë²• ??44ì¡?(ì±„ë¬´ë¶ˆì´?‰ì„ ?´ìœ ë¡????´ì?) ê³„ì•½???´ì???ì±„ë¬´ë¶ˆì´?‰ì´ ?ˆì„ ???ë‹¹??ê¸°ê°„???•í•˜???´í–‰ ìµœê³ ë¥??˜ê³  ê·?ê¸°ê°„ ?´ì— ?´í–‰?˜ì? ?„ë‹ˆ??ê²½ìš°???????ˆë‹¤. ì±„ë¬´?ê? ?´í–‰??ê±°ë????Œì—??ìµœê³ ë¥?ê¸°ë‹¤ë¦¬ì? ?„ë‹ˆ?˜ê³  ?´ì??????ˆë‹¤.",
                        "source": "ë¯¼ë²• ??44ì¡?,
                        "relevance_score": 0.93,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "544"}
                    }
                ]
            },
            {
                "query": "?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??,
                "query_type": "legal_advice",
                "description": "ê°€ì¡±ë²• - ?´í˜¼ ?ˆì°¨",
                "retrieved_docs": [
                    {
                        "content": "ë¯¼ë²• ??34ì¡?(?‘ì˜???´í˜¼) ë¶€ë¶€???‘ì˜?˜ì—¬ ?´í˜¼?????ˆë‹¤. ?‘ì˜???´í˜¼?€ ê°€ì¡±ê?ê³„ë“±ë¡ë²•???•í•œ ë°”ì— ?°ë¼ ? ê³ ?¨ìœ¼ë¡œì¨ ê·??¨ë ¥???ê¸´??",
                        "source": "ë¯¼ë²• ??34ì¡?,
                        "relevance_score": 0.94,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "834"}
                    },
                    {
                        "content": "ë¯¼ë²• ??40ì¡?(?¬íŒ???´í˜¼) ë¶€ë¶€???¼ë°©?€ ?¤ìŒ ê°??¸ì˜ ?´ëŠ ?˜ë‚˜???´ë‹¹?˜ëŠ” ?¬ìœ ê°€ ?ˆëŠ” ê²½ìš°?ëŠ” ê°€?•ë²•?ì— ?´í˜¼??ì²?µ¬?????ˆë‹¤. 1. ë°°ìš°?ì— ë¶€?•í•œ ?‰ìœ„ê°€ ?ˆì—ˆ????2. ë°°ìš°?ê? ?…ì˜ë¡??¤ë¥¸ ?¼ë°©??? ê¸°????3. ë°°ìš°???ëŠ” ê·?ì§ê³„ì¡´ì†?¼ë¡œë¶€???¬íˆ ë¶€?ì ˆ???€?°ë? ë°›ì•˜????4. ?ê¸° ?ëŠ” ë°°ìš°?ì˜ ì§ê³„ì¡´ì†?¼ë¡œë¶€???¬íˆ ë¶€?ì ˆ???€?°ë? ??ë°°ìš°?ì— ?€?˜ì—¬ ?´í˜¼??ì²?µ¬?????ˆë‹¤.",
                        "source": "ë¯¼ë²• ??40ì¡?,
                        "relevance_score": 0.91,
                        "metadata": {"law_name": "ë¯¼ë²•", "article_no": "840"}
                    }
                ]
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"?ŒìŠ¤??{i}/{len(test_cases)}: {test_case['description']}")
            print(f"ì§ˆë¬¸: {test_case['query']}")
            print(f"ê²€??ê²°ê³¼: {len(test_case['retrieved_docs'])}ê°?)
            print(f"{'='*80}\n")

            # ì´ˆê¸° ?íƒœ ?ì„±
            state = create_initial_legal_state(test_case['query'], f"test-session-{i}")
            state["query_type"] = test_case['query_type']
            state["retrieved_docs"] = test_case['retrieved_docs']

            # generate_answer_enhanced ?¤í–‰
            result = workflow.generate_answer_enhanced(state)

            # ê²°ê³¼ ê²€ì¦?
            answer = result.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("answer", "") or str(answer)
            if not answer:
                answer = ""

            assert answer, "?µë????ì„±?˜ì? ?Šì•˜?µë‹ˆ??

            # ê²€??ê²°ê³¼ê°€ ?µë????¬í•¨?˜ì—ˆ?”ì? ?•ì¸
            answer_lower = str(answer).lower()
            has_citation = False
            cited_sources = []

            for doc in test_case['retrieved_docs']:
                source = doc.get("source", "")
                content_preview = doc.get("content", "")[:50]

                # ì¶œì²˜ê°€ ?µë????¬í•¨?˜ì—ˆ?”ì? ?•ì¸
                if source and (source in answer or any(keyword in answer for keyword in source.split())):
                    has_citation = True
                    cited_sources.append(source)

                # ì¡°ë¬¸ ë²ˆí˜¸ê°€ ?µë????¬í•¨?˜ì—ˆ?”ì? ?•ì¸
                article_no = doc.get("metadata", {}).get("article_no", "")
                if article_no and article_no in answer:
                    has_citation = True

            # ?µë? ê¸¸ì´ ?•ì¸ (?ˆë¬´ ì§§ìœ¼ë©??„ë¡¬?„íŠ¸ê°€ ì¶œë ¥??ê²ƒì¼ ???ˆìŒ)
            answer_length = len(answer)
            is_too_short = answer_length < 100
            is_too_long = answer_length > 5000

            print(f"?“ ?µë? ê¸¸ì´: {answer_length}??)
            print(f"?“š ?¸ìš©???ŒìŠ¤: {len(cited_sources)}ê°?/ {len(test_case['retrieved_docs'])}ê°?)
            if cited_sources:
                print(f"   ?¸ìš©???ŒìŠ¤ ëª©ë¡: {', '.join(cited_sources[:5])}")
            print(f"?“‹ ?µë? ë¯¸ë¦¬ë³´ê¸°: {answer[:200]}...")

            # ê²€ì¦?ê²°ê³¼
            test_passed = True
            issues = []

            if not has_citation:
                test_passed = False
                issues.append("ê²€??ê²°ê³¼ê°€ ?µë????¸ìš©?˜ì? ?Šì•˜?µë‹ˆ??)

            if is_too_short:
                test_passed = False
                issues.append(f"?µë????ˆë¬´ ì§§ìŠµ?ˆë‹¤ ({answer_length}??")

            if is_too_long:
                issues.append(f"? ï¸ ?µë???ë§¤ìš° ê¹ë‹ˆ??({answer_length}??")

            if test_passed:
                print(f"???ŒìŠ¤???µê³¼: ê²€??ê²°ê³¼ê°€ ?ì ˆ??ë°˜ì˜?˜ì—ˆ?µë‹ˆ??)
                results.append(True)
            else:
                print(f"? ï¸ ?ŒìŠ¤???¤íŒ¨:")
                for issue in issues:
                    print(f"   - {issue}")
                results.append(False)

        # ì¢…í•© ê²°ê³¼
        print(f"\n{'='*80}")
        print("ì¢…í•© ?ŒìŠ¤??ê²°ê³¼")
        print(f"{'='*80}")
        passed = sum(results)
        total = len(results)
        print(f"???µê³¼: {passed}/{total}")
        print(f"???¤íŒ¨: {total - passed}/{total}")
        print(f"{'='*80}\n")

        return all(results)

    except Exception as e:
        print(f"???ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_search_results()
    if success:
        print("??ëª¨ë“  ê²€??ê²°ê³¼ ?¬í•¨ ?ŒìŠ¤?¸ê? ?±ê³µ?ˆìŠµ?ˆë‹¤!")
    else:
        print("? ï¸ ?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤.")
