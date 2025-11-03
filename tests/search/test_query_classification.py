#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì§ˆì˜ ë¶„ë¥˜ ?ŒìŠ¤???¤í¬ë¦½íŠ¸
?¤ì œ ë²•ë¥  ì§ˆì˜ë¥??€?ìœ¼ë¡??°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?œìŠ¤???ŒìŠ¤??
classify_question_type ë©”ì„œ???ŒìŠ¤???¬í•¨
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.answer_structure_enhancer import (
    AnswerStructureEnhancer,
    QuestionType,
)
from source.services.database_keyword_manager import DatabaseKeywordManager


def get_real_world_queries() -> List[Dict[str, Any]]:
    """?¤ì œ ë²•ë¥  ì§ˆì˜ ?°ì´??""
    return [
        # ?ë? ê²€??ê´€??
        {
            "query": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½ ?´ì? ???„ì•½ê¸?ê´€???€ë²•ì› ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
            "expected_type": "precedent_search",
            "category": "?ë? ê²€??,
            "difficulty": "high"
        },
        {
            "query": "?´í˜¼ ???¬ì‚°ë¶„í•  ê´€??? ì‚¬ ?ë?ê°€ ?ˆë‚˜??",
            "expected_type": "precedent_search",
            "category": "?ë? ê²€??,
            "difficulty": "medium"
        },
        {
            "query": "ê·¼ë¡œ???´ê³  ê´€??ìµœê·¼ ?ë?ë¥??Œë ¤ì£¼ì„¸??,
            "expected_type": "precedent_search",
            "category": "?ë? ê²€??,
            "difficulty": "medium"
        },

        # ê³„ì•½??ê²€??ê´€??
        {
            "query": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½?œì˜ ?„ì•½ê¸?ì¡°í•­??ê²€? í•´ì£¼ì„¸??,
            "expected_type": "contract_review",
            "category": "ê³„ì•½??ê²€??,
            "difficulty": "high"
        },
        {
            "query": "?„ë?ì°?ê³„ì•½?œì—???˜ì •???„ìš”??ë¶€ë¶„ì´ ?ˆë‚˜??",
            "expected_type": "contract_review",
            "category": "ê³„ì•½??ê²€??,
            "difficulty": "medium"
        },
        {
            "query": "ê·¼ë¡œê³„ì•½?œì˜ ë¶ˆë¦¬??ì¡°í•­???•ì¸?´ì£¼?¸ìš”",
            "expected_type": "contract_review",
            "category": "ê³„ì•½??ê²€??,
            "difficulty": "medium"
        },

        # ?´í˜¼ ?ˆì°¨ ê´€??
        {
            "query": "?‘ì˜?´í˜¼ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
            "expected_type": "divorce_procedure",
            "category": "?´í˜¼ ?ˆì°¨",
            "difficulty": "low"
        },
        {
            "query": "?¬íŒ?´í˜¼ ? ì²­ ???„ìš”???œë¥˜??ë¬´ì—‡?¸ê???",
            "expected_type": "divorce_procedure",
            "category": "?´í˜¼ ?ˆì°¨",
            "difficulty": "medium"
        },
        {
            "query": "?´í˜¼ ???‘ìœ¡ê¶Œê³¼ ë©´ì ‘êµì„­ê¶Œì? ?´ë–»ê²?ê²°ì •?˜ë‚˜??",
            "expected_type": "divorce_procedure",
            "category": "?´í˜¼ ?ˆì°¨",
            "difficulty": "high"
        },

        # ?ì† ?ˆì°¨ ê´€??
        {
            "query": "?ì† ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
            "expected_type": "inheritance_procedure",
            "category": "?ì† ?ˆì°¨",
            "difficulty": "low"
        },
        {
            "query": "? ì–¸???ˆëŠ” ê²½ìš° ?ì† ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            "expected_type": "inheritance_procedure",
            "category": "?ì† ?ˆì°¨",
            "difficulty": "medium"
        },
        {
            "query": "?ì†??? ê³ ???¸ì œê¹Œì? ?´ì•¼ ?˜ë‚˜??",
            "expected_type": "inheritance_procedure",
            "category": "?ì† ?ˆì°¨",
            "difficulty": "medium"
        },

        # ?•ì‚¬ ?¬ê±´ ê´€??
        {
            "query": "?¬ê¸°ì£„ë¡œ ê³ ì†Œ?¹í–ˆ?”ë° ?´ë–»ê²??´ì•¼ ?˜ë‚˜??",
            "expected_type": "criminal_case",
            "category": "?•ì‚¬ ?¬ê±´",
            "difficulty": "high"
        },
        {
            "query": "êµí†µ?¬ê³ ë¡?ê³¼ì‹¤ì¹˜ìƒ?ì£„ê°€ ?ìš©?˜ë‚˜??",
            "expected_type": "criminal_case",
            "category": "?•ì‚¬ ?¬ê±´",
            "difficulty": "medium"
        },
        {
            "query": "?•ì‚¬?¬ê±´?ì„œ ë³€?¸ì¸ ? ì„?€ ?„ìˆ˜?¸ê???",
            "expected_type": "criminal_case",
            "category": "?•ì‚¬ ?¬ê±´",
            "difficulty": "low"
        },

        # ?¸ë™ ë¶„ìŸ ê´€??
        {
            "query": "ë¶€?¹í•´ê³ ë¡œ ?¸ë™?„ì›?Œì— ? ì²­?˜ë ¤ê³??©ë‹ˆ??,
            "expected_type": "labor_dispute",
            "category": "?¸ë™ ë¶„ìŸ",
            "difficulty": "medium"
        },
        {
            "query": "?„ê¸ˆì²´ë¶ˆë¡??¸í•œ êµ¬ì œ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
            "expected_type": "labor_dispute",
            "category": "?¸ë™ ë¶„ìŸ",
            "difficulty": "medium"
        },
        {
            "query": "ê·¼ë¡œ?œê°„ ?„ë°˜?¼ë¡œ ?¸í•œ ë¶„ìŸ ?´ê²° ë°©ë²•?€?",
            "expected_type": "labor_dispute",
            "category": "?¸ë™ ë¶„ìŸ",
            "difficulty": "medium"
        },

        # ?ˆì°¨ ?ˆë‚´ ê´€??
        {
            "query": "?Œì•¡?¬ê±´?¬íŒ?ˆì°¨???´ë–»ê²?? ì²­?˜ë‚˜??",
            "expected_type": "procedure_guide",
            "category": "?ˆì°¨ ?ˆë‚´",
            "difficulty": "medium"
        },
        {
            "query": "ë¯¼ì‚¬ì¡°ì • ? ì²­ ë°©ë²•ê³??ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
            "expected_type": "procedure_guide",
            "category": "?ˆì°¨ ?ˆë‚´",
            "difficulty": "medium"
        },
        {
            "query": "ê°€?•ë²•???´í˜¼ì¡°ì • ? ì²­?€ ?´ë–»ê²??˜ë‚˜??",
            "expected_type": "procedure_guide",
            "category": "?ˆì°¨ ?ˆë‚´",
            "difficulty": "medium"
        },

        # ?©ì–´ ?´ì„¤ ê´€??
        {
            "query": "ë¶ˆë²•?‰ìœ„???˜ë??€ êµ¬ì„±?”ê±´???¤ëª…?´ì£¼?¸ìš”",
            "expected_type": "term_explanation",
            "category": "?©ì–´ ?´ì„¤",
            "difficulty": "high"
        },
        {
            "query": "?í•´ë°°ìƒê³??„ìë£Œì˜ ì°¨ì´?ì? ë¬´ì—‡?¸ê???",
            "expected_type": "term_explanation",
            "category": "?©ì–´ ?´ì„¤",
            "difficulty": "medium"
        },
        {
            "query": "ì±„ê¶Œê³?ì±„ë¬´??ê°œë…???½ê²Œ ?¤ëª…?´ì£¼?¸ìš”",
            "expected_type": "term_explanation",
            "category": "?©ì–´ ?´ì„¤",
            "difficulty": "low"
        },

        # ë²•ë¥  ì¡°ì–¸ ê´€??
        {
            "query": "ê³„ì•½ ?„ë°˜?¼ë¡œ ?í•´ë¥??…ì—ˆ?”ë° ?´ë–»ê²??´ì•¼ ?˜ë‚˜??",
            "expected_type": "legal_advice",
            "category": "ë²•ë¥  ì¡°ì–¸",
            "difficulty": "high"
        },
        {
            "query": "?´ì›ƒê³¼ì˜ ?ŒìŒ ë¶„ìŸ ?´ê²° ë°©ë²•??ì¡°ì–¸?´ì£¼?¸ìš”",
            "expected_type": "legal_advice",
            "category": "ë²•ë¥  ì¡°ì–¸",
            "difficulty": "medium"
        },
        {
            "query": "ì§ì¥?ì„œ ?±í¬ë¡±ì„ ?¹í–ˆ?”ë° ?´ë–»ê²??€ì²˜í•´???˜ë‚˜??",
            "expected_type": "legal_advice",
            "category": "ë²•ë¥  ì¡°ì–¸",
            "difficulty": "high"
        },

        # ë²•ë¥  ë¬¸ì˜ ê´€??
        {
            "query": "ë¯¼ë²• ??50ì¡°ì˜ ?´ìš©???Œë ¤ì£¼ì„¸??,
            "expected_type": "law_inquiry",
            "category": "ë²•ë¥  ë¬¸ì˜",
            "difficulty": "medium"
        },
        {
            "query": "ê·¼ë¡œê¸°ì?ë²•ì—???•í•œ ìµœì??„ê¸ˆ?€ ?¼ë§ˆ?¸ê???",
            "expected_type": "law_inquiry",
            "category": "ë²•ë¥  ë¬¸ì˜",
            "difficulty": "low"
        },
        {
            "query": "?•ë²• ??57ì¡°ì˜ ì²˜ë²Œ ê¸°ì??€ ?´ë–»ê²??˜ë‚˜??",
            "expected_type": "law_inquiry",
            "category": "ë²•ë¥  ë¬¸ì˜",
            "difficulty": "medium"
        },

        # ?¼ë°˜ ì§ˆë¬¸ ê´€??
        {
            "query": "ë²•ë¥  ?ë‹´?€ ?´ë””??ë°›ì„ ???ˆë‚˜??",
            "expected_type": "general_question",
            "category": "?¼ë°˜ ì§ˆë¬¸",
            "difficulty": "low"
        },
        {
            "query": "ë³€?¸ì‚¬ ? ì„ ë¹„ìš©?€ ?¼ë§ˆ???œë‚˜??",
            "expected_type": "general_question",
            "category": "?¼ë°˜ ì§ˆë¬¸",
            "difficulty": "low"
        },
        {
            "query": "ë²•ì›?ì„œ ?Œì†¡???œê¸°?˜ë ¤ë©??´ë–»ê²??´ì•¼ ?˜ë‚˜??",
            "expected_type": "general_question",
            "category": "?¼ë°˜ ì§ˆë¬¸",
            "difficulty": "medium"
        }
    ]


def test_query_classification():
    """ì§ˆì˜ ë¶„ë¥˜ ?ŒìŠ¤??""
    print("=" * 80)
    print("?¤ì œ ë²•ë¥  ì§ˆì˜ ë¶„ë¥˜ ?ŒìŠ¤??)
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()
    queries = get_real_world_queries()

    # ê²°ê³¼ ?€??
    results = {
        "total": len(queries),
        "correct": 0,
        "incorrect": 0,
        "by_category": {},
        "by_difficulty": {},
        "detailed_results": []
    }

    print(f"\nì´?{len(queries)}ê°œì˜ ì§ˆì˜ë¥??ŒìŠ¤?¸í•©?ˆë‹¤...\n")

    for i, query_data in enumerate(queries, 1):
        query = query_data["query"]
        expected_type = query_data["expected_type"]
        category = query_data["category"]
        difficulty = query_data["difficulty"]

        print(f"?ŒìŠ¤??{i:2d}: {query}")

        try:
            # ì§ˆë¬¸ ? í˜• ë§¤í•‘
            mapped_type = enhancer._map_question_type("", query)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            # ê²°ê³¼ ?ì •
            is_correct = mapped_type_name == expected_type
            status = "?? if is_correct else "??

            print(f"         {status} ?ˆìƒ: {expected_type} | ?¤ì œ: {mapped_type_name}")

            # ?µê³„ ?…ë°?´íŠ¸
            if is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1

            # ì¹´í…Œê³ ë¦¬ë³??µê³„
            if category not in results["by_category"]:
                results["by_category"][category] = {"total": 0, "correct": 0}
            results["by_category"][category]["total"] += 1
            if is_correct:
                results["by_category"][category]["correct"] += 1

            # ?œì´?„ë³„ ?µê³„
            if difficulty not in results["by_difficulty"]:
                results["by_difficulty"][difficulty] = {"total": 0, "correct": 0}
            results["by_difficulty"][difficulty]["total"] += 1
            if is_correct:
                results["by_difficulty"][difficulty]["correct"] += 1

            # ?ì„¸ ê²°ê³¼ ?€??
            results["detailed_results"].append({
                "query": query,
                "expected": expected_type,
                "actual": mapped_type_name,
                "correct": is_correct,
                "category": category,
                "difficulty": difficulty
            })

        except Exception as e:
            print(f"         ???¤ë¥˜ ë°œìƒ: {e}")
            results["incorrect"] += 1

    return results


def analyze_results(results: Dict[str, Any]):
    """ê²°ê³¼ ë¶„ì„ ë°?ë¦¬í¬???ì„±"""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??ê²°ê³¼ ë¶„ì„")
    print("=" * 80)

    # ?„ì²´ ?•í™•??
    accuracy = (results["correct"] / results["total"]) * 100
    print(f"\n?“Š ?„ì²´ ?•í™•?? {accuracy:.1f}% ({results['correct']}/{results['total']})")

    # ì¹´í…Œê³ ë¦¬ë³??•í™•??
    print(f"\n?“‹ ì¹´í…Œê³ ë¦¬ë³??•í™•??")
    for category, stats in results["by_category"].items():
        cat_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"   {category:12s}: {cat_accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")

    # ?œì´?„ë³„ ?•í™•??
    print(f"\n?¯ ?œì´?„ë³„ ?•í™•??")
    for difficulty, stats in results["by_difficulty"].items():
        diff_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"   {difficulty:8s}: {diff_accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")

    # ?¤ë¶„ë¥?ë¶„ì„
    print(f"\n???¤ë¶„ë¥??¬ë?:")
    incorrect_cases = [r for r in results["detailed_results"] if not r["correct"]]

    for case in incorrect_cases:
        print(f"   ì§ˆì˜: {case['query'][:50]}...")
        print(f"   ?ˆìƒ: {case['expected']} | ?¤ì œ: {case['actual']} | ì¹´í…Œê³ ë¦¬: {case['category']}")
        print()

    return accuracy


def test_edge_cases():
    """?£ì? ì¼€?´ìŠ¤ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?£ì? ì¼€?´ìŠ¤ ?ŒìŠ¤??)
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()

    edge_cases = [
        {
            "query": "?ë??€ ê³„ì•½??ëª¨ë‘ ê´€?¨ëœ ì§ˆë¬¸?…ë‹ˆ??,
            "description": "ë³µí•© ?¤ì›Œ??
        },
        {
            "query": "ë²•ë¥ ",
            "description": "ë§¤ìš° ì§§ì? ì§ˆì˜"
        },
        {
            "query": "?´í˜¼?˜ë©´???ì†??ê°™ì´ ì²˜ë¦¬?˜ê³  ?¶ì???ê³„ì•½?œë„ ê²€? ë°›ê³??ë???ì°¾ì•„ì£¼ì„¸??,
            "description": "ë§¤ìš° ê¸?ë³µí•© ì§ˆì˜"
        },
        {
            "query": "123456789",
            "description": "?«ìë§??¬í•¨"
        },
        {
            "query": "!@#$%^&*()",
            "description": "?¹ìˆ˜ë¬¸ìë§??¬í•¨"
        },
        {
            "query": "",
            "description": "ë¹?ì§ˆì˜"
        },
        {
            "query": "ë²•ë¥ ?ë‹´ë³€?¸ì‚¬ê³„ì•½?œì´?¼ìƒ?í˜•?¬ë…¸?™ì ˆì°¨ìš©?´ì¡°?¸ë¬¸??,
            "description": "ëª¨ë“  ?¤ì›Œ???¬í•¨"
        }
    ]

    print(f"\n?£ì? ì¼€?´ìŠ¤ {len(edge_cases)}ê°œë? ?ŒìŠ¤?¸í•©?ˆë‹¤...\n")

    for i, case in enumerate(edge_cases, 1):
        query = case["query"]
        description = case["description"]

        print(f"?£ì? ì¼€?´ìŠ¤ {i}: {description}")
        print(f"   ì§ˆì˜: '{query}'")

        try:
            mapped_type = enhancer._map_question_type("", query)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            print(f"   ê²°ê³¼: {mapped_type_name}")
            print(f"   ?íƒœ: {'???•ìƒ ì²˜ë¦¬' if mapped_type_name else '???¤ë¥˜'}")

        except Exception as e:
            print(f"   ?íƒœ: ???¤ë¥˜ ë°œìƒ - {e}")

        print()


def test_performance_with_real_queries():
    """?¤ì œ ì§ˆì˜ë¡??±ëŠ¥ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?¤ì œ ì§ˆì˜ ?±ëŠ¥ ?ŒìŠ¤??)
    print("=" * 80)

    import time

    enhancer = AnswerStructureEnhancer()
    queries = get_real_world_queries()

    # ?±ëŠ¥ ?ŒìŠ¤??
    print(f"\n{len(queries)}ê°œì˜ ?¤ì œ ì§ˆì˜ë¡??±ëŠ¥??ì¸¡ì •?©ë‹ˆ??..")

    start_time = time.time()
    for query_data in queries:
        mapped_type = enhancer._map_question_type("", query_data["query"])
    end_time = time.time()

    total_time = end_time - start_time

    if len(queries) == 0:
        print(f"\n? ï¸  ì§ˆì˜ ?°ì´?°ê? ?†ì–´ ?±ëŠ¥ ?ŒìŠ¤?¸ë? ê±´ë„ˆ?ë‹ˆ??")
        return

    if total_time == 0:
        total_time = 0.001  # ìµœì†Œê°??¤ì •?˜ì—¬ division by zero ë°©ì?

    avg_time = total_time / len(queries) * 1000  # msë¡?ë³€??

    print(f"\n?“ˆ ?±ëŠ¥ ê²°ê³¼:")
    print(f"   ?„ì²´ ì²˜ë¦¬ ?œê°„: {total_time:.3f}ì´?)
    print(f"   ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.1f}ms/ì§ˆì˜")
    print(f"   ì²˜ë¦¬?? {len(queries)/total_time:.1f}ì§ˆì˜/ì´?)


def test_classify_question_type():
    """classify_question_type ë©”ì„œ???ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("classify_question_type ë©”ì„œ???ŒìŠ¤??)
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤??
    test_cases = [
        # ë²•ë¥  ë¬¸ì˜ ?ŒìŠ¤??
        ("ë¯¼ë²• ??23ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???", QuestionType.LAW_INQUIRY),
        ("?•ë²• ??50ì¡?ì²˜ë²Œ ê¸°ì??€?", QuestionType.LAW_INQUIRY),
        ("ê·¼ë¡œê¸°ì?ë²???5ì¡??˜ë???", QuestionType.LAW_INQUIRY),
        ("?ë²• ??23ì¡??´ì„?´ì£¼?¸ìš”", QuestionType.LAW_INQUIRY),
        ("?Œë²• ??0ì¡??´ìš©?€?", QuestionType.LAW_INQUIRY),
        ("?¹í—ˆë²???5ì¡?ê·œì •?€?", QuestionType.LAW_INQUIRY),

        # ?ë? ê²€???ŒìŠ¤??
        ("?€ë²•ì› ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??, QuestionType.PRECEDENT_SEARCH),
        ("ê´€???ë?ê°€ ?ˆë‚˜??", QuestionType.PRECEDENT_SEARCH),
        ("ê³ ë“±ë²•ì› ?ê²°???Œë ¤ì£¼ì„¸??, QuestionType.PRECEDENT_SEARCH),
        ("ì§€ë°©ë²•???ë? ê²€??, QuestionType.PRECEDENT_SEARCH),

        # ê³„ì•½??ê²€???ŒìŠ¤??
        ("ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??, QuestionType.CONTRACT_REVIEW),
        ("??ê³„ì•½ ì¡°í•­??ë¶ˆë¦¬?œê???", QuestionType.CONTRACT_REVIEW),
        ("ê³„ì•½???˜ì •???„ìš”?œê???", QuestionType.CONTRACT_REVIEW),

        # ?´í˜¼ ?ˆì°¨ ?ŒìŠ¤??
        ("?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??, QuestionType.DIVORCE_PROCEDURE),
        ("?‘ì˜?´í˜¼ ë°©ë²•?€?", QuestionType.DIVORCE_PROCEDURE),
    ]

    print(f"\n{len(test_cases)}ê°œì˜ ?ŒìŠ¤??ì¼€?´ìŠ¤ ?¤í–‰ ì¤?..\n")

    passed = 0
    failed = 0

    for i, (question, expected_type) in enumerate(test_cases, 1):
        try:
            result_type = enhancer.classify_question_type(question)
            is_correct = result_type == expected_type
            status = "?? if is_correct else "??

            print(f"{i:2d}. {status} ì§ˆë¬¸: '{question}'")
            print(f"    ?ˆìƒ: {expected_type.value}")
            print(f"    ê²°ê³¼: {result_type.value}")
            print()

            if is_correct:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"{i:2d}. ???¤ë¥˜: '{question}'")
            print(f"    ?¤ë¥˜ ë©”ì‹œì§€: {e}")
            print()
            failed += 1

    print(f"\n?“Š ê²°ê³¼: {passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")


def test_edge_cases_classification():
    """?£ì? ì¼€?´ìŠ¤ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?£ì? ì¼€?´ìŠ¤ ?ŒìŠ¤??)
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()

    edge_cases = [
        ("", QuestionType.GENERAL_QUESTION),  # ë¹?ë¬¸ì??
        ("   ", QuestionType.GENERAL_QUESTION),  # ê³µë°±ë§?
        ("ë¯¼ë²•", QuestionType.GENERAL_QUESTION),  # ?¨ì–´ë§?
        ("??23ì¡?, QuestionType.LAW_INQUIRY),  # ì¡°ë¬¸ë§?
        ("123ì¡?, QuestionType.GENERAL_QUESTION),  # ?«ì+ì¡°ë¬¸
        ("ë¯¼ë²• ??, QuestionType.GENERAL_QUESTION),  # ë¶ˆì™„?„í•œ ì¡°ë¬¸
        ("?œì¡°", QuestionType.GENERAL_QUESTION),  # ?˜ëª»??ì¡°ë¬¸
        ("ë¯¼ë²• ??23ì¡???56??, QuestionType.LAW_INQUIRY),  # ë³µí•© ì¡°ë¬¸
        ("ë¯¼ë²•ê³??•ë²•", QuestionType.LAW_INQUIRY),  # ?¬ëŸ¬ ë²•ë ¹
        ("?ë??€ ê³„ì•½??, QuestionType.PRECEDENT_SEARCH),  # ?¬ëŸ¬ ?¤ì›Œ??(?°ì„ ?œìœ„)
    ]

    print(f"\n?£ì? ì¼€?´ìŠ¤ {len(edge_cases)}ê°??ŒìŠ¤??ì¤?..\n")

    for i, (question, expected_type) in enumerate(edge_cases, 1):
        try:
            result_type = enhancer.classify_question_type(question)
            is_correct = result_type == expected_type
            status = "?? if is_correct else "??

            print(f"{i:2d}. {status} ì§ˆë¬¸: '{question}'")
            print(f"    ?ˆìƒ: {expected_type.value}")
            print(f"    ê²°ê³¼: {result_type.value}")
            print()

        except Exception as e:
            print(f"{i:2d}. ???¤ë¥˜: '{question}'")
            print(f"    ?¤ë¥˜ ë©”ì‹œì§€: {e}")
            print()


def test_classify_performance():
    """classify_question_type ?±ëŠ¥ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("classify_question_type ?±ëŠ¥ ?ŒìŠ¤??)
    print("=" * 80)

    enhancer = AnswerStructureEnhancer()

    # ?±ëŠ¥ ?ŒìŠ¤?¸ìš© ì§ˆë¬¸??
    test_questions = [
        "ë¯¼ë²• ??23ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???",
        "?€ë²•ì› ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
        "ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??,
        "?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
    ] * 10  # 40ê°?ì§ˆë¬¸

    print(f"\n{len(test_questions)}ê°?ì§ˆë¬¸?¼ë¡œ ?±ëŠ¥ ?ŒìŠ¤???¤í–‰...")

    start_time = time.time()
    for question in test_questions:
        enhancer.classify_question_type(question)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = (total_time / len(test_questions)) * 1000  # msë¡?ë³€??

    print(f"\n?“ˆ ?±ëŠ¥ ê²°ê³¼:")
    print(f"   ?„ì²´ ì²˜ë¦¬ ?œê°„: {total_time:.3f}ì´?)
    print(f"   ?‰ê·  ì²˜ë¦¬ ?œê°„: {avg_time:.2f}ms/ì§ˆë¬¸")
    print(f"   ì²˜ë¦¬?? {len(test_questions)/total_time:.1f}ì§ˆë¬¸/ì´?)


def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("=" * 80)
    print("ë²•ë¥  ì§ˆì˜ ë¶„ë¥˜ ?œìŠ¤??ì¢…í•© ?ŒìŠ¤??)
    print("=" * 80)

    try:
        # 1. ?¤ì œ ì§ˆì˜ ë¶„ë¥˜ ?ŒìŠ¤??
        results = test_query_classification()

        # 2. ê²°ê³¼ ë¶„ì„
        accuracy = analyze_results(results)

        # 3. ?£ì? ì¼€?´ìŠ¤ ?ŒìŠ¤??
        test_edge_cases()

        # 4. ?±ëŠ¥ ?ŒìŠ¤??
        test_performance_with_real_queries()

        # ìµœì¢… ?‰ê?
        print("\n" + "=" * 80)
        print("ìµœì¢… ?‰ê?")
        print("=" * 80)

        if accuracy >= 90:
            grade = "A+ (?°ìˆ˜)"
        elif accuracy >= 80:
            grade = "A (?‘í˜¸)"
        elif accuracy >= 70:
            grade = "B (ë³´í†µ)"
        elif accuracy >= 60:
            grade = "C (ë¯¸í¡)"
        else:
            grade = "D (ë¶ˆëŸ‰)"

        print(f"?¯ ?„ì²´ ?•í™•?? {accuracy:.1f}%")
        print(f"?“Š ?±ê¸‰: {grade}")

        if accuracy >= 80:
            print("???œìŠ¤?œì´ ?¤ìš©?ìœ¼ë¡??¬ìš© ê°€?¥í•œ ?˜ì??…ë‹ˆ??")
        else:
            print("? ï¸ ?œìŠ¤??ê°œì„ ???„ìš”?©ë‹ˆ??")

        print("\n?‰ ì§ˆì˜ ë¶„ë¥˜ ?ŒìŠ¤?¸ê? ?„ë£Œ?˜ì—ˆ?µë‹ˆ??")

        return accuracy >= 80

    except Exception as e:
        print(f"???ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
