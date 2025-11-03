#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?¤ì›Œ??ê´€ë¦??œìŠ¤???ŒìŠ¤??
"""

import os
import sys
from typing import Any, Dict, List

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.answer_structure_enhancer import AnswerStructureEnhancer
from source.services.database_keyword_manager import DatabaseKeywordManager


def test_database_keyword_manager():
    """?°ì´?°ë² ?´ìŠ¤ ?¤ì›Œ??ê´€ë¦¬ì ?ŒìŠ¤??""
    print("=" * 60)
    print("?°ì´?°ë² ?´ìŠ¤ ?¤ì›Œ??ê´€ë¦¬ì ?ŒìŠ¤??)
    print("=" * 60)

    db_manager = DatabaseKeywordManager()

    # 1. ì§ˆë¬¸ ? í˜• ì¡°íšŒ ?ŒìŠ¤??
    print("\n1. ì§ˆë¬¸ ? í˜• ì¡°íšŒ ?ŒìŠ¤??)
    question_types = db_manager.get_all_question_types()
    print(f"   ?±ë¡??ì§ˆë¬¸ ? í˜• ?? {len(question_types)}")
    for qt in question_types[:5]:  # ?ìœ„ 5ê°œë§Œ ?œì‹œ
        print(f"   - {qt['type_name']}: {qt['display_name']}")

    # 2. ?¤ì›Œ??ì¡°íšŒ ?ŒìŠ¤??
    print("\n2. ?¤ì›Œ??ì¡°íšŒ ?ŒìŠ¤??)
    test_types = ["precedent_search", "contract_review", "divorce_procedure"]

    for q_type in test_types:
        keywords = db_manager.get_keywords_for_type(q_type, limit=5)
        print(f"   {q_type}: {len(keywords)}ê°??¤ì›Œ??)
        for kw in keywords[:3]:  # ?ìœ„ 3ê°œë§Œ ?œì‹œ
            print(f"     - {kw['keyword']} ({kw['weight_level']}, {kw['weight_value']})")

    # 3. ?¤ì›Œ??ê²€???ŒìŠ¤??
    print("\n3. ?¤ì›Œ??ê²€???ŒìŠ¤??)
    search_terms = ["?ë?", "ê³„ì•½??, "?´í˜¼"]

    for term in search_terms:
        results = db_manager.search_keywords(term, limit=5)
        print(f"   '{term}' ê²€??ê²°ê³¼: {len(results)}ê°?)
        for result in results[:2]:  # ?ìœ„ 2ê°œë§Œ ?œì‹œ
            print(f"     - {result['question_type']}: {result['keyword']} ({result['weight_level']})")

    # 4. ?¨í„´ ì¡°íšŒ ?ŒìŠ¤??
    print("\n4. ?¨í„´ ì¡°íšŒ ?ŒìŠ¤??)
    for q_type in test_types:
        patterns = db_manager.get_patterns_for_type(q_type)
        print(f"   {q_type}: {len(patterns)}ê°??¨í„´")
        for pattern in patterns[:2]:  # ?ìœ„ 2ê°œë§Œ ?œì‹œ
            print(f"     - {pattern['pattern'][:50]}...")

    # 5. ?µê³„ ì¡°íšŒ ?ŒìŠ¤??
    print("\n5. ?µê³„ ì¡°íšŒ ?ŒìŠ¤??)
    stats = db_manager.get_keyword_statistics()
    print(f"   ?„ì²´ ?¤ì›Œ???? {stats.get('total_keywords', 0)}")
    print(f"   ê³ ê?ì¤‘ì¹˜ ?¤ì›Œ?? {stats.get('high_weight_count', 0)}")
    print(f"   ì¤‘ê?ì¤‘ì¹˜ ?¤ì›Œ?? {stats.get('medium_weight_count', 0)}")
    print(f"   ?€ê°€ì¤‘ì¹˜ ?¤ì›Œ?? {stats.get('low_weight_count', 0)}")

    print("\n???°ì´?°ë² ?´ìŠ¤ ?¤ì›Œ??ê´€ë¦¬ì ?ŒìŠ¤???„ë£Œ!")


def test_answer_structure_enhancer():
    """?µë? êµ¬ì¡°???¥ìƒ ?œìŠ¤???ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?µë? êµ¬ì¡°???¥ìƒ ?œìŠ¤???ŒìŠ¤??(DB ê¸°ë°˜)")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤??
    test_cases = [
        {
            "question": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½?œì—???„ì•½ê¸?ì¡°í•­??ê²€? í•´ì£¼ì„¸??,
            "question_type": "contract_review",
            "domain": "ë¯¼ì‚¬ë²?
        },
        {
            "question": "?´í˜¼ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
            "question_type": "divorce_procedure",
            "domain": "ê°€ì¡±ë²•"
        },
        {
            "question": "?€ë²•ì› ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
            "question_type": "precedent_search",
            "domain": "?¼ë°˜"
        },
        {
            "question": "?ì† ?ˆì°¨???€???Œê³  ?¶ìŠµ?ˆë‹¤",
            "question_type": "inheritance_procedure",
            "domain": "ê°€ì¡±ë²•"
        },
        {
            "question": "ë²•ë¥  ?©ì–´???˜ë?ë¥??¤ëª…?´ì£¼?¸ìš”",
            "question_type": "term_explanation",
            "domain": "?¼ë°˜"
        },
        {
            "question": "?¸ë™ ë¶„ìŸ ?´ê²° ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "question_type": "labor_dispute",
            "domain": "?¸ë™ë²?
        },
        {
            "question": "?•ì‚¬ ?¬ê±´ ì²˜ë¦¬ ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            "question_type": "criminal_case",
            "domain": "?•ì‚¬ë²?
        },
        {
            "question": "ë²•ë¥  ?ë‹´??ë°›ê³  ?¶ìŠµ?ˆë‹¤",
            "question_type": "legal_advice",
            "domain": "?¼ë°˜"
        },
        {
            "question": "ë²•ë ¹???€??ë¬¸ì˜?œë¦½?ˆë‹¤",
            "question_type": "law_inquiry",
            "domain": "?¼ë°˜"
        },
        {
            "question": "?¼ë°˜?ì¸ ë²•ë¥  ì§ˆë¬¸?…ë‹ˆ??,
            "question_type": "general",
            "domain": "?¼ë°˜"
        }
    ]

    success_count = 0
    total_count = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        print(f"\n?ŒìŠ¤??ì¼€?´ìŠ¤ {i}: {case['question']}")

        try:
            # êµ¬ì¡°???¥ìƒ ?¤í–‰
            result = enhancer.enhance_answer_structure(
                answer="?ŒìŠ¤???µë??…ë‹ˆ??",
                question_type=case['question_type'],
                question=case['question'],
                domain=case['domain']
            )

            if 'error' in result:
                print(f"   ???¤ë¥˜: {result['error']}")
            else:
                print(f"   ???±ê³µ!")
                print(f"     ì§ˆë¬¸ ? í˜•: {result.get('question_type', 'Unknown')}")
                print(f"     ?œí”Œë¦? {result.get('template_used', 'Unknown')}")

                quality_metrics = result.get('quality_metrics', {})
                print(f"     êµ¬ì¡°???ìˆ˜: {quality_metrics.get('structure_score', 0.0):.2f}")
                print(f"     ?„ì²´ ?ìˆ˜: {quality_metrics.get('overall_score', 0.0):.2f}")

                success_count += 1

        except Exception as e:
            print(f"   ???ˆì™¸ ë°œìƒ: {e}")

    print(f"\n" + "=" * 60)
    print(f"?µë? êµ¬ì¡°???¥ìƒ ?ŒìŠ¤??ê²°ê³¼: {success_count}/{total_count} ?±ê³µ")
    print(f"?±ê³µë¥? {(success_count/total_count)*100:.1f}%")
    print("=" * 60)

    return success_count == total_count


def test_question_type_mapping():
    """ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?ŒìŠ¤??(DB ê¸°ë°˜)")
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # ë§¤í•‘ ?ŒìŠ¤??ì¼€?´ìŠ¤??
    mapping_tests = [
        ("?ë?ë¥?ì°¾ì•„ì£¼ì„¸??, "precedent_search"),
        ("ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??, "contract_review"),
        ("?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??, "divorce_procedure"),
        ("?ì† ?ˆì°¨???´ë–»ê²??˜ë‚˜??", "inheritance_procedure"),
        ("ë²”ì£„ ì²˜ë²Œ???€???Œê³  ?¶ìŠµ?ˆë‹¤", "criminal_case"),
        ("?¸ë™ ë¶„ìŸ ?´ê²° ë°©ë²•", "labor_dispute"),
        ("?ˆì°¨ ?ˆë‚´ë¥??”ì²­?©ë‹ˆ??, "procedure_guide"),
        ("?©ì–´???˜ë?ë¥??¤ëª…?´ì£¼?¸ìš”", "term_explanation"),
        ("ë²•ë¥  ì¡°ì–¸??ë°›ê³  ?¶ìŠµ?ˆë‹¤", "legal_advice"),
        ("ë²•ë ¹???€??ë¬¸ì˜?œë¦½?ˆë‹¤", "law_inquiry"),
        ("?¼ë°˜?ì¸ ì§ˆë¬¸?…ë‹ˆ??, "general_question")
    ]

    correct_mappings = 0
    total_mappings = len(mapping_tests)

    for question, expected_type in mapping_tests:
        try:
            mapped_type = enhancer._map_question_type("", question)
            mapped_type_name = mapped_type.value if hasattr(mapped_type, 'value') else str(mapped_type)

            is_correct = mapped_type_name == expected_type
            status = "?? if is_correct else "??

            print(f"{status} '{question}' -> {mapped_type_name} (?ˆìƒ: {expected_type})")

            if is_correct:
                correct_mappings += 1

        except Exception as e:
            print(f"??'{question}' -> ?¤ë¥˜: {e}")

    print(f"\n" + "=" * 60)
    print(f"ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?ŒìŠ¤??ê²°ê³¼: {correct_mappings}/{total_mappings} ?•í™•")
    print(f"?•í™•?? {(correct_mappings/total_mappings)*100:.1f}%")
    print("=" * 60)

    return correct_mappings >= total_mappings * 0.8  # 80% ?´ìƒ ?•í™•??


def test_performance():
    """?±ëŠ¥ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?±ëŠ¥ ?ŒìŠ¤??)
    print("=" * 60)

    import time

    db_manager = DatabaseKeywordManager()
    enhancer = AnswerStructureEnhancer()

    # ?°ì´?°ë² ?´ìŠ¤ ì¡°íšŒ ?±ëŠ¥ ?ŒìŠ¤??
    print("\n1. ?°ì´?°ë² ?´ìŠ¤ ì¡°íšŒ ?±ëŠ¥ ?ŒìŠ¤??)

    start_time = time.time()
    for _ in range(100):
        keywords = db_manager.get_keywords_for_type("precedent_search", limit=10)
    db_time = time.time() - start_time
    print(f"   100???¤ì›Œ??ì¡°íšŒ: {db_time:.3f}ì´?(?‰ê· : {db_time/100*1000:.1f}ms)")

    # ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?±ëŠ¥ ?ŒìŠ¤??
    print("\n2. ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?±ëŠ¥ ?ŒìŠ¤??)

    test_questions = [
        "?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
        "ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??,
        "?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
        "?ì† ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
        "ë²”ì£„ ì²˜ë²Œ???€???Œê³  ?¶ìŠµ?ˆë‹¤"
    ]

    start_time = time.time()
    for _ in range(50):
        for question in test_questions:
            mapped_type = enhancer._map_question_type("", question)
    mapping_time = time.time() - start_time
    print(f"   250??ì§ˆë¬¸ ? í˜• ë§¤í•‘: {mapping_time:.3f}ì´?(?‰ê· : {mapping_time/250*1000:.1f}ms)")

    print("\n???±ëŠ¥ ?ŒìŠ¤???„ë£Œ!")


def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?¤ì›Œ??ê´€ë¦??œìŠ¤??ì¢…í•© ?ŒìŠ¤??)

    try:
        # 1. ?°ì´?°ë² ?´ìŠ¤ ?¤ì›Œ??ê´€ë¦¬ì ?ŒìŠ¤??
        test_database_keyword_manager()

        # 2. ?µë? êµ¬ì¡°???¥ìƒ ?œìŠ¤???ŒìŠ¤??
        structure_test_passed = test_answer_structure_enhancer()

        # 3. ì§ˆë¬¸ ? í˜• ë§¤í•‘ ?ŒìŠ¤??
        mapping_test_passed = test_question_type_mapping()

        # 4. ?±ëŠ¥ ?ŒìŠ¤??
        test_performance()

        # ?„ì²´ ê²°ê³¼
        print(f"\n" + "=" * 60)
        print("?„ì²´ ?ŒìŠ¤??ê²°ê³¼")
        print("=" * 60)
        print(f"?µë? êµ¬ì¡°???¥ìƒ: {'???µê³¼' if structure_test_passed else '???¤íŒ¨'}")
        print(f"ì§ˆë¬¸ ? í˜• ë§¤í•‘: {'???µê³¼' if mapping_test_passed else '???¤íŒ¨'}")

        if structure_test_passed and mapping_test_passed:
            print("?‰ ëª¨ë“  ?ŒìŠ¤?¸ê? ?µê³¼?ˆìŠµ?ˆë‹¤!")
            print("?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?¤ì›Œ??ê´€ë¦??œìŠ¤?œì´ ?•ìƒ?ìœ¼ë¡??‘ë™?©ë‹ˆ??")
        else:
            print("? ï¸ ?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤.")
            print("ì¶”ê? ?˜ì •???„ìš”?????ˆìŠµ?ˆë‹¤.")

        return structure_test_passed and mapping_test_passed

    except Exception as e:
        print(f"???ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
