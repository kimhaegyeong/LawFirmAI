#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?œí”Œë¦??œìŠ¤??+ classify_question_type ?µí•© ?ŒìŠ¤??
"""

import os
import sys

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.answer_structure_enhancer import (
    AnswerStructureEnhancer,
    QuestionType,
)


def test_integrated_system():
    """?µí•© ?œìŠ¤???ŒìŠ¤??""
    print("=" * 60)
    print("?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?œí”Œë¦??œìŠ¤??+ classify_question_type ?µí•© ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # ?µí•© ?ŒìŠ¤??ì¼€?´ìŠ¤??
    test_cases = [
        {
            "question": "ë¯¼ë²• ??23ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???",
            "answer": "ë¯¼ë²• ??23ì¡°ëŠ” ê³„ì•½???´ì œ??ê´€??ê·œì •?…ë‹ˆ?? ??ì¡°í•­???°ë¥´ë©?ê³„ì•½ ?¹ì‚¬?ëŠ” ?ë?ë°©ì´ ê³„ì•½???´í–‰?˜ì? ?Šì„ ê²½ìš° ê³„ì•½???´ì œ?????ˆìŠµ?ˆë‹¤.",
            "expected_type": QuestionType.LAW_INQUIRY
        },
        {
            "question": "ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??,
            "answer": "?œê³µ?´ì£¼??ê³„ì•½?œë? ê²€? í•œ ê²°ê³¼, ëª?ê°€ì§€ ì£¼ì˜?¬í•­???ˆìŠµ?ˆë‹¤. ?¹íˆ ??ì¡°ì˜ ?í•´ë°°ìƒ ì¡°í•­??ê³¼ë„?˜ê²Œ ë¶ˆë¦¬?????ˆìŠµ?ˆë‹¤.",
            "expected_type": QuestionType.CONTRACT_REVIEW
        },
        {
            "question": "?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
            "answer": "?´í˜¼ ?ˆì°¨???¬ê²Œ ?‘ì˜?´í˜¼, ì¡°ì •?´í˜¼, ?¬íŒ?´í˜¼?¼ë¡œ ?˜ë‰©?ˆë‹¤. ?‘ì˜?´í˜¼??ê°€??ê°„ë‹¨?˜ë©°, ?‘ë°©???©ì˜?˜ë©´ ê°€?•ë²•?ì— ? ì²­?????ˆìŠµ?ˆë‹¤.",
            "expected_type": QuestionType.DIVORCE_PROCEDURE
        },
        {
            "question": "?€ë²•ì› ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
            "answer": "ê´€???€ë²•ì› ?ë?ë¥?ì°¾ì•˜?µë‹ˆ?? 2023??2345 ?¬ê±´?ì„œ ?€ë²•ì›?€ ê³„ì•½ ?´ì œ???”ê±´???€??ëª…í™•??ê¸°ì????œì‹œ?ˆìŠµ?ˆë‹¤.",
            "expected_type": QuestionType.PRECEDENT_SEARCH
        },
        {
            "question": "?¸ë™ ë¶„ìŸ ?´ê²° ë°©ë²•",
            "answer": "?¸ë™ ë¶„ìŸ??ë°œìƒ??ê²½ìš° ?¸ë™?„ì›?Œì— ? ì²­?˜ê±°??ë²•ì›???Œì†¡???œê¸°?????ˆìŠµ?ˆë‹¤. ë¨¼ì? ?¸ë™?„ì›??ì¡°ì •???µí•´ ?´ê²°???œë„?˜ëŠ” ê²ƒì´ ì¢‹ìŠµ?ˆë‹¤.",
            "expected_type": QuestionType.LABOR_DISPUTE
        }
    ]

    print(f"\nì´?{len(test_cases)}ê°??µí•© ?ŒìŠ¤??ì¼€?´ìŠ¤ ?¤í–‰ ì¤?..\n")

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        answer = test_case["answer"]
        expected_type = test_case["expected_type"]

        print(f"{i}. ?ŒìŠ¤??ì¼€?´ìŠ¤: {question}")
        print("-" * 50)

        try:
            # 1. ì§ˆë¬¸ ? í˜• ë¶„ë¥˜ ?ŒìŠ¤??
            classified_type = enhancer.classify_question_type(question)
            classification_correct = classified_type == expected_type

            print(f"   ì§ˆë¬¸ ? í˜• ë¶„ë¥˜:")
            print(f"     ?ˆìƒ: {expected_type.value}")
            print(f"     ê²°ê³¼: {classified_type.value}")
            print(f"     ?íƒœ: {'?? if classification_correct else '??}")

            # 2. ?µë? êµ¬ì¡°???ŒìŠ¤??
            structure_result = enhancer.enhance_answer_structure(
                answer=answer,
                question_type=classified_type.value,
                question=question
            )

            if "error" not in structure_result:
                print(f"   ?µë? êµ¬ì¡°??")
                print(f"     ?ë³¸ ê¸¸ì´: {len(answer)} ë¬¸ì")
                print(f"     êµ¬ì¡°??ê¸¸ì´: {len(structure_result['structured_answer'])} ë¬¸ì")
                print(f"     ?¬ìš©???œí”Œë¦? {structure_result['template_used']}")
                print(f"     ?„ì²´ ?ìˆ˜: {structure_result['quality_metrics']['overall_score']:.2f}")
                print(f"     ?íƒœ: ??)

                structure_success = True
            else:
                print(f"   ?µë? êµ¬ì¡°??")
                print(f"     ?¤ë¥˜: {structure_result['error']}")
                print(f"     ?íƒœ: ??)
                structure_success = False

            # 3. ?µí•© ?±ê³µ ?¬ë?
            if classification_correct and structure_success:
                success_count += 1
                print(f"   ?„ì²´ ê²°ê³¼: ???±ê³µ")
            else:
                print(f"   ?„ì²´ ê²°ê³¼: ???¤íŒ¨")

            print()

        except Exception as e:
            print(f"   ?¤ë¥˜ ë°œìƒ: {e}")
            print(f"   ?„ì²´ ê²°ê³¼: ???¤íŒ¨")
            print()

    # ê²°ê³¼ ?”ì•½
    success_rate = (success_count / len(test_cases)) * 100

    print("=" * 60)
    print("?µí•© ?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("=" * 60)
    print(f"ì´??ŒìŠ¤??ì¼€?´ìŠ¤: {len(test_cases)}")
    print(f"?±ê³µ??ì¼€?´ìŠ¤: {success_count}")
    print(f"?¤íŒ¨??ì¼€?´ìŠ¤: {len(test_cases) - success_count}")
    print(f"?±ê³µë¥? {success_rate:.1f}%")

    if success_rate >= 90:
        grade = "A+"
    elif success_rate >= 80:
        grade = "A"
    elif success_rate >= 70:
        grade = "B"
    elif success_rate >= 60:
        grade = "C"
    else:
        grade = "D"

    print(f"?±ê¸‰: {grade}")
    print("=" * 60)

    return success_rate


def test_performance_comparison():
    """?±ëŠ¥ ë¹„êµ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?±ëŠ¥ ë¹„êµ ?ŒìŠ¤??)
    print("=" * 60)

    import time

    enhancer = AnswerStructureEnhancer()

    test_questions = [
        "ë¯¼ë²• ??23ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???",
        "ê³„ì•½?œë? ê²€? í•´ì£¼ì„¸??,
        "?´í˜¼ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
        "?ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
        "ë²•ë¥  ?ë‹´???„ìš”?©ë‹ˆ??
    ] * 20  # 100ê°?ì§ˆë¬¸

    test_answer = "?´ê²ƒ?€ ?ŒìŠ¤???µë??…ë‹ˆ?? ë²•ë¥ ???´ìš©???¬í•¨?˜ê³  ?ˆìŠµ?ˆë‹¤."

    print(f"\n{len(test_questions)}ê°?ì§ˆë¬¸???€???±ëŠ¥ ?ŒìŠ¤??ì¤?..")

    # 1. ì§ˆë¬¸ ? í˜• ë¶„ë¥˜ ?±ëŠ¥
    start_time = time.time()
    for question in test_questions:
        enhancer.classify_question_type(question)
    classification_time = time.time() - start_time

    # 2. ?µë? êµ¬ì¡°???±ëŠ¥
    start_time = time.time()
    for question in test_questions:
        classified_type = enhancer.classify_question_type(question)
        enhancer.enhance_answer_structure(
            answer=test_answer,
            question_type=classified_type.value,
            question=question
        )
    structure_time = time.time() - start_time

    # 3. ?µí•© ?±ëŠ¥
    total_time = classification_time + structure_time

    print(f"\n?±ëŠ¥ ê²°ê³¼:")
    print(f"  ì§ˆë¬¸ ? í˜• ë¶„ë¥˜ ?œê°„: {classification_time:.3f}ì´?)
    print(f"  ?µë? êµ¬ì¡°???œê°„: {structure_time:.3f}ì´?)
    print(f"  ì´?ì²˜ë¦¬ ?œê°„: {total_time:.3f}ì´?)
    print(f"  ?‰ê·  ì²˜ë¦¬ ?œê°„: {total_time/len(test_questions)*1000:.2f}ms")
    print(f"  ì´ˆë‹¹ ì²˜ë¦¬?? {len(test_questions)/total_time:.0f} questions/sec")

    print(f"\nê°œë³„ ?±ëŠ¥:")
    print(f"  ë¶„ë¥˜ ?‰ê· : {classification_time/len(test_questions)*1000:.2f}ms")
    print(f"  êµ¬ì¡°???‰ê· : {structure_time/len(test_questions)*1000:.2f}ms")


def test_database_integration():
    """?°ì´?°ë² ?´ìŠ¤ ?µí•© ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?°ì´?°ë² ?´ìŠ¤ ?µí•© ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # ?°ì´?°ë² ?´ìŠ¤?ì„œ ?œí”Œë¦??•ë³´ ì¡°íšŒ
    print("\n1. ?°ì´?°ë² ?´ìŠ¤ ?œí”Œë¦?ì¡°íšŒ ?ŒìŠ¤??)
    print("-" * 30)

    question_types = [
        "law_inquiry",
        "contract_review",
        "divorce_procedure",
        "precedent_search",
        "labor_dispute"
    ]

    for question_type in question_types:
        template_info = enhancer.get_template_info(question_type)
        print(f"  {question_type}:")
        print(f"    ?œëª©: {template_info['title']}")
        print(f"    ?¹ì…˜ ?? {template_info['section_count']}")
        print(f"    ?ŒìŠ¤: {template_info['source']}")
        print()

    # ?™ì  ?œí”Œë¦?ë¦¬ë¡œ???ŒìŠ¤??
    print("\n2. ?™ì  ?œí”Œë¦?ë¦¬ë¡œ???ŒìŠ¤??)
    print("-" * 30)

    print("  ?œí”Œë¦?ë¦¬ë¡œ??ì¤?..")
    enhancer.reload_templates()
    print("  ???œí”Œë¦?ë¦¬ë¡œ???„ë£Œ")

    # ?°ì´?°ë² ?´ìŠ¤ ?µê³„ ì¡°íšŒ
    print("\n3. ?°ì´?°ë² ?´ìŠ¤ ?µê³„ ì¡°íšŒ")
    print("-" * 30)

    # template_db_managerê°€ ?†ëŠ” ê²½ìš°ë¥??€ë¹„í•œ ?ˆì „ ì²˜ë¦¬
    if hasattr(enhancer, 'template_db_manager') and enhancer.template_db_manager:
        try:
            stats = enhancer.template_db_manager.get_template_statistics()
            print(f"  ?„ì²´ ?œí”Œë¦? {stats.get('total_templates', 0)}")
            print(f"  ?œì„± ?œí”Œë¦? {stats.get('active_templates', 0)}")
            print(f"  ?„ì²´ ?¹ì…˜: {stats.get('total_sections', 0)}")
            print(f"  ?œì„± ?¹ì…˜: {stats.get('active_sections', 0)}")
        except Exception as e:
            print(f"  ? ï¸  ?µê³„ ì¡°íšŒ ?¤íŒ¨: {e}")
            print(f"  ?œí”Œë¦??œìŠ¤?œì? ?•ìƒ ?™ì‘ ì¤‘ì…?ˆë‹¤.")
    else:
        # ?œí”Œë¦??œìŠ¤?œì? ?˜ë“œì½”ë”©???œí”Œë¦¿ì„ ?¬ìš©?˜ë?ë¡??µê³„??ì§ì ‘ ê³„ì‚°
        template_count = len(enhancer.structure_templates) if hasattr(enhancer, 'structure_templates') else 0
        print(f"  ?„ì²´ ?œí”Œë¦? {template_count}ê°?(?˜ë“œì½”ë”©)")
        print(f"  ?œí”Œë¦??œìŠ¤?? ?•ìƒ ?™ì‘ ì¤?)


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    print("?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜ ?œí”Œë¦??œìŠ¤??+ classify_question_type ?µí•© ?ŒìŠ¤??)

    try:
        # 1. ?µí•© ?œìŠ¤???ŒìŠ¤??
        success_rate = test_integrated_system()

        # 2. ?±ëŠ¥ ë¹„êµ ?ŒìŠ¤??
        test_performance_comparison()

        # 3. ?°ì´?°ë² ?´ìŠ¤ ?µí•© ?ŒìŠ¤??
        test_database_integration()

        print(f"\n?‰ ëª¨ë“  ?µí•© ?ŒìŠ¤?¸ê? ?„ë£Œ?˜ì—ˆ?µë‹ˆ??")
        print(f"?“Š ìµœì¢… ?±ê³µë¥? {success_rate:.1f}%")

        if success_rate >= 90:
            print("?† ?°ìˆ˜???±ëŠ¥??ë³´ì—¬ì£¼ê³  ?ˆìŠµ?ˆë‹¤!")
        elif success_rate >= 80:
            print("?‘ ?‘í˜¸???±ëŠ¥??ë³´ì—¬ì£¼ê³  ?ˆìŠµ?ˆë‹¤!")
        else:
            print("? ï¸  ê°œì„ ???„ìš”??ë¶€ë¶„ì´ ?ˆìŠµ?ˆë‹¤.")

    except Exception as e:
        print(f"???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
