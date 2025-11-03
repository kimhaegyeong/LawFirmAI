#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?µë? ?ˆì§ˆ ?¥ìƒ ?œìŠ¤???ŒìŠ¤??
?¨ê¸° ê°œì„  ë°©ì•ˆ ?ìš© ê²°ê³¼ ê²€ì¦?
"""

import json
import os
import sys
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from source.services.answer_quality_enhancer import AnswerQualityEnhancer
from source.services.answer_structure_enhancer import AnswerStructureEnhancer
from source.services.context_quality_enhancer import ContextQualityEnhancer
from source.services.keyword_coverage_enhancer import KeywordCoverageEnhancer
from source.services.legal_term_validator import LegalTermValidator


def test_keyword_coverage_enhancement():
    """?¤ì›Œ???¬í•¨???¥ìƒ ?ŒìŠ¤??""
    print("=" * 60)
    print("?¤ì›Œ???¬í•¨???¥ìƒ ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = KeywordCoverageEnhancer()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤
    test_cases = [
        {
            "answer": "ê³„ì•½??ê²€? ëŠ” ì¤‘ìš”?©ë‹ˆ?? ?¹ì‚¬?ì? ì¡°ê±´???•ì¸?´ì•¼ ?©ë‹ˆ??",
            "query_type": "contract_review",
            "question": "ê³„ì•½??ê²€????ì£¼ì˜?¬í•­?€ ë¬´ì—‡?¸ê???",
            "expected_improvement": True
        },
        {
            "answer": "?´í˜¼ ?ˆì°¨??ë³µì¡?©ë‹ˆ?? ë²•ì›??? ì²­?´ì•¼ ?©ë‹ˆ??",
            "query_type": "divorce_procedure",
            "question": "?´í˜¼ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
            "expected_improvement": True
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n?ŒìŠ¤??ì¼€?´ìŠ¤ {i}:")
        print(f"ì§ˆë¬¸: {case['question']}")
        print(f"?ë³¸ ?µë?: {case['answer']}")

        # ?¤ì›Œ???¬í•¨??ë¶„ì„
        analysis = enhancer.analyze_keyword_coverage(
            case['answer'], case['query_type'], case['question']
        )

        print(f"?„ì¬ ?¬í•¨?? {analysis.get('current_coverage', 0.0):.2f}")
        print(f"ëª©í‘œ ?¬í•¨?? {analysis.get('target_coverage', 0.7):.2f}")
        print(f"ê°œì„  ?„ìš”: {analysis.get('needs_improvement', False)}")

        # ?¥ìƒ ?œì•ˆ
        enhancement = enhancer.enhance_keyword_coverage(
            case['answer'], case['query_type'], case['question']
        )

        if enhancement.get('status') == 'needs_improvement':
            print("ê°œì„  ?œì•ˆ:")
            for action in enhancement.get('action_plan', []):
                print(f"  - {action}")
        else:
            print("ëª©í‘œ ?¬í•¨?„ë? ?¬ì„±?ˆìŠµ?ˆë‹¤!")


def test_answer_structure_enhancement():
    """?µë? êµ¬ì¡°???¥ìƒ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?µë? êµ¬ì¡°???¥ìƒ ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = AnswerStructureEnhancer()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤
    test_cases = [
        {
            "answer": "ê³„ì•½??ê²€? ëŠ” ì¤‘ìš”?©ë‹ˆ?? ?¹ì‚¬?ì? ì¡°ê±´???•ì¸?´ì•¼ ?©ë‹ˆ?? ë²•ì  ê·¼ê±°???„ìš”?©ë‹ˆ??",
            "question_type": "contract_review",
            "question": "ê³„ì•½??ê²€????ì£¼ì˜?¬í•­?€ ë¬´ì—‡?¸ê???",
            "domain": "ë¯¼ì‚¬ë²?
        },
        {
            "answer": "?´í˜¼ ?ˆì°¨???‘ì˜?´í˜¼, ì¡°ì •?´í˜¼, ?¬íŒ?´í˜¼???ˆìŠµ?ˆë‹¤. ê°ê° ?¤ë¥¸ ?ˆì°¨ë¥??°ë¦…?ˆë‹¤.",
            "question_type": "divorce_procedure",
            "question": "?´í˜¼ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
            "domain": "ê°€ì¡±ë²•"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n?ŒìŠ¤??ì¼€?´ìŠ¤ {i}:")
        print(f"ì§ˆë¬¸: {case['question']}")
        print(f"?ë³¸ ?µë?: {case['answer']}")

        # êµ¬ì¡°???¥ìƒ
        enhancement = enhancer.enhance_answer_structure(
            case['answer'], case['question_type'], case['question'], case['domain']
        )

        if 'structured_answer' in enhancement:
            print(f"êµ¬ì¡°?”ëœ ?µë?:\n{enhancement['structured_answer']}")

        quality_metrics = enhancement.get('quality_metrics', {})
        print(f"êµ¬ì¡°???ˆì§ˆ ?ìˆ˜: {quality_metrics.get('overall_score', 0.0):.2f}")


def test_legal_term_validation():
    """ë²•ë¥  ?©ì–´ ?•í™•??ê²€ì¦??ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("ë²•ë¥  ?©ì–´ ?•í™•??ê²€ì¦??ŒìŠ¤??)
    print("=" * 60)

    validator = LegalTermValidator()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤
    test_cases = [
        {
            "answer": "ê³„ì•½?€ ?¹ì‚¬??ê°„ì˜ ?˜ì‚¬?œì‹œ???©ì¹˜???˜í•˜???±ë¦½?©ë‹ˆ?? ë¯¼ë²• ??05ì¡°ì— ?°ë¥´ë©?ê³„ì•½?€ ?¹ì‚¬??ê°„ì˜ ?˜ì‚¬?œì‹œ???©ì¹˜???˜í•˜???±ë¦½?©ë‹ˆ??",
            "domain": "ë¯¼ì‚¬ë²?,
            "expected_accuracy": 0.8
        },
        {
            "answer": "?´í˜¼?€ ?¼ì¸ê´€ê³„ë? ?´ì†Œ?˜ëŠ” ë²•ë¥ ?‰ìœ„?…ë‹ˆ?? ?‘ì˜?´í˜¼, ì¡°ì •?´í˜¼, ?¬íŒ?´í˜¼??ë°©ë²•???ˆìŠµ?ˆë‹¤.",
            "domain": "ê°€ì¡±ë²•",
            "expected_accuracy": 0.7
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n?ŒìŠ¤??ì¼€?´ìŠ¤ {i}:")
        print(f"?ë³¸ ?µë?: {case['answer']}")

        # ?©ì–´ ?•í™•??ê²€ì¦?
        validation = validator.validate_legal_terms(case['answer'], case['domain'])

        print(f"ì¶”ì¶œ???©ì–´: {validation.get('extracted_terms', [])}")
        print(f"?„ì²´ ?•í™•?? {validation.get('overall_accuracy', 0.0):.2f}")

        # ?¥ìƒ ?œì•ˆ
        enhancement = validator.enhance_term_accuracy(case['answer'], case['domain'])

        if enhancement.get('status') == 'needs_improvement':
            print("ê°œì„  ?œì•ˆ:")
            for action in enhancement.get('action_plan', []):
                print(f"  - {action}")


def test_context_quality_enhancement():
    """ì»¨í…?¤íŠ¸ ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("ì»¨í…?¤íŠ¸ ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = ContextQualityEnhancer()

    # ?ŒìŠ¤??ê²€??ê²°ê³¼
    test_search_results = [
        {
            "title": "ê³„ì•½??ê²€??ê°€?´ë“œ",
            "content": "ê³„ì•½??ê²€?????¹ì‚¬?? ëª©ì , ì¡°ê±´, ê¸°ê°„???•ì¸?´ì•¼ ?©ë‹ˆ?? ë¯¼ë²• ??05ì¡°ì— ?°ë¥´ë©?ê³„ì•½?€ ?¹ì‚¬??ê°„ì˜ ?˜ì‚¬?œì‹œ???©ì¹˜???˜í•˜???±ë¦½?©ë‹ˆ??",
            "source": "ë²•ë¬´ë¶€"
        },
        {
            "title": "ê³„ì•½???‘ì„± ?”ë ¹",
            "content": "ê³„ì•½???‘ì„± ??ëª…í™•??ì¡°ê±´ê³?ê¸°ê°„??ëª…ì‹œ?´ì•¼ ?©ë‹ˆ?? ?´ì? ì¡°ê±´ê³??„ì•½ê¸?ì¡°í•­???¬í•¨?˜ëŠ” ê²ƒì´ ì¢‹ìŠµ?ˆë‹¤.",
            "source": "ë²•ë¥ ? ë¬¸"
        },
        {
            "title": "ê³„ì•½ ë¶„ìŸ ?´ê²°",
            "content": "ê³„ì•½ ë¶„ìŸ ë°œìƒ ??ì¡°ì •, ì¤‘ì¬, ?Œì†¡??ë°©ë²•???ˆìŠµ?ˆë‹¤. ê°ê°???¥ë‹¨?ì„ ê³ ë ¤?˜ì—¬ ? íƒ?´ì•¼ ?©ë‹ˆ??",
            "source": "?€ë²•ì›"
        }
    ]

    query = "ê³„ì•½??ê²€????ì£¼ì˜?¬í•­?€ ë¬´ì—‡?¸ê???"
    question_type = "contract_review"
    domain = "ë¯¼ì‚¬ë²?

    print(f"ì§ˆë¬¸: {query}")
    print(f"ê²€??ê²°ê³¼ ?? {len(test_search_results)}")

    # ì»¨í…?¤íŠ¸ ?ˆì§ˆ ?¥ìƒ
    enhancement = enhancer.enhance_context_quality(
        test_search_results, query, question_type, domain
    )

    if 'optimized_context' in enhancement:
        print(f"ìµœì ?”ëœ ì»¨í…?¤íŠ¸ ê¸¸ì´: {enhancement['optimized_context'].get('total_length', 0)}")
        print(f"? íƒ???ŒìŠ¤ ?? {enhancement['optimized_context'].get('source_count', 0)}")

    quality_metrics = enhancement.get('quality_metrics', {})
    print(f"ì»¨í…?¤íŠ¸ ?ˆì§ˆ ?ìˆ˜: {quality_metrics.get('overall_quality', 0.0):.2f}")


def test_integrated_quality_enhancement():
    """?µí•© ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??""
    print("\n" + "=" * 60)
    print("?µí•© ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??)
    print("=" * 60)

    enhancer = AnswerQualityEnhancer()

    # ?ŒìŠ¤??ì¼€?´ìŠ¤
    test_case = {
        "answer": "ê³„ì•½??ê²€? ëŠ” ì¤‘ìš”?©ë‹ˆ?? ?¹ì‚¬?ì? ì¡°ê±´???•ì¸?´ì•¼ ?©ë‹ˆ??",
        "query": "ê³„ì•½??ê²€????ì£¼ì˜?¬í•­?€ ë¬´ì—‡?¸ê???",
        "question_type": "contract_review",
        "domain": "ë¯¼ì‚¬ë²?,
        "search_results": [
            {
                "title": "ê³„ì•½??ê²€??ê°€?´ë“œ",
                "content": "ê³„ì•½??ê²€?????¹ì‚¬?? ëª©ì , ì¡°ê±´, ê¸°ê°„???•ì¸?´ì•¼ ?©ë‹ˆ?? ë¯¼ë²• ??05ì¡°ì— ?°ë¥´ë©?ê³„ì•½?€ ?¹ì‚¬??ê°„ì˜ ?˜ì‚¬?œì‹œ???©ì¹˜???˜í•˜???±ë¦½?©ë‹ˆ??",
                "source": "ë²•ë¬´ë¶€"
            }
        ],
        "sources": [
            {"title": "ê³„ì•½??ê²€??ê°€?´ë“œ", "content": "ê³„ì•½??ê²€?????¹ì‚¬?? ëª©ì , ì¡°ê±´, ê¸°ê°„???•ì¸?´ì•¼ ?©ë‹ˆ??", "source": "ë²•ë¬´ë¶€"}
        ]
    }

    print(f"ì§ˆë¬¸: {test_case['query']}")
    print(f"?ë³¸ ?µë?: {test_case['answer']}")

    # ?µí•© ?ˆì§ˆ ?¥ìƒ
    enhancement_result = enhancer.enhance_answer_quality(
        test_case['answer'],
        test_case['query'],
        test_case['question_type'],
        test_case['domain'],
        test_case['search_results'],
        test_case['sources']
    )

    if 'error' not in enhancement_result:
        print(f"\n?¥ìƒ???µë?:\n{enhancement_result.get('enhanced_answer', '')}")

        # ?ˆì§ˆ ?¥ìƒ ë³´ê³ ??
        quality_report = enhancer.get_quality_report(enhancement_result)

        print(f"\n?ˆì§ˆ ?¥ìƒ ë³´ê³ ??")
        print(f"?ë³¸ ?ˆì§ˆ: {quality_report['summary']['original_quality']:.2f}")
        print(f"ìµœì¢… ?ˆì§ˆ: {quality_report['summary']['final_quality']:.2f}")
        print(f"ê°œì„  ?¨ê³¼: {quality_report['summary']['improvement']:.2f}%")

        print(f"\n?ì„¸ ë©”íŠ¸ë¦?")
        for metric, data in quality_report['detailed_metrics'].items():
            print(f"  {metric}: {data['original']:.2f} ??{data['final']:.2f} ({data['improvement']:.2f}%)")

        print(f"\nê¶Œì¥?¬í•­:")
        for recommendation in quality_report['recommendations']:
            print(f"  - {recommendation}")
    else:
        print(f"?¤ë¥˜ ë°œìƒ: {enhancement_result['error']}")


def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("LawFirmAI ?µë? ?ˆì§ˆ ?¥ìƒ ?œìŠ¤???ŒìŠ¤??)
    print("=" * 60)
    print(f"?ŒìŠ¤???œì‘ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. ?¤ì›Œ???¬í•¨???¥ìƒ ?ŒìŠ¤??
        test_keyword_coverage_enhancement()

        # 2. ?µë? êµ¬ì¡°???¥ìƒ ?ŒìŠ¤??
        test_answer_structure_enhancement()

        # 3. ë²•ë¥  ?©ì–´ ?•í™•??ê²€ì¦??ŒìŠ¤??
        test_legal_term_validation()

        # 4. ì»¨í…?¤íŠ¸ ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??
        test_context_quality_enhancement()

        # 5. ?µí•© ?ˆì§ˆ ?¥ìƒ ?ŒìŠ¤??
        test_integrated_quality_enhancement()

        print("\n" + "=" * 60)
        print("ëª¨ë“  ?ŒìŠ¤?¸ê? ?„ë£Œ?˜ì—ˆ?µë‹ˆ??")
        print("=" * 60)

    except Exception as e:
        print(f"\n?ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
