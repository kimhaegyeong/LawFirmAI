# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?¬ì „ ?•ì¥ ?ŒìŠ¤???¤í¬ë¦½íŠ¸
"""

import json
import logging
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.legal_term_extraction.term_extractor import LegalTermExtractor
from scripts.data_processing.legal_term_extraction.domain_expander import DomainTermExpander
from scripts.data_processing.legal_term_extraction.quality_validator import QualityValidator
from scripts.data_processing.legal_term_extraction.dictionary_integrator import DictionaryIntegrator

logger = logging.getLogger(__name__)


def test_term_extraction():
    """?©ì–´ ì¶”ì¶œ ?ŒìŠ¤??""
    print("=== ?©ì–´ ì¶”ì¶œ ?ŒìŠ¤??===")
    
    extractor = LegalTermExtractor()
    
    # ?ŒìŠ¤???°ì´???”ë ‰? ë¦¬
    test_data_dir = "data/processed/assembly/law/20251013_ml/20251010"
    
    if not Path(test_data_dir).exists():
        print(f"?ŒìŠ¤???°ì´???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {test_data_dir}")
        return False
    
    try:
        # ?©ì–´ ì¶”ì¶œ ?¤í–‰
        results = extractor.process_directory(test_data_dir, min_frequency=2)
        
        print(f"ì²˜ë¦¬???Œì¼: {results['processed_files']}")
        print(f"ì¶”ì¶œ??ì´??©ì–´: {results['total_terms_extracted']}")
        print(f"?„í„°ë§????©ì–´: {sum(len(terms) for terms in results['filtered_terms'].values())}")
        
        # ?¨í„´ë³??©ì–´ ??ì¶œë ¥
        print("\n?¨í„´ë³??©ì–´ ??")
        for pattern_name, terms in results['filtered_terms'].items():
            print(f"  {pattern_name}: {len(terms)}ê°?)
        
        # ?ìœ„ ë¹ˆë„ ?©ì–´ ì¶œë ¥
        print("\n?ìœ„ ë¹ˆë„ ?©ì–´ (?ìœ„ 10ê°?:")
        for term, freq in list(results['term_frequencies'].items())[:10]:
            print(f"  {term}: {freq}??)
        
        return True
        
    except Exception as e:
        print(f"?©ì–´ ì¶”ì¶œ ?ŒìŠ¤???¤íŒ¨: {e}")
        return False


def test_domain_expansion():
    """?„ë©”???•ì¥ ?ŒìŠ¤??""
    print("\n=== ?„ë©”???•ì¥ ?ŒìŠ¤??===")
    
    expander = DomainTermExpander()
    
    # ?ŒìŠ¤?¸ìš© ì¶”ì¶œ???©ì–´ ?°ì´??
    test_extracted_terms = {
        "legal_concepts": ["?í•´ë°°ìƒ", "ê³„ì•½", "?Œì†¡", "?•ë²Œ"],
        "legal_actions": ["ë°°ìƒ", "ë³´ìƒ", "ì²?µ¬", "?œê¸°"],
        "legal_procedures": ["?ˆì°¨", "? ì²­", "?¬ë¦¬", "?ê²°"],
        "legal_entities": ["ë²•ì›", "ê²€??, "ë³€?¸ì‚¬", "?¼ê³ ??],
        "legal_documents": ["?Œì¥", "?µë???, "ì¦ê±°", "?ê²°??]
    }
    
    try:
        # ?„ë©”?¸ë³„ ?©ì–´ ?•ì¥
        domain_terms = expander.expand_domain_terms(test_extracted_terms)
        
        print("?„ë©”?¸ë³„ ?©ì–´ ??")
        for domain, categories in domain_terms.items():
            total_terms = sum(len(terms) for terms in categories.values())
            print(f"  {domain}: {total_terms}ê°?)
        
        # ?¥ìƒ???¬ì „ ?ì„±
        enhanced_dict = expander.generate_enhanced_dictionary(test_extracted_terms, domain_terms)
        
        print(f"\n?¥ìƒ???¬ì „ ì´??©ì–´ ?? {len(enhanced_dict)}")
        
        # ?˜í”Œ ?©ì–´ ?•ë³´ ì¶œë ¥
        print("\n?˜í”Œ ?©ì–´ ?•ë³´:")
        sample_terms = list(enhanced_dict.keys())[:3]
        for term in sample_terms:
            info = enhanced_dict[term]
            print(f"  {term}:")
            print(f"    ?™ì˜?? {info.get('synonyms', [])}")
            print(f"    ê´€???©ì–´: {info.get('related_terms', [])}")
            print(f"    ê´€??ë²•ë¥ : {info.get('related_laws', [])}")
            print(f"    ? ë¢°?? {info.get('confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"?„ë©”???•ì¥ ?ŒìŠ¤???¤íŒ¨: {e}")
        return False


def test_quality_validation():
    """?ˆì§ˆ ê²€ì¦??ŒìŠ¤??""
    print("\n=== ?ˆì§ˆ ê²€ì¦??ŒìŠ¤??===")
    
    validator = QualityValidator()
    
    # ?ŒìŠ¤?¸ìš© ?¬ì „ ?°ì´??
    test_dictionary = {
        "?í•´ë°°ìƒ": {
            "synonyms": ["ë°°ìƒ", "ë³´ìƒ", "?¼í•´ë³´ìƒ"],
            "related_terms": ["ë¶ˆë²•?‰ìœ„", "ì±„ë¬´ë¶ˆì´??, "ê³¼ì‹¤", "ê³ ì˜"],
            "related_laws": ["ë¯¼ë²• ??50ì¡?, "ë¯¼ë²• ??51ì¡?],
            "precedent_keywords": ["?í•´ë°°ìƒì²?µ¬ê¶?, "ë°°ìƒì±…ì„"],
            "confidence": 0.9,
            "frequency": 10
        },
        "ê³„ì•½": {
            "synonyms": ["ê³„ì•½??, "?½ì •"],
            "related_terms": ["ê³„ì•½?´ì?", "ê³„ì•½?„ë°˜"],
            "related_laws": ["ë¯¼ë²• ??05ì¡?],
            "precedent_keywords": ["ê³„ì•½?´ì?ê¶?],
            "confidence": 0.8,
            "frequency": 8
        },
        "?€?ˆì§ˆ?©ì–´": {
            "synonyms": [],
            "related_terms": [],
            "related_laws": [],
            "precedent_keywords": [],
            "confidence": 0.3,
            "frequency": 1
        }
    }
    
    try:
        # ?ˆì§ˆ ê²€ì¦??¤í–‰
        validation_summary = validator.validate_dictionary_quality(test_dictionary)
        
        print("?ˆì§ˆ ê²€ì¦?ê²°ê³¼:")
        print(f"  ì´??©ì–´ ?? {validation_summary['total_terms']}")
        print(f"  ê³ í’ˆì§??©ì–´: {validation_summary['high_quality_terms']}")
        print(f"  ì¤‘í’ˆì§??©ì–´: {validation_summary['medium_quality_terms']}")
        print(f"  ?€?ˆì§ˆ ?©ì–´: {validation_summary['low_quality_terms']}")
        print(f"  ?œì™¸???©ì–´: {validation_summary['rejected_terms']}")
        
        # ê°œì„  ?œì•ˆ ?ì„±
        suggestions = validator.generate_improvement_suggestions(test_dictionary)
        
        print("\nê°œì„  ?œì•ˆ:")
        for suggestion in suggestions["overall_suggestions"]:
            print(f"  ??{suggestion}")
        
        # ê³ í’ˆì§??©ì–´ ?„í„°ë§?
        high_quality_dict = validator.filter_high_quality_terms(test_dictionary)
        
        print(f"\nê³ í’ˆì§??©ì–´ ?? {len(high_quality_dict)}")
        
        return True
        
    except Exception as e:
        print(f"?ˆì§ˆ ê²€ì¦??ŒìŠ¤???¤íŒ¨: {e}")
        return False


def test_dictionary_integration():
    """?¬ì „ ?µí•© ?ŒìŠ¤??""
    print("\n=== ?¬ì „ ?µí•© ?ŒìŠ¤??===")
    
    integrator = DictionaryIntegrator()
    
    # ê¸°ì¡´ ?¬ì „ (?ŒìŠ¤?¸ìš©)
    existing_dict = {
        "?í•´ë°°ìƒ": {
            "synonyms": ["ë°°ìƒ", "ë³´ìƒ"],
            "related_terms": ["ë¶ˆë²•?‰ìœ„", "ì±„ë¬´ë¶ˆì´??],
            "related_laws": ["ë¯¼ë²• ??50ì¡?],
            "precedent_keywords": ["?í•´ë°°ìƒì²?µ¬ê¶?],
            "confidence": 0.8,
            "frequency": 5
        }
    }
    
    # ?¥ìƒ???¬ì „ (?ŒìŠ¤?¸ìš©)
    enhanced_dict = {
        "?í•´ë°°ìƒ": {
            "synonyms": ["ë°°ìƒ", "ë³´ìƒ", "?¼í•´ë³´ìƒ"],
            "related_terms": ["ë¶ˆë²•?‰ìœ„", "ì±„ë¬´ë¶ˆì´??, "ê³¼ì‹¤", "ê³ ì˜"],
            "related_laws": ["ë¯¼ë²• ??50ì¡?, "ë¯¼ë²• ??51ì¡?],
            "precedent_keywords": ["?í•´ë°°ìƒì²?µ¬ê¶?, "ë°°ìƒì±…ì„"],
            "confidence": 0.9,
            "frequency": 10
        },
        "ê³„ì•½": {
            "synonyms": ["ê³„ì•½??, "?½ì •", "?©ì˜"],
            "related_terms": ["ê³„ì•½?´ì?", "ê³„ì•½?„ë°˜", "ê³„ì•½?´í–‰"],
            "related_laws": ["ë¯¼ë²• ??05ì¡?, "ë¯¼ë²• ??43ì¡?],
            "precedent_keywords": ["ê³„ì•½?´ì?ê¶?, "ê³„ì•½?„ë°˜"],
            "confidence": 0.8,
            "frequency": 8
        }
    }
    
    try:
        # ?¬ì „ ?µí•© ?¤í–‰
        merged_dict, integration_stats = integrator.merge_dictionaries(
            existing_dict, enhanced_dict
        )
        
        print("?µí•© ê²°ê³¼:")
        print(f"  ê¸°ì¡´ ?©ì–´ ?? {integration_stats['existing_terms']}")
        print(f"  ?¥ìƒ???©ì–´ ?? {integration_stats['enhanced_terms']}")
        print(f"  ?µí•©???©ì–´ ?? {integration_stats['merged_terms']}")
        print(f"  ?ˆë¡œ ì¶”ê????©ì–´: {integration_stats['new_terms']}")
        print(f"  ?…ë°?´íŠ¸???©ì–´: {integration_stats['updated_terms']}")
        print(f"  ?œì™¸???©ì–´: {integration_stats['rejected_terms']}")
        
        # ?µí•©???¬ì „ ê²€ì¦?
        validation_results = integrator.validate_integrated_dictionary(merged_dict)
        
        print(f"\n?µí•©???¬ì „ ê²€ì¦?")
        print(f"  ì´??©ì–´ ?? {validation_results['total_terms']}")
        print(f"  ?™ì˜?´ê? ?ˆëŠ” ?©ì–´: {validation_results['terms_with_synonyms']}")
        print(f"  ê´€???©ì–´ê°€ ?ˆëŠ” ?©ì–´: {validation_results['terms_with_related_terms']}")
        print(f"  ê´€??ë²•ë¥ ???ˆëŠ” ?©ì–´: {validation_results['terms_with_related_laws']}")
        print(f"  ê³ ì‹ ë¢°ë„ ?©ì–´: {validation_results['high_confidence_terms']}")
        
        return True
        
    except Exception as e:
        print(f"?¬ì „ ?µí•© ?ŒìŠ¤???¤íŒ¨: {e}")
        return False


def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("ë²•ë¥  ?©ì–´ ?¬ì „ ?•ì¥ ?œìŠ¤???ŒìŠ¤???œì‘\n")
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_results = []
    
    # ê°??¨ê³„ë³??ŒìŠ¤???¤í–‰
    test_results.append(("?©ì–´ ì¶”ì¶œ", test_term_extraction()))
    test_results.append(("?„ë©”???•ì¥", test_domain_expansion()))
    test_results.append(("?ˆì§ˆ ê²€ì¦?, test_quality_validation()))
    test_results.append(("?¬ì „ ?µí•©", test_dictionary_integration()))
    
    # ?ŒìŠ¤??ê²°ê³¼ ?”ì•½
    print("\n=== ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
    passed_tests = 0
    for test_name, result in test_results:
        status = "?µê³¼" if result else "?¤íŒ¨"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\nì´?{len(test_results)}ê°??ŒìŠ¤??ì¤?{passed_tests}ê°??µê³¼")
    
    if passed_tests == len(test_results):
        print("ëª¨ë“  ?ŒìŠ¤?¸ê? ?±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
        return True
    else:
        print("?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤. ë¡œê·¸ë¥??•ì¸?´ì£¼?¸ìš”.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
