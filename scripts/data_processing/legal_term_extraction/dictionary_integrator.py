# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•©ê¸?
ì¶”ì¶œ???©ì–´?¤ì„ ê¸°ì¡´ ?¬ì „ê³??µí•©?˜ì—¬ ìµœì¢… ?¬ì „???ì„±
"""

import json
import logging
from typing import Dict, List, Set
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class DictionaryIntegrator:
    """ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•©ê¸?""
    
    def __init__(self):
        """ì´ˆê¸°??""
        self.logger = logging.getLogger(__name__)
        
        # ?µí•© ?¤ì •
        self.integration_settings = {
            "merge_strategy": "enhance",  # enhance, replace, append
            "duplicate_handling": "merge",  # merge, keep_original, keep_new
            "quality_threshold": 0.6,
            "min_frequency": 3
        }
    
    def load_existing_dictionary(self, file_path: str) -> Dict:
        """ê¸°ì¡´ ?¬ì „ ë¡œë“œ"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            self.logger.info(f"Loaded existing dictionary with {len(dictionary)} terms")
            return dictionary
        except FileNotFoundError:
            self.logger.warning(f"Existing dictionary not found: {file_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading existing dictionary: {e}")
            return {}
    
    def load_enhanced_dictionary(self, file_path: str) -> Dict:
        """?¥ìƒ???¬ì „ ë¡œë“œ"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            self.logger.info(f"Loaded enhanced dictionary with {len(dictionary)} terms")
            return dictionary
        except Exception as e:
            self.logger.error(f"Error loading enhanced dictionary: {e}")
            return {}
    
    def merge_dictionaries(self, existing_dict: Dict, enhanced_dict: Dict) -> Dict:
        """?¬ì „ ?µí•©"""
        self.logger.info("Starting dictionary integration")
        
        merged_dict = existing_dict.copy()
        integration_stats = {
            "existing_terms": len(existing_dict),
            "enhanced_terms": len(enhanced_dict),
            "merged_terms": 0,
            "new_terms": 0,
            "updated_terms": 0,
            "rejected_terms": 0
        }
        
        for term, term_info in enhanced_dict.items():
            if term in merged_dict:
                # ê¸°ì¡´ ?©ì–´ ?…ë°?´íŠ¸
                merged_dict[term] = self._merge_term_info(
                    merged_dict[term], term_info
                )
                integration_stats["updated_terms"] += 1
            else:
                # ???©ì–´ ì¶”ê? (ëª¨ë“  ?©ì–´ ì¶”ê?)
                merged_dict[term] = term_info
                integration_stats["new_terms"] += 1
            
            integration_stats["merged_terms"] += 1
        
        self.logger.info(f"Integration completed: {integration_stats}")
        return merged_dict, integration_stats
    
    def _merge_term_info(self, existing_info: Dict, new_info: Dict) -> Dict:
        """?©ì–´ ?•ë³´ ?µí•©"""
        merged_info = existing_info.copy()
        
        # ?™ì˜???µí•©
        existing_synonyms = set(existing_info.get("synonyms", []))
        new_synonyms = set(new_info.get("synonyms", []))
        merged_info["synonyms"] = list(existing_synonyms.union(new_synonyms))
        
        # ê´€???©ì–´ ?µí•©
        existing_related = set(existing_info.get("related_terms", []))
        new_related = set(new_info.get("related_terms", []))
        merged_info["related_terms"] = list(existing_related.union(new_related))
        
        # ê´€??ë²•ë¥  ?µí•©
        existing_laws = set(existing_info.get("related_laws", []))
        new_laws = set(new_info.get("related_laws", []))
        merged_info["related_laws"] = list(existing_laws.union(new_laws))
        
        # ?ë? ?¤ì›Œ???µí•©
        existing_precedents = set(existing_info.get("precedent_keywords", []))
        new_precedents = set(new_info.get("precedent_keywords", []))
        merged_info["precedent_keywords"] = list(existing_precedents.union(new_precedents))
        
        # ? ë¢°???…ë°?´íŠ¸ (???’ì? ê°?? íƒ)
        existing_confidence = existing_info.get("confidence", 0.0)
        new_confidence = new_info.get("confidence", 0.0)
        merged_info["confidence"] = max(existing_confidence, new_confidence)
        
        # ë¹ˆë„ ?…ë°?´íŠ¸ (?©ì‚°)
        existing_frequency = existing_info.get("frequency", 0)
        new_frequency = new_info.get("frequency", 0)
        merged_info["frequency"] = existing_frequency + new_frequency
        
        # ?„ë©”??ë°?ì¹´í…Œê³ ë¦¬ ?…ë°?´íŠ¸
        if "domain" in new_info:
            merged_info["domain"] = new_info["domain"]
        if "category" in new_info:
            merged_info["category"] = new_info["category"]
        
        return merged_info
    
    def _should_add_term(self, term_info: Dict) -> bool:
        """?©ì–´ ì¶”ê? ?¬ë? ?ë‹¨"""
        # ?ˆì§ˆ ê¸°ì? ?•ì¸
        confidence = term_info.get("confidence", 0.0)
        if confidence < self.integration_settings["quality_threshold"]:
            return False
        
        # ë¹ˆë„ ê¸°ì? ?•ì¸
        frequency = term_info.get("frequency", 0)
        if frequency < self.integration_settings["min_frequency"]:
            return False
        
        # ê¸°ë³¸?ì¸ ?•ë³´ê°€ ?ˆìœ¼ë©?ì¶”ê? (??ê´€?€??ê¸°ì?)
        return True
    
    def validate_integrated_dictionary(self, dictionary: Dict) -> Dict:
        """?µí•©???¬ì „ ê²€ì¦?""
        validation_results = {
            "total_terms": len(dictionary),
            "terms_with_synonyms": 0,
            "terms_with_related_terms": 0,
            "terms_with_related_laws": 0,
            "terms_with_precedent_keywords": 0,
            "high_confidence_terms": 0,
            "domain_distribution": defaultdict(int),
            "category_distribution": defaultdict(int)
        }
        
        for term, term_info in dictionary.items():
            # ?„ìˆ˜ ?„ë“œ ?•ì¸
            if term_info.get("synonyms"):
                validation_results["terms_with_synonyms"] += 1
            if term_info.get("related_terms"):
                validation_results["terms_with_related_terms"] += 1
            if term_info.get("related_laws"):
                validation_results["terms_with_related_laws"] += 1
            if term_info.get("precedent_keywords"):
                validation_results["terms_with_precedent_keywords"] += 1
            
            # ? ë¢°???•ì¸
            if term_info.get("confidence", 0.0) >= 0.8:
                validation_results["high_confidence_terms"] += 1
            
            # ?„ë©”??ë°?ì¹´í…Œê³ ë¦¬ ë¶„í¬
            domain = term_info.get("domain", "ê¸°í?")
            category = term_info.get("category", "ê¸°í?")
            validation_results["domain_distribution"][domain] += 1
            validation_results["category_distribution"][category] += 1
        
        return validation_results
    
    def generate_integration_report(self, integration_stats: Dict, validation_results: Dict) -> str:
        """?µí•© ë³´ê³ ???ì„±"""
        report = []
        report.append("=== ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•© ë³´ê³ ??===\n")
        
        # ?µí•© ?µê³„
        report.append("1. ?µí•© ?µê³„")
        report.append(f"   ê¸°ì¡´ ?©ì–´ ?? {integration_stats['existing_terms']}")
        report.append(f"   ?¥ìƒ???©ì–´ ?? {integration_stats['enhanced_terms']}")
        report.append(f"   ?µí•©???©ì–´ ?? {integration_stats['merged_terms']}")
        report.append(f"   ?ˆë¡œ ì¶”ê????©ì–´: {integration_stats['new_terms']}")
        report.append(f"   ?…ë°?´íŠ¸???©ì–´: {integration_stats['updated_terms']}")
        report.append(f"   ?œì™¸???©ì–´: {integration_stats['rejected_terms']}")
        report.append("")
        
        # ê²€ì¦?ê²°ê³¼
        report.append("2. ê²€ì¦?ê²°ê³¼")
        report.append(f"   ì´??©ì–´ ?? {validation_results['total_terms']}")
        report.append(f"   ?™ì˜?´ê? ?ˆëŠ” ?©ì–´: {validation_results['terms_with_synonyms']} ({validation_results['terms_with_synonyms']/validation_results['total_terms']:.1%})")
        report.append(f"   ê´€???©ì–´ê°€ ?ˆëŠ” ?©ì–´: {validation_results['terms_with_related_terms']} ({validation_results['terms_with_related_terms']/validation_results['total_terms']:.1%})")
        report.append(f"   ê´€??ë²•ë¥ ???ˆëŠ” ?©ì–´: {validation_results['terms_with_related_laws']} ({validation_results['terms_with_related_laws']/validation_results['total_terms']:.1%})")
        report.append(f"   ?ë? ?¤ì›Œ?œê? ?ˆëŠ” ?©ì–´: {validation_results['terms_with_precedent_keywords']} ({validation_results['terms_with_precedent_keywords']/validation_results['total_terms']:.1%})")
        report.append(f"   ê³ ì‹ ë¢°ë„ ?©ì–´: {validation_results['high_confidence_terms']} ({validation_results['high_confidence_terms']/validation_results['total_terms']:.1%})")
        report.append("")
        
        # ?„ë©”??ë¶„í¬
        report.append("3. ?„ë©”?¸ë³„ ë¶„í¬")
        for domain, count in validation_results["domain_distribution"].items():
            percentage = count / validation_results["total_terms"] * 100
            report.append(f"   {domain}: {count}ê°?({percentage:.1f}%)")
        report.append("")
        
        # ì¹´í…Œê³ ë¦¬ ë¶„í¬
        report.append("4. ì¹´í…Œê³ ë¦¬ë³?ë¶„í¬ (?ìœ„ 10ê°?")
        sorted_categories = sorted(validation_results["category_distribution"].items(), 
                                 key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            percentage = count / validation_results["total_terms"] * 100
            report.append(f"   {category}: {count}ê°?({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def save_integrated_dictionary(self, dictionary: Dict, output_path: str):
        """?µí•©???¬ì „ ?€??""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Integrated dictionary saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving integrated dictionary: {e}")


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•©ê¸?)
    parser.add_argument('--existing_dict', type=str, 
                       default='data/legal_term_dictionary.json',
                       help='ê¸°ì¡´ ?¬ì „ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--enhanced_dict', type=str, required=True,
                       help='?¥ìƒ???¬ì „ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--output_file', type=str,
                       default='data/enhanced_legal_term_dictionary.json',
                       help='?µí•©???¬ì „ ì¶œë ¥ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--quality_threshold', type=float, default=0.6,
                       help='?ˆì§ˆ ê¸°ì? (ê¸°ë³¸ê°? 0.6)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ë¡œê·¸ ?ˆë²¨')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?¬ì „ ?µí•© ?¤í–‰
    integrator = DictionaryIntegrator()
    
    # ê¸°ì¡´ ?¬ì „ ë¡œë“œ
    existing_dict = integrator.load_existing_dictionary(args.existing_dict)
    
    # ?¥ìƒ???¬ì „ ë¡œë“œ
    enhanced_dict = integrator.load_enhanced_dictionary(args.enhanced_dict)
    
    # ?¬ì „ ?µí•©
    merged_dict, integration_stats = integrator.merge_dictionaries(
        existing_dict, enhanced_dict
    )
    
    # ?µí•©???¬ì „ ê²€ì¦?
    validation_results = integrator.validate_integrated_dictionary(merged_dict)
    
    # ?µí•© ë³´ê³ ???ì„± ë°?ì¶œë ¥
    report = integrator.generate_integration_report(integration_stats, validation_results)
    print(report)
    
    # ?µí•©???¬ì „ ?€??
    integrator.save_integrated_dictionary(merged_dict, args.output_file)
    
    print(f"\n?µí•©???¬ì „??{args.output_file}???€?¥ë˜?ˆìŠµ?ˆë‹¤.")
    print(f"ì´?{len(merged_dict)}ê°œì˜ ?©ì–´ê°€ ?¬í•¨?˜ì–´ ?ˆìŠµ?ˆë‹¤.")


if __name__ == "__main__":
    main()
