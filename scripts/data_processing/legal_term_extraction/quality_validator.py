# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦??œìŠ¤??
ì¶”ì¶œ???©ì–´?¤ì˜ ?ˆì§ˆ??ê²€ì¦í•˜ê³?ê°œì„ ?¬í•­???œì•ˆ
"""

import json
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class QualityValidator:
    """ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦ê¸°"""
    
    def __init__(self):
        """ì´ˆê¸°??""
        self.logger = logging.getLogger(__name__)
        
        # ?ˆì§ˆ ê²€ì¦?ê¸°ì?
        self.quality_criteria = {
            "min_synonyms": 0,      # ìµœì†Œ ?™ì˜????
            "min_related_terms": 0,  # ìµœì†Œ ê´€???©ì–´ ??
            "min_related_laws": 0,   # ìµœì†Œ ê´€??ë²•ë¥  ??
            "min_precedent_keywords": 0,  # ìµœì†Œ ?ë? ?¤ì›Œ????
            "min_confidence": 0.3,   # ìµœì†Œ ? ë¢°??
            "min_frequency": 1       # ìµœì†Œ ë¹ˆë„
        }
        
        # ?ˆì§ˆ ?ìˆ˜ ê°€ì¤‘ì¹˜
        self.quality_weights = {
            "synonyms": 0.2,        # ?™ì˜??20%
            "related_terms": 0.25,   # ê´€???©ì–´ 25%
            "related_laws": 0.25,    # ê´€??ë²•ë¥  25%
            "precedent_keywords": 0.15,  # ?ë? ?¤ì›Œ??15%
            "confidence": 0.15      # ? ë¢°??15%
        }
        
        # ê²€ì¦?ê²°ê³¼ ?€??
        self.validation_results = defaultdict(dict)
        self.quality_issues = defaultdict(list)
        self.improvement_suggestions = defaultdict(list)
    
    def validate_term_quality(self, term: str, term_info: Dict) -> Tuple[float, List[str]]:
        """ê°œë³„ ?©ì–´ ?ˆì§ˆ ê²€ì¦?""
        quality_score = 0.0
        issues = []
        
        # 1. ?™ì˜??ê²€ì¦?
        synonyms = term_info.get("synonyms", [])
        if len(synonyms) >= self.quality_criteria["min_synonyms"]:
            quality_score += self.quality_weights["synonyms"]
        else:
            issues.append(f"?™ì˜??ë¶€ì¡?(?„ì¬: {len(synonyms)}, ìµœì†Œ: {self.quality_criteria['min_synonyms']})")
        
        # 2. ê´€???©ì–´ ê²€ì¦?
        related_terms = term_info.get("related_terms", [])
        if len(related_terms) >= self.quality_criteria["min_related_terms"]:
            quality_score += self.quality_weights["related_terms"]
        else:
            issues.append(f"ê´€???©ì–´ ë¶€ì¡?(?„ì¬: {len(related_terms)}, ìµœì†Œ: {self.quality_criteria['min_related_terms']})")
        
        # 3. ê´€??ë²•ë¥  ê²€ì¦?
        related_laws = term_info.get("related_laws", [])
        if len(related_laws) >= self.quality_criteria["min_related_laws"]:
            quality_score += self.quality_weights["related_laws"]
        else:
            issues.append(f"ê´€??ë²•ë¥  ë¶€ì¡?(?„ì¬: {len(related_laws)}, ìµœì†Œ: {self.quality_criteria['min_related_laws']})")
        
        # 4. ?ë? ?¤ì›Œ??ê²€ì¦?
        precedent_keywords = term_info.get("precedent_keywords", [])
        if len(precedent_keywords) >= self.quality_criteria["min_precedent_keywords"]:
            quality_score += self.quality_weights["precedent_keywords"]
        else:
            issues.append(f"?ë? ?¤ì›Œ??ë¶€ì¡?(?„ì¬: {len(precedent_keywords)}, ìµœì†Œ: {self.quality_criteria['min_precedent_keywords']})")
        
        # 5. ? ë¢°??ê²€ì¦?
        confidence = term_info.get("confidence", 0.0)
        if confidence >= self.quality_criteria["min_confidence"]:
            quality_score += self.quality_weights["confidence"]
        else:
            issues.append(f"? ë¢°??ë¶€ì¡?(?„ì¬: {confidence:.2f}, ìµœì†Œ: {self.quality_criteria['min_confidence']})")
        
        # 6. ë¹ˆë„ ê²€ì¦?
        frequency = term_info.get("frequency", 0)
        if frequency < self.quality_criteria["min_frequency"]:
            issues.append(f"ë¹ˆë„ ë¶€ì¡?(?„ì¬: {frequency}, ìµœì†Œ: {self.quality_criteria['min_frequency']})")
        
        return quality_score, issues
    
    def validate_dictionary_quality(self, dictionary: Dict) -> Dict:
        """?„ì²´ ?¬ì „ ?ˆì§ˆ ê²€ì¦?""
        self.logger.info("Starting dictionary quality validation")
        
        validation_summary = {
            "total_terms": len(dictionary),
            "high_quality_terms": 0,
            "medium_quality_terms": 0,
            "low_quality_terms": 0,
            "rejected_terms": 0,
            "quality_distribution": defaultdict(int),
            "common_issues": Counter(),
            "domain_quality": defaultdict(dict)
        }
        
        for term, term_info in dictionary.items():
            # ê°œë³„ ?©ì–´ ê²€ì¦?
            quality_score, issues = self.validate_term_quality(term, term_info)
            
            # ê²€ì¦?ê²°ê³¼ ?€??
            self.validation_results[term] = {
                "quality_score": quality_score,
                "issues": issues,
                "term_info": term_info
            }
            
            # ?ˆì§ˆ ?±ê¸‰ ë¶„ë¥˜
            if quality_score >= 0.8:
                validation_summary["high_quality_terms"] += 1
                quality_grade = "high"
            elif quality_score >= 0.6:
                validation_summary["medium_quality_terms"] += 1
                quality_grade = "medium"
            elif quality_score >= 0.4:
                validation_summary["low_quality_terms"] += 1
                quality_grade = "low"
            else:
                validation_summary["rejected_terms"] += 1
                quality_grade = "rejected"
            
            validation_summary["quality_distribution"][quality_grade] += 1
            
            # ê³µí†µ ë¬¸ì œ???˜ì§‘
            for issue in issues:
                validation_summary["common_issues"][issue] += 1
            
            # ?„ë©”?¸ë³„ ?ˆì§ˆ ë¶„ì„
            domain = term_info.get("domain", "ê¸°í?")
            if domain not in validation_summary["domain_quality"]:
                validation_summary["domain_quality"][domain] = {
                    "total": 0, "high": 0, "medium": 0, "low": 0, "rejected": 0
                }
            
            validation_summary["domain_quality"][domain]["total"] += 1
            validation_summary["domain_quality"][domain][quality_grade] += 1
        
        return validation_summary
    
    def generate_improvement_suggestions(self, dictionary: Dict) -> Dict:
        """ê°œì„  ?œì•ˆ ?ì„±"""
        suggestions = {
            "overall_suggestions": [],
            "term_specific_suggestions": {},
            "domain_suggestions": defaultdict(list)
        }
        
        # ?„ì²´ ê°œì„  ?œì•ˆ
        total_terms = len(dictionary)
        high_quality_ratio = sum(1 for result in self.validation_results.values() 
                               if result["quality_score"] >= 0.8) / total_terms
        
        if high_quality_ratio < 0.7:
            suggestions["overall_suggestions"].append(
                f"?„ì²´ ?©ì–´??ê³ í’ˆì§?ë¹„ìœ¨????Šµ?ˆë‹¤ ({high_quality_ratio:.1%}). "
                "?™ì˜?´ì? ê´€???©ì–´ë¥???ë§ì´ ì¶”ê??˜ì„¸??"
            )
        
        # ?©ì–´ë³?ê°œì„  ?œì•ˆ
        for term, result in self.validation_results.items():
            term_suggestions = []
            
            if result["quality_score"] < 0.6:
                term_info = result["term_info"]
                
                if len(term_info.get("synonyms", [])) < 2:
                    term_suggestions.append("?™ì˜?´ë? ??ì¶”ê??˜ì„¸??)
                
                if len(term_info.get("related_terms", [])) < 3:
                    term_suggestions.append("ê´€???©ì–´ë¥???ì¶”ê??˜ì„¸??)
                
                if len(term_info.get("related_laws", [])) < 2:
                    term_suggestions.append("ê´€??ë²•ë¥ ????ì¶”ê??˜ì„¸??)
                
                if len(term_info.get("precedent_keywords", [])) < 2:
                    term_suggestions.append("?ë? ?¤ì›Œ?œë? ??ì¶”ê??˜ì„¸??)
            
            if term_suggestions:
                suggestions["term_specific_suggestions"][term] = term_suggestions
        
        # ?„ë©”?¸ë³„ ê°œì„  ?œì•ˆ
        domain_quality = defaultdict(list)
        for term, result in self.validation_results.items():
            domain = result["term_info"].get("domain", "ê¸°í?")
            domain_quality[domain].append(result["quality_score"])
        
        for domain, scores in domain_quality.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.7:
                suggestions["domain_suggestions"][domain].append(
                    f"{domain} ?„ë©”?¸ì˜ ?‰ê·  ?ˆì§ˆ????Šµ?ˆë‹¤ ({avg_score:.2f}). "
                    "?´ë‹¹ ?„ë©”?¸ì˜ ?©ì–´ ?•ë³´ë¥?ë³´ê°•?˜ì„¸??"
                )
        
        return suggestions
    
    def filter_high_quality_terms(self, dictionary: Dict, min_quality: float = 0.6) -> Dict:
        """ê³ í’ˆì§??©ì–´ë§??„í„°ë§?""
        filtered_dict = {}
        
        for term, term_info in dictionary.items():
            if term in self.validation_results:
                quality_score = self.validation_results[term]["quality_score"]
                if quality_score >= min_quality:
                    filtered_dict[term] = term_info
        
        self.logger.info(f"Filtered {len(filtered_dict)} high-quality terms from {len(dictionary)} total terms")
        return filtered_dict
    
    def generate_quality_report(self, validation_summary: Dict, suggestions: Dict) -> str:
        """?ˆì§ˆ ë³´ê³ ???ì„±"""
        report = []
        report.append("=== ë²•ë¥  ?©ì–´ ?¬ì „ ?ˆì§ˆ ê²€ì¦?ë³´ê³ ??===\n")
        
        # ?„ì²´ ?µê³„
        report.append("1. ?„ì²´ ?µê³„")
        report.append(f"   ì´??©ì–´ ?? {validation_summary['total_terms']}")
        report.append(f"   ê³ í’ˆì§??©ì–´: {validation_summary['high_quality_terms']} ({validation_summary['high_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   ì¤‘í’ˆì§??©ì–´: {validation_summary['medium_quality_terms']} ({validation_summary['medium_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   ?€?ˆì§ˆ ?©ì–´: {validation_summary['low_quality_terms']} ({validation_summary['low_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   ?œì™¸???©ì–´: {validation_summary['rejected_terms']} ({validation_summary['rejected_terms']/validation_summary['total_terms']:.1%})")
        report.append("")
        
        # ?„ë©”?¸ë³„ ?ˆì§ˆ
        report.append("2. ?„ë©”?¸ë³„ ?ˆì§ˆ")
        for domain, stats in validation_summary["domain_quality"].items():
            if stats["total"] > 0:
                high_ratio = stats["high"] / stats["total"]
                report.append(f"   {domain}: {stats['total']}ê°?(ê³ í’ˆì§?{high_ratio:.1%})")
        report.append("")
        
        # ê³µí†µ ë¬¸ì œ??
        report.append("3. ê³µí†µ ë¬¸ì œ??(?ìœ„ 5ê°?")
        for issue, count in validation_summary["common_issues"].most_common(5):
            report.append(f"   {issue}: {count}ê°?)
        report.append("")
        
        # ê°œì„  ?œì•ˆ
        report.append("4. ê°œì„  ?œì•ˆ")
        for suggestion in suggestions["overall_suggestions"]:
            report.append(f"   - {suggestion}")
        report.append("")
        
        return "\n".join(report)
    
    def save_validation_results(self, output_path: str, validation_summary: Dict, suggestions: Dict):
        """ê²€ì¦?ê²°ê³¼ ?€??""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "validation_summary": validation_summary,
                "suggestions": suggestions,
                "detailed_results": dict(self.validation_results)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Validation results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦ê¸°')
    parser.add_argument('--input_file', type=str, required=True,
                       help='ê²€ì¦í•  ?¬ì „ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--output_file', type=str,
                       default='data/extracted_terms/quality_validation_results.json',
                       help='ê²€ì¦?ê²°ê³¼ ì¶œë ¥ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--min_quality', type=float, default=0.6,
                       help='ìµœì†Œ ?ˆì§ˆ ê¸°ì? (ê¸°ë³¸ê°? 0.6)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ë¡œê·¸ ?ˆë²¨')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?ˆì§ˆ ê²€ì¦??¤í–‰
    validator = QualityValidator()
    
    # ?…ë ¥ ?Œì¼ ë¡œë“œ
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # ?ˆì§ˆ ê²€ì¦?
    validation_summary = validator.validate_dictionary_quality(dictionary)
    
    # ê°œì„  ?œì•ˆ ?ì„±
    suggestions = validator.generate_improvement_suggestions(dictionary)
    
    # ê³ í’ˆì§??©ì–´ ?„í„°ë§?
    high_quality_dict = validator.filter_high_quality_terms(dictionary, args.min_quality)
    
    # ê²°ê³¼ ?€??
    validator.save_validation_results(args.output_file, validation_summary, suggestions)
    
    # ?ˆì§ˆ ë³´ê³ ???ì„± ë°?ì¶œë ¥
    report = validator.generate_quality_report(validation_summary, suggestions)
    print(report)
    
    # ê³ í’ˆì§??¬ì „ ?€??
    high_quality_output = args.output_file.replace('.json', '_high_quality.json')
    with open(high_quality_output, 'w', encoding='utf-8') as f:
        json.dump(high_quality_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nê³ í’ˆì§??©ì–´ {len(high_quality_dict)}ê°œê? {high_quality_output}???€?¥ë˜?ˆìŠµ?ˆë‹¤.")


if __name__ == "__main__":
    main()
