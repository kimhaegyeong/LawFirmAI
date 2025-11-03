#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦??¤í¬ë¦½íŠ¸

?„ì²˜ë¦¬ëœ ?°ì´?°ì˜ ?ˆì§ˆ??ê²€ì¦í•˜ê³?ë³´ê³ ?œë? ?ì„±?©ë‹ˆ??
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

from source.data.data_processor import LegalDataProcessor

class ProcessedDataValidator:
    """?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦??´ë˜??""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.logger = logging.getLogger(__name__)
        
        # ?ˆì§ˆ ì§€???¤ì •
        self.quality_metrics = {
            "completeness": 0.95,      # ?„ì„±??95% ?´ìƒ
            "accuracy": 0.98,          # ?•í™•??98% ?´ìƒ
            "consistency": 0.90,       # ?¼ê???90% ?´ìƒ
            "term_normalization": 0.90 # ?©ì–´ ?•ê·œ??90% ?´ìƒ
        }
    
    def validate_all_data(self, processed_dir: str = "data/processed") -> Dict[str, Any]:
        """ëª¨ë“  ?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦?""
        self.logger.info("?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦??œì‘")
        
        validation_results = {
            "overall": {
                "total_documents": 0,
                "valid_documents": 0,
                "invalid_documents": 0,
                "validation_passed": False,
                "quality_score": 0.0
            },
            "by_type": {},
            "issues": [],
            "recommendations": []
        }
        
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            self.logger.error(f"?„ì²˜ë¦¬ëœ ?°ì´???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŒ: {processed_dir}")
            return validation_results
        
        # ?°ì´??? í˜•ë³?ê²€ì¦?
        data_types = ["laws", "precedents", "constitutional_decisions", "legal_interpretations", "legal_terms"]
        
        for data_type in data_types:
            type_result = self.validate_data_type(data_type, processed_path)
            validation_results["by_type"][data_type] = type_result
            validation_results["overall"]["total_documents"] += type_result["total_documents"]
            validation_results["overall"]["valid_documents"] += type_result["valid_documents"]
            validation_results["overall"]["invalid_documents"] += type_result["invalid_documents"]
        
        # ?„ì²´ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
        if validation_results["overall"]["total_documents"] > 0:
            # ? íš¨??ê¸°ë°˜ ?ˆì§ˆ ?ìˆ˜
            validity_score = validation_results["overall"]["valid_documents"] / validation_results["overall"]["total_documents"]
            
            # ?°ì´??? í˜•ë³??ˆì§ˆ ì§€???‰ê· 
            quality_scores = []
            for data_type, type_result in validation_results["by_type"].items():
                if type_result.get("quality_score"):
                    quality_scores.append(type_result["quality_score"])
            
            # ?„ì²´ ?ˆì§ˆ ?ìˆ˜ (? íš¨??70% + ?ˆì§ˆì§€??30%)
            if quality_scores:
                avg_quality_score = sum(quality_scores) / len(quality_scores)
                validation_results["overall"]["quality_score"] = validity_score * 0.7 + avg_quality_score * 0.3
            else:
                validation_results["overall"]["quality_score"] = validity_score
            
            validation_results["overall"]["validation_passed"] = (
                validation_results["overall"]["quality_score"] >= 0.8
            )
        
        # ê¶Œì¥?¬í•­ ?ì„±
        validation_results["recommendations"] = self.generate_recommendations(validation_results)
        
        # ê²€ì¦?ê²°ê³¼ ?€??
        self.save_validation_report(validation_results)
        
        return validation_results
    
    def validate_data_type(self, data_type: str, processed_path: Path) -> Dict[str, Any]:
        """?¹ì • ?°ì´??? í˜• ê²€ì¦?""
        self.logger.info(f"{data_type} ?°ì´??ê²€ì¦??œì‘")
        
        data_dir = processed_path / data_type
        if not data_dir.exists():
            return {
                "total_documents": 0,
                "valid_documents": 0,
                "invalid_documents": 0,
                "validation_passed": False,
                "issues": [f"{data_type} ?”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŒ"],
                "quality_metrics": {}
            }
        
        json_files = list(data_dir.glob("*.json"))
        total_documents = 0
        valid_documents = 0
        invalid_documents = 0
        issues = []
        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "term_normalization": 0.0
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
                
                for doc in documents:
                    total_documents += 1
                    
                    # ë¬¸ì„œ ? íš¨??ê²€??
                    is_valid, doc_issues = self.processor.validate_document(doc)
                    
                    if is_valid:
                        valid_documents += 1
                    else:
                        invalid_documents += 1
                        issues.extend(doc_issues)
                    
                    # ?ˆì§ˆ ì§€??ê³„ì‚°
                    doc_metrics = self.calculate_quality_metrics(doc)
                    for metric, value in doc_metrics.items():
                        if metric in quality_metrics:
                            quality_metrics[metric] += value
                
            except Exception as e:
                self.logger.error(f"ê²€ì¦?ì¤??¤ë¥˜ {json_file}: {e}")
                issues.append(f"?Œì¼ ?½ê¸° ?¤ë¥˜: {json_file}")
        
        # ?‰ê·  ?ˆì§ˆ ì§€??ê³„ì‚°
        if total_documents > 0:
            for metric in quality_metrics:
                quality_metrics[metric] = quality_metrics[metric] / total_documents
        
        # ?„ì²´ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
        quality_score = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.0
        
        validation_passed = (
            total_documents > 0 and 
            (valid_documents / total_documents) >= 0.9
        )
        
        return {
            "total_documents": total_documents,
            "valid_documents": valid_documents,
            "invalid_documents": invalid_documents,
            "validation_passed": validation_passed,
            "issues": issues,
            "quality_metrics": quality_metrics,
            "quality_score": quality_score
        }
    
    def calculate_quality_metrics(self, document: Dict[str, Any]) -> Dict[str, float]:
        """ë¬¸ì„œ???ˆì§ˆ ì§€??ê³„ì‚°"""
        metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "term_normalization": 0.0
        }
        
        # ?„ì„±??ê³„ì‚° - ë²•ë ¹ ?°ì´?°ì— ë§ëŠ” ?„ë“œ??
        required_fields = ["id", "law_name", "articles"]
        present_fields = sum(1 for field in required_fields if field in document and document[field])
        
        # ì¶”ê? ?„ë“œ??(?ˆìœ¼ë©?ë³´ë„ˆ??
        bonus_fields = ["chunks", "article_chunks", "cleaned_content", "entities"]
        bonus_count = sum(1 for field in bonus_fields if field in document and document[field])
        
        # ?„ì„±??= ?„ìˆ˜ ?„ë“œ (70%) + ë³´ë„ˆ???„ë“œ (30%)
        metrics["completeness"] = (present_fields / len(required_fields)) * 0.7 + (bonus_count / len(bonus_fields)) * 0.3
        
        # ?¼ê???ê³„ì‚° - ì¡°ë¬¸ê³?ì²?¬???¼ê???
        if "articles" in document and document["articles"]:
            articles_count = len(document["articles"])
            chunks_count = len(document.get("chunks", []))
            article_chunks_count = len(document.get("article_chunks", []))
            
            # ì¡°ë¬¸???ˆê³  ì²?¬ê°€ ?ì„±?˜ì—ˆ?¼ë©´ ?¼ê????’ìŒ
            if articles_count > 0 and (chunks_count > 0 or article_chunks_count > 0):
                metrics["consistency"] = 1.0
            else:
                metrics["consistency"] = 0.5
        else:
            # ì¡°ë¬¸???†ìœ¼ë©?ê¸°ë³¸ ?¼ê???
            metrics["consistency"] = 0.3
        
        # ?©ì–´ ?•ê·œ??ê³„ì‚° - ë²•ë ¹ ?°ì´???¹ì„± ê³ ë ¤
        if "entities" in document and document["entities"]:
            # ?”í‹°?°ê? ?ˆìœ¼ë©??•ê·œ??ì§„í–‰??ê²ƒìœ¼ë¡?ê°„ì£¼
            metrics["term_normalization"] = 0.8
        elif "cleaned_content" in document and document["cleaned_content"]:
            # ?•ë¦¬???´ìš©???ˆìœ¼ë©?ê¸°ë³¸ ?•ê·œ??
            metrics["term_normalization"] = 0.6
        else:
            # ê¸°ë³¸ê°?
            metrics["term_normalization"] = 0.4
        
        return metrics
    
    def generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """ê²€ì¦?ê²°ê³¼ ê¸°ë°˜ ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []
        
        overall = validation_results["overall"]
        
        if overall["quality_score"] < 0.9:
            recommendations.append("?„ì²´ ?°ì´???ˆì§ˆ??ëª©í‘œì¹?90%) ë¯¸ë§Œ?…ë‹ˆ?? ?°ì´???„ì²˜ë¦?ê³¼ì •???¬ê?? í•˜?¸ìš”.")
        
        if overall["invalid_documents"] > 0:
            recommendations.append(f"{overall['invalid_documents']}ê°œì˜ ? íš¨?˜ì? ?Šì? ë¬¸ì„œê°€ ?ˆìŠµ?ˆë‹¤. ?´ë‹¹ ë¬¸ì„œ?¤ì„ ?˜ì •?˜ì„¸??")
        
        for data_type, type_result in validation_results["by_type"].items():
            if not type_result["validation_passed"]:
                recommendations.append(f"{data_type} ?°ì´?°ì˜ ?ˆì§ˆ??ê¸°ì???ë¯¸ë‹¬?©ë‹ˆ?? ì¶”ê? ?„ì²˜ë¦¬ê? ?„ìš”?©ë‹ˆ??")
            
            # quality_metricsê°€ ì¡´ì¬?˜ëŠ” ê²½ìš°?ë§Œ ì²´í¬
            if "quality_metrics" in type_result:
                if type_result["quality_metrics"].get("completeness", 1.0) < 0.95:
                    recommendations.append(f"{data_type} ?°ì´?°ì˜ ?„ì„±?„ê? ??Šµ?ˆë‹¤. ?„ìˆ˜ ?„ë“œê°€ ?„ë½?˜ì—ˆ?????ˆìŠµ?ˆë‹¤.")
                
                if type_result["quality_metrics"].get("term_normalization", 1.0) < 0.9:
                    recommendations.append(f"{data_type} ?°ì´?°ì˜ ?©ì–´ ?•ê·œ?”ê? ë¶€ì¡±í•©?ˆë‹¤. ë²•ë¥  ?©ì–´ ?¬ì „???…ë°?´íŠ¸?˜ì„¸??")
        
        return recommendations
    
    def save_validation_report(self, validation_results: Dict[str, Any]):
        """ê²€ì¦?ë³´ê³ ???€??""
        report_file = Path("data/processed") / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"ê²€ì¦?ë³´ê³ ???€???„ë£Œ: {report_file}")
    
    def print_validation_summary(self, validation_results: Dict[str, Any]):
        """ê²€ì¦?ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
        overall = validation_results["overall"]
        
        print("\n=== ?°ì´??ê²€ì¦?ê²°ê³¼ ===")
        print(f"ì´?ë¬¸ì„œ ?? {overall['total_documents']}")
        print(f"? íš¨??ë¬¸ì„œ: {overall['valid_documents']}")
        print(f"? íš¨?˜ì? ?Šì? ë¬¸ì„œ: {overall['invalid_documents']}")
        print(f"?ˆì§ˆ ?ìˆ˜: {overall.get('quality_score', 0.0):.2%}")
        print(f"ê²€ì¦??µê³¼: {'?? if overall['validation_passed'] else '?„ë‹ˆ??}")
        
        print("\n=== ?°ì´??? í˜•ë³?ê²°ê³¼ ===")
        for data_type, type_result in validation_results["by_type"].items():
            print(f"{data_type}:")
            print(f"  - ì´?ë¬¸ì„œ: {type_result['total_documents']}")
            print(f"  - ? íš¨??ë¬¸ì„œ: {type_result['valid_documents']}")
            print(f"  - ê²€ì¦??µê³¼: {'?? if type_result['validation_passed'] else '?„ë‹ˆ??}")
            if type_result['issues']:
                print(f"  - ?´ìŠˆ ?? {len(type_result['issues'])}")
        
        if validation_results["recommendations"]:
            print("\n=== ê¶Œì¥?¬í•­ ===")
            for i, rec in enumerate(validation_results["recommendations"], 1):
                print(f"{i}. {rec}")

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description="?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦?)
    parser.add_argument("--processed-dir", default="data/processed",
                       help="?„ì²˜ë¦¬ëœ ?°ì´???”ë ‰? ë¦¬")
    parser.add_argument("--data-type", 
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       default="all",
                       help="ê²€ì¦í•  ?°ì´??? í˜•")
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = ProcessedDataValidator()
    
    if args.data_type == "all":
        validation_results = validator.validate_all_data(args.processed_dir)
    else:
        # ?¹ì • ?°ì´??? í˜•ë§?ê²€ì¦?
        processed_path = Path(args.processed_dir)
        type_result = validator.validate_data_type(args.data_type, processed_path)
        validation_results = {
            "overall": type_result,
            "by_type": {args.data_type: type_result},
            "issues": type_result.get("issues", []),
            "recommendations": []
        }
    
    validator.print_validation_summary(validation_results)

if __name__ == "__main__":
    main()
