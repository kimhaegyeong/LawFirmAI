#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?°ì´???ˆì§ˆ ê²€ì¦??¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸???˜ì§‘??ë²•ë¥  ?°ì´?°ì˜ ?ˆì§ˆ??ê²€ì¦í•˜ê³?ë³´ê³ ?œë? ?ì„±?©ë‹ˆ??
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.data_processor import DataProcessor

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """?°ì´???ˆì§ˆ ê²€ì¦??´ë˜??""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.validation_results = {
            "laws": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "precedents": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "legal_terms": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "administrative_rules": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "local_ordinances": {"total": 0, "valid": 0, "invalid": 0, "errors": []}
        }
    
    def validate_law_data(self, law_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """ë²•ë ¹ ?°ì´??ê²€ì¦?""
        errors = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ["law_id", "law_name", "effective_date", "content"]
        for field in required_fields:
            if not law_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # ë²•ë ¹ëª??•ì‹ ê²€ì¦?
        law_name = law_data.get("law_name", "")
        if law_name and not re.match(r'^[ê°€-??+ë²?', law_name):
            if not re.match(r'^[ê°€-??+ë²•ë¥ $', law_name):
                errors.append(f"Invalid law name format: {law_name}")
        
        # ?œí–‰?¼ì ?•ì‹ ê²€ì¦?
        effective_date = law_data.get("effective_date", "")
        if effective_date and not re.match(r'^\d{4}\.\d{2}\.\d{2}$', effective_date):
            errors.append(f"Invalid effective date format: {effective_date}")
        
        # ?´ìš© ê¸¸ì´ ê²€ì¦?
        content = law_data.get("content", "")
        if content and len(content) < 100:
            errors.append("Content too short (minimum 100 characters)")
        
        return len(errors) == 0, errors
    
    def validate_precedent_data(self, precedent_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """?ë? ?°ì´??ê²€ì¦?""
        errors = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ["precedent_id", "case_name", "court", "decision_date"]
        for field in required_fields:
            if not precedent_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # ?¬ê±´ë²ˆí˜¸ ?•ì‹ ê²€ì¦?
        precedent_id = precedent_data.get("precedent_id", "")
        if precedent_id and not re.match(r'^\d{4}[ê°€-??\d{4}$', precedent_id):
            errors.append(f"Invalid precedent ID format: {precedent_id}")
        
        # ë²•ì›ëª?ê²€ì¦?
        court = precedent_data.get("court", "")
        valid_courts = ["?€ë²•ì›", "ê³ ë“±ë²•ì›", "ì§€ë°©ë²•??, "ê°€?•ë²•??, "?‰ì •ë²•ì›", "?¹í—ˆë²•ì›", "?˜ì›ì§€ë°©ë²•??, "?œìš¸ê³ ë“±ë²•ì›"]
        if court and not any(valid_court in court for valid_court in valid_courts):
            errors.append(f"Invalid court name: {court}")
        
        # ? ê³ ?¼ì ?•ì‹ ê²€ì¦?
        decision_date = precedent_data.get("decision_date", "")
        if decision_date and not re.match(r'^\d{4}\.\d{2}\.\d{2}$', decision_date):
            errors.append(f"Invalid decision date format: {decision_date}")
        
        return len(errors) == 0, errors
    
    def validate_legal_term_data(self, term_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """ë²•ë ¹?©ì–´ ?°ì´??ê²€ì¦?""
        errors = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ["term_id", "term_name", "definition"]
        for field in required_fields:
            if not term_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # ?©ì–´ëª?ê¸¸ì´ ê²€ì¦?
        term_name = term_data.get("term_name", "")
        if term_name and len(term_name) < 2:
            errors.append("Term name too short (minimum 2 characters)")
        
        # ?•ì˜ ê¸¸ì´ ê²€ì¦?
        definition = term_data.get("definition", "")
        if definition and len(definition) < 10:
            errors.append("Definition too short (minimum 10 characters)")
        
        return len(errors) == 0, errors
    
    def validate_administrative_rule_data(self, rule_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """?‰ì •ê·œì¹™ ?°ì´??ê²€ì¦?""
        errors = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ["rule_id", "rule_name", "issuing_agency", "content"]
        for field in required_fields:
            if not rule_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # ë°œë ¹ê¸°ê? ê²€ì¦?
        issuing_agency = rule_data.get("issuing_agency", "")
        if issuing_agency and len(issuing_agency) < 3:
            errors.append("Issuing agency name too short")
        
        return len(errors) == 0, errors
    
    def validate_local_ordinance_data(self, ordinance_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """?ì¹˜ë²•ê·œ ?°ì´??ê²€ì¦?""
        errors = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ["ordinance_id", "ordinance_name", "issuing_authority", "content"]
        for field in required_fields:
            if not ordinance_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # ë°œë ¹ê¸°ê? ê²€ì¦?(ì§€?ì²´ëª??¬í•¨)
        issuing_authority = ordinance_data.get("issuing_authority", "")
        if issuing_authority and not any(keyword in issuing_authority for keyword in ["??, "??, "êµ?, "êµ?, "?¹ë³„??, "ê´‘ì—­??]):
            errors.append(f"Invalid issuing authority format: {issuing_authority}")
        
        return len(errors) == 0, errors
    
    def validate_data_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """?°ì´???Œì¼ ê²€ì¦?""
        logger.info(f"ê²€ì¦?ì¤? {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            validation_result = {
                "file_path": str(file_path),
                "data_type": data_type,
                "total_items": len(data),
                "valid_items": 0,
                "invalid_items": 0,
                "errors": []
            }
            
            for i, item in enumerate(data):
                is_valid, errors = self._validate_item(item, data_type)
                
                if is_valid:
                    validation_result["valid_items"] += 1
                else:
                    validation_result["invalid_items"] += 1
                    validation_result["errors"].extend([f"Item {i}: {error}" for error in errors])
            
            # ?„ì²´ ?µê³„ ?…ë°?´íŠ¸
            self.validation_results[data_type]["total"] += validation_result["total_items"]
            self.validation_results[data_type]["valid"] += validation_result["valid_items"]
            self.validation_results[data_type]["invalid"] += validation_result["invalid_items"]
            self.validation_results[data_type]["errors"].extend(validation_result["errors"])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"?Œì¼ ê²€ì¦?ì¤??¤ë¥˜ ë°œìƒ {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "data_type": data_type,
                "error": str(e),
                "total_items": 0,
                "valid_items": 0,
                "invalid_items": 0
            }
    
    def _validate_item(self, item: Dict[str, Any], data_type: str) -> Tuple[bool, List[str]]:
        """ê°œë³„ ?°ì´????ª© ê²€ì¦?""
        if data_type == "laws":
            return self.validate_law_data(item)
        elif data_type == "precedents":
            return self.validate_precedent_data(item)
        elif data_type == "legal_terms":
            return self.validate_legal_term_data(item)
        elif data_type == "administrative_rules":
            return self.validate_administrative_rule_data(item)
        elif data_type == "local_ordinances":
            return self.validate_local_ordinance_data(item)
        else:
            return False, [f"Unknown data type: {data_type}"]
    
    def validate_all_data(self) -> Dict[str, Any]:
        """ëª¨ë“  ?°ì´??ê²€ì¦?""
        logger.info("?„ì²´ ?°ì´???ˆì§ˆ ê²€ì¦??œì‘")
        
        data_directories = {
            "laws": Path("data/processed/laws"),
            "precedents": Path("data/processed/precedents"),
            "legal_terms": Path("data/processed/legal_terms"),
            "administrative_rules": Path("data/processed/administrative_rules"),
            "local_ordinances": Path("data/processed/local_ordinances")
        }
        
        validation_results = {}
        
        for data_type, data_dir in data_directories.items():
            if not data_dir.exists():
                logger.warning(f"?°ì´???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {data_dir}")
                continue
            
            validation_results[data_type] = []
            
            # JSON ?Œì¼ ê²€??
            for json_file in data_dir.glob("*.json"):
                result = self.validate_data_file(json_file, data_type)
                validation_results[data_type].append(result)
        
        return validation_results
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """?ˆì§ˆ ê²€ì¦?ë³´ê³ ???ì„±"""
        report = {
            "validation_date": datetime.now().isoformat(),
            "overall_summary": {},
            "detailed_results": validation_results,
            "quality_metrics": {}
        }
        
        # ?„ì²´ ?”ì•½ ê³„ì‚°
        total_items = sum(self.validation_results[data_type]["total"] for data_type in self.validation_results)
        total_valid = sum(self.validation_results[data_type]["valid"] for data_type in self.validation_results)
        total_invalid = sum(self.validation_results[data_type]["invalid"] for data_type in self.validation_results)
        
        report["overall_summary"] = {
            "total_items": total_items,
            "valid_items": total_valid,
            "invalid_items": total_invalid,
            "quality_score": (total_valid / total_items * 100) if total_items > 0 else 0
        }
        
        # ?°ì´??? í˜•ë³??ˆì§ˆ ì§€??
        for data_type, stats in self.validation_results.items():
            if stats["total"] > 0:
                report["quality_metrics"][data_type] = {
                    "completeness": (stats["valid"] / stats["total"]) * 100,
                    "error_rate": (stats["invalid"] / stats["total"]) * 100,
                    "total_errors": len(stats["errors"])
                }
        
        # ë³´ê³ ???€??
        report_path = Path("docs/data_quality_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# ?°ì´???ˆì§ˆ ê²€ì¦?ë³´ê³ ??n\n")
            f.write(f"**ê²€ì¦??¼ì‹œ**: {report['validation_date']}\n\n")
            
            f.write(f"## ?„ì²´ ?”ì•½\n")
            f.write(f"- **ì´??°ì´????ª©**: {report['overall_summary']['total_items']:,}ê°?n")
            f.write(f"- **? íš¨????ª©**: {report['overall_summary']['valid_items']:,}ê°?n")
            f.write(f"- **ë¬´íš¨????ª©**: {report['overall_summary']['invalid_items']:,}ê°?n")
            f.write(f"- **?ˆì§ˆ ?ìˆ˜**: {report['overall_summary']['quality_score']:.1f}%\n\n")
            
            f.write(f"## ?°ì´??? í˜•ë³??ˆì§ˆ ì§€??n")
            for data_type, metrics in report["quality_metrics"].items():
                f.write(f"### {data_type}\n")
                f.write(f"- **?„ì„±??*: {metrics['completeness']:.1f}%\n")
                f.write(f"- **?¤ë¥˜??*: {metrics['error_rate']:.1f}%\n")
                f.write(f"- **ì´??¤ë¥˜ ??*: {metrics['total_errors']}ê°?n\n")
            
            f.write(f"## ?ì„¸ ?¤ë¥˜ ëª©ë¡\n")
            for data_type, stats in self.validation_results.items():
                if stats["errors"]:
                    f.write(f"### {data_type}\n")
                    for error in stats["errors"][:10]:  # ?ìœ„ 10ê°??¤ë¥˜ë§??œì‹œ
                        f.write(f"- {error}\n")
                    if len(stats["errors"]) > 10:
                        f.write(f"- ... ë°?{len(stats['errors']) - 10}ê°?ì¶”ê? ?¤ë¥˜\n")
                    f.write("\n")
        
        logger.info(f"?ˆì§ˆ ê²€ì¦?ë³´ê³ ???ì„±: {report_path}")
        return report


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("?°ì´???ˆì§ˆ ê²€ì¦??¤í¬ë¦½íŠ¸ ?œì‘")
    
    try:
        # ê²€ì¦ê¸° ì´ˆê¸°??
        validator = DataQualityValidator()
        
        # ?„ì²´ ?°ì´??ê²€ì¦?
        validation_results = validator.validate_all_data()
        
        # ?ˆì§ˆ ë³´ê³ ???ì„±
        quality_report = validator.generate_quality_report(validation_results)
        
        logger.info("?°ì´???ˆì§ˆ ê²€ì¦??„ë£Œ")
        logger.info(f"?„ì²´ ?ˆì§ˆ ?ìˆ˜: {quality_report['overall_summary']['quality_score']:.1f}%")
        logger.info(f"? íš¨????ª©: {quality_report['overall_summary']['valid_items']:,}ê°?)
        logger.info(f"ë¬´íš¨????ª©: {quality_report['overall_summary']['invalid_items']:,}ê°?)
        
    except Exception as e:
        logger.error(f"?°ì´???ˆì§ˆ ê²€ì¦?ì¤??¤ë¥˜ ë°œìƒ: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
