# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?¬ì „ êµ¬ì¶• ?µí•© ë¹Œë”
?„ì²´ ?Œì´?„ë¼?¸ì„ ?¤í–‰?˜ì—¬ ?¥ìƒ??ë²•ë¥  ?©ì–´ ?¬ì „??êµ¬ì¶•
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import sys
import os

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.legal_term_extraction.term_extractor import LegalTermExtractor
from scripts.data_processing.legal_term_extraction.domain_expander import DomainTermExpander
from scripts.data_processing.legal_term_extraction.quality_validator import QualityValidator
from scripts.data_processing.legal_term_extraction.dictionary_integrator import DictionaryIntegrator

logger = logging.getLogger(__name__)


class LegalTermDictionaryBuilder:
    """ë²•ë¥  ?©ì–´ ?¬ì „ êµ¬ì¶• ?µí•© ë¹Œë”"""
    
    def __init__(self):
        """ì´ˆê¸°??""
        self.logger = logging.getLogger(__name__)
        
        # ?Œì´?„ë¼??êµ¬ì„± ?”ì†Œ
        self.extractor = LegalTermExtractor()
        self.expander = DomainTermExpander()
        self.validator = QualityValidator()
        self.integrator = DictionaryIntegrator()
        
        # ê²°ê³¼ ?€??
        self.pipeline_results = {}
    
    def run_full_pipeline(self, 
                         data_directory: str,
                         existing_dictionary_path: str = "data/legal_term_dictionary.json",
                         output_directory: str = "data/extracted_terms",
                         min_frequency: int = 5,
                         quality_threshold: float = 0.6) -> Dict:
        """?„ì²´ ?Œì´?„ë¼???¤í–‰"""
        
        self.logger.info("Starting legal term dictionary building pipeline")
        
        # 1?¨ê³„: ?©ì–´ ì¶”ì¶œ
        self.logger.info("Step 1: Extracting terms from corpus")
        extraction_results = self._extract_terms(data_directory, min_frequency)
        
        # 2?¨ê³„: ?„ë©”?¸ë³„ ?•ì¥
        self.logger.info("Step 2: Expanding domain-specific terms")
        domain_results = self._expand_domain_terms(extraction_results)
        
        # 3?¨ê³„: ?ˆì§ˆ ê²€ì¦?
        self.logger.info("Step 3: Validating term quality")
        validation_results = self._validate_quality(domain_results)
        
        # 4?¨ê³„: ?¬ì „ ?µí•©
        self.logger.info("Step 4: Integrating dictionaries")
        integration_results = self._integrate_dictionaries(
            existing_dictionary_path, validation_results, quality_threshold
        )
        
        # ìµœì¢… ê²°ê³¼ ?•ë¦¬
        final_results = {
            "pipeline_summary": {
                "extraction": extraction_results,
                "domain_expansion": domain_results,
                "validation": validation_results,
                "integration": integration_results
            },
            "final_dictionary_path": integration_results["output_path"],
            "total_terms": integration_results["total_terms"],
            "quality_score": integration_results["quality_score"]
        }
        
        self.logger.info("Pipeline completed successfully")
        return final_results
    
    def _extract_terms(self, data_directory: str, min_frequency: int) -> Dict:
        """1?¨ê³„: ?©ì–´ ì¶”ì¶œ"""
        try:
            # ?©ì–´ ì¶”ì¶œ ?¤í–‰
            extraction_results = self.extractor.process_directory(data_directory, min_frequency)
            
            # ê²°ê³¼ ?€??
            output_path = "data/extracted_terms/raw_extracted_terms.json"
            self.extractor.save_results(output_path, extraction_results)
            
            self.pipeline_results["extraction"] = {
                "output_path": output_path,
                "processed_files": extraction_results["processed_files"],
                "total_terms": extraction_results["total_terms_extracted"],
                "filtered_terms": sum(len(terms) for terms in extraction_results["filtered_terms"].values())
            }
            
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error in term extraction: {e}")
            raise
    
    def _expand_domain_terms(self, extraction_results: Dict) -> Dict:
        """2?¨ê³„: ?„ë©”?¸ë³„ ?©ì–´ ?•ì¥"""
        try:
            # ì¶”ì¶œ???©ì–´ë¡??„ë©”???•ì¥
            domain_terms = self.expander.expand_domain_terms(extraction_results["filtered_terms"])
            
            # ?¥ìƒ???¬ì „ ?ì„±
            enhanced_dict = self.expander.generate_enhanced_dictionary(
                extraction_results["filtered_terms"], domain_terms
            )
            
            # ê²°ê³¼ ?€??
            output_path = "data/extracted_terms/domain_expanded_terms.json"
            self.expander.save_enhanced_dictionary(enhanced_dict, output_path)
            
            self.pipeline_results["domain_expansion"] = {
                "output_path": output_path,
                "total_terms": len(enhanced_dict),
                "domains": list(domain_terms.keys())
            }
            
            return enhanced_dict
            
        except Exception as e:
            self.logger.error(f"Error in domain expansion: {e}")
            raise
    
    def _validate_quality(self, enhanced_dict: Dict) -> Dict:
        """3?¨ê³„: ?ˆì§ˆ ê²€ì¦?""
        try:
            # ?ˆì§ˆ ê²€ì¦??¤í–‰
            validation_summary = self.validator.validate_dictionary_quality(enhanced_dict)
            
            # ê°œì„  ?œì•ˆ ?ì„±
            suggestions = self.validator.generate_improvement_suggestions(enhanced_dict)
            
            # ê³ í’ˆì§??©ì–´ ?„í„°ë§?
            high_quality_dict = self.validator.filter_high_quality_terms(enhanced_dict)
            
            # ê²°ê³¼ ?€??
            output_path = "data/extracted_terms/quality_validated_terms.json"
            self.validator.save_validation_results(output_path, validation_summary, suggestions)
            
            # ê³ í’ˆì§??¬ì „ ?€??
            high_quality_path = "data/extracted_terms/high_quality_terms.json"
            with open(high_quality_path, 'w', encoding='utf-8') as f:
                json.dump(high_quality_dict, f, ensure_ascii=False, indent=2)
            
            self.pipeline_results["validation"] = {
                "output_path": output_path,
                "high_quality_path": high_quality_path,
                "total_terms": len(enhanced_dict),
                "high_quality_terms": len(high_quality_dict),
                "quality_ratio": len(high_quality_dict) / len(enhanced_dict) if enhanced_dict else 0
            }
            
            return high_quality_dict
            
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            raise
    
    def _integrate_dictionaries(self, 
                              existing_dict_path: str, 
                              validated_dict: Dict, 
                              quality_threshold: float) -> Dict:
        """4?¨ê³„: ?¬ì „ ?µí•©"""
        try:
            # ê¸°ì¡´ ?¬ì „ ë¡œë“œ
            existing_dict = self.integrator.load_existing_dictionary(existing_dict_path)
            
            # ?¬ì „ ?µí•©
            merged_dict, integration_stats = self.integrator.merge_dictionaries(
                existing_dict, validated_dict
            )
            
            # ?µí•©???¬ì „ ê²€ì¦?
            validation_results = self.integrator.validate_integrated_dictionary(merged_dict)
            
            # ìµœì¢… ?¬ì „ ?€??
            output_path = "data/enhanced_legal_term_dictionary.json"
            self.integrator.save_integrated_dictionary(merged_dict, output_path)
            
            # ?µí•© ë³´ê³ ???ì„±
            report = self.integrator.generate_integration_report(integration_stats, validation_results)
            
            # ë³´ê³ ???€??
            report_path = "data/extracted_terms/integration_report.txt"
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.pipeline_results["integration"] = {
                "output_path": output_path,
                "report_path": report_path,
                "total_terms": len(merged_dict),
                "new_terms": integration_stats["new_terms"],
                "updated_terms": integration_stats["updated_terms"],
                "quality_score": validation_results["high_confidence_terms"] / len(merged_dict) if merged_dict else 0
            }
            
            return {
                "output_path": output_path,
                "total_terms": len(merged_dict),
                "quality_score": validation_results["high_confidence_terms"] / len(merged_dict) if merged_dict else 0,
                "integration_stats": integration_stats,
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in dictionary integration: {e}")
            raise
    
    def generate_pipeline_report(self, results: Dict) -> str:
        """?Œì´?„ë¼???¤í–‰ ë³´ê³ ???ì„±"""
        report = []
        report.append("=== ë²•ë¥  ?©ì–´ ?¬ì „ êµ¬ì¶• ?Œì´?„ë¼???¤í–‰ ë³´ê³ ??===\n")
        
        # ?„ì²´ ?”ì•½
        report.append("1. ?„ì²´ ?”ì•½")
        report.append(f"   ìµœì¢… ?¬ì „ ê²½ë¡œ: {results['final_dictionary_path']}")
        report.append(f"   ì´??©ì–´ ?? {results['total_terms']}")
        report.append(f"   ?ˆì§ˆ ?ìˆ˜: {results['quality_score']:.2f}")
        report.append("")
        
        # ?¨ê³„ë³?ê²°ê³¼
        pipeline_summary = results["pipeline_summary"]
        
        report.append("2. 1?¨ê³„: ?©ì–´ ì¶”ì¶œ")
        extraction = pipeline_summary["extraction"]
        report.append(f"   ì²˜ë¦¬???Œì¼: {extraction['processed_files']}")
        report.append(f"   ì¶”ì¶œ???©ì–´: {extraction['total_terms']}")
        report.append(f"   ?„í„°ë§????©ì–´: {extraction['filtered_terms']}")
        report.append("")
        
        report.append("3. 2?¨ê³„: ?„ë©”?¸ë³„ ?•ì¥")
        domain_expansion = pipeline_summary["domain_expansion"]
        report.append(f"   ?•ì¥???©ì–´: {domain_expansion['total_terms']}")
        report.append(f"   ?„ë©”???? {len(domain_expansion['domains'])}")
        report.append("")
        
        report.append("4. 3?¨ê³„: ?ˆì§ˆ ê²€ì¦?)
        validation = pipeline_summary["validation"]
        report.append(f"   ê²€ì¦ëœ ?©ì–´: {validation['total_terms']}")
        report.append(f"   ê³ í’ˆì§??©ì–´: {validation['high_quality_terms']}")
        report.append(f"   ?ˆì§ˆ ë¹„ìœ¨: {validation['quality_ratio']:.1%}")
        report.append("")
        
        report.append("5. 4?¨ê³„: ?¬ì „ ?µí•©")
        integration = pipeline_summary["integration"]
        report.append(f"   ?µí•©???©ì–´: {integration['total_terms']}")
        report.append(f"   ?ˆë¡œ ì¶”ê????©ì–´: {integration['new_terms']}")
        report.append(f"   ?…ë°?´íŠ¸???©ì–´: {integration['updated_terms']}")
        report.append(f"   ìµœì¢… ?ˆì§ˆ ?ìˆ˜: {integration['quality_score']:.2f}")
        
        return "\n".join(report)
    
    def save_pipeline_results(self, results: Dict, output_path: str):
        """?Œì´?„ë¼??ê²°ê³¼ ?€??""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Pipeline results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline results: {e}")


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description='ë²•ë¥  ?©ì–´ ?¬ì „ êµ¬ì¶• ?µí•© ë¹Œë”')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='ë²•ë¥  ?°ì´???”ë ‰? ë¦¬ ê²½ë¡œ')
    parser.add_argument('--existing_dict', type=str,
                       default='data/legal_term_dictionary.json',
                       help='ê¸°ì¡´ ?¬ì „ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--output_dir', type=str,
                       default='data/extracted_terms',
                       help='ì¶œë ¥ ?”ë ‰? ë¦¬ ê²½ë¡œ')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='ìµœì†Œ ë¹ˆë„ (ê¸°ë³¸ê°? 5)')
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
    
    # ?Œì´?„ë¼???¤í–‰
    builder = LegalTermDictionaryBuilder()
    
    try:
        results = builder.run_full_pipeline(
            data_directory=args.data_dir,
            existing_dictionary_path=args.existing_dict,
            output_directory=args.output_dir,
            min_frequency=args.min_frequency,
            quality_threshold=args.quality_threshold
        )
        
        # ?Œì´?„ë¼??ë³´ê³ ???ì„± ë°?ì¶œë ¥
        report = builder.generate_pipeline_report(results)
        print(report)
        
        # ê²°ê³¼ ?€??
        builder.save_pipeline_results(results, f"{args.output_dir}/pipeline_results.json")
        
        print(f"\n?Œì´?„ë¼?¸ì´ ?±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
        print(f"ìµœì¢… ?¬ì „: {results['final_dictionary_path']}")
        print(f"ì´??©ì–´ ?? {results['total_terms']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
