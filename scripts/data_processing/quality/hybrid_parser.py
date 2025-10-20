#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Article Parser

This module provides a hybrid parsing approach that combines rule-based and ML parsing,
selects the best result based on quality metrics, and applies post-processing corrections.

Usage:
    parser = HybridArticleParser()
    result = parser.parse_law_content(law_content)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parsers to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'utilities' / 'parsers'))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'ml_training' / 'model_training'))

try:
    from parsers.improved_article_parser import ImprovedArticleParser
    from ml_enhanced_parser import MLEnhancedArticleParser
    from data_quality_validator import DataQualityValidator, QualityReport
    PARSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some parsers not available: {e}")
    PARSERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridArticleParser:
    """Hybrid parser that combines rule-based and ML parsing approaches"""
    
    def __init__(self, ml_model_path: str = "models/article_classifier.pkl"):
        """
        Initialize hybrid parser
        
        Args:
            ml_model_path: Path to ML model file
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsers
        self.rule_based_parser = None
        self.ml_parser = None
        self.quality_validator = None
        
        self.parsers_available = PARSERS_AVAILABLE
        
        if self.parsers_available:
            try:
                self.rule_based_parser = ImprovedArticleParser()
                self.ml_parser = MLEnhancedArticleParser(ml_model_path)
                self.quality_validator = DataQualityValidator()
                self.logger.info("Hybrid parser initialized with all components")
            except Exception as e:
                self.logger.error(f"Error initializing parsers: {e}")
                self.parsers_available = False
        
        if not self.parsers_available:
            self.logger.warning("Falling back to basic parsing only")
    
    def parse_law_content(self, law_content: str, law_name: str = "Unknown") -> Dict[str, Any]:
        """
        Parse law content using hybrid approach
        
        Args:
            law_content: Law content to parse
            law_name: Name of the law
            
        Returns:
            Dict[str, Any]: Parsed result with quality information
        """
        try:
            self.logger.info(f"Starting hybrid parsing for law: {law_name}")
            
            # Initialize result structure
            result = {
                'law_name': law_name,
                'parsing_method': 'hybrid',
                'parsing_timestamp': datetime.now().isoformat(),
                'rule_based_result': None,
                'ml_result': None,
                'final_result': None,
                'quality_comparison': None,
                'auto_corrected': False,
                'manual_review_required': False
            }
            
            if not self.parsers_available:
                return self._create_fallback_result(law_content, law_name)
            
            # Run both parsers
            rule_based_result = self._run_rule_based_parser(law_content)
            ml_result = self._run_ml_parser(law_content)
            
            result['rule_based_result'] = rule_based_result
            result['ml_result'] = ml_result
            
            # Compare results and select best
            final_result = self._select_best_result(rule_based_result, ml_result)
            result['final_result'] = final_result
            
            # Apply post-processing corrections
            corrected_result = self._apply_post_processing_corrections(final_result)
            result['final_result'] = corrected_result
            
            # Check if manual review is required
            if corrected_result.get('quality_score', 0) < 0.6:
                result['manual_review_required'] = True
                self.logger.warning(f"Low quality result for {law_name}, manual review recommended")
            
            self.logger.info(f"Hybrid parsing completed for {law_name}. Quality score: {corrected_result.get('quality_score', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in hybrid parsing: {e}")
            return self._create_error_result(law_content, law_name, str(e))
    
    def _run_rule_based_parser(self, law_content: str) -> Dict[str, Any]:
        """Run rule-based parser"""
        try:
            if not self.rule_based_parser:
                return {'error': 'Rule-based parser not available'}
            
            result = self.rule_based_parser.parse_law_document(law_content)
            result['parsing_method'] = 'rule_based'
            
            # Add quality validation
            if self.quality_validator:
                quality_report = self.quality_validator.validate_parsing_quality(result)
                if quality_report:
                    result['quality_score'] = quality_report.overall_score
                    result['quality_report'] = quality_report
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in rule-based parsing: {e}")
            return {'error': str(e), 'parsing_method': 'rule_based'}
    
    def _run_ml_parser(self, law_content: str) -> Dict[str, Any]:
        """Run ML-enhanced parser"""
        try:
            if not self.ml_parser:
                return {'error': 'ML parser not available'}
            
            # Use the new validation-enabled method
            if hasattr(self.ml_parser, 'parse_law_with_validation'):
                result = self.ml_parser.parse_law_with_validation(law_content)
            else:
                result = self.ml_parser.parse_law_document(law_content)
                # Add quality validation manually
                if self.quality_validator:
                    quality_report = self.quality_validator.validate_parsing_quality(result)
                    if quality_report:
                        result['quality_score'] = quality_report.overall_score
                        result['quality_report'] = quality_report
            
            result['parsing_method'] = 'ml_enhanced'
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ML parsing: {e}")
            return {'error': str(e), 'parsing_method': 'ml_enhanced'}
    
    def _select_best_result(self, rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best result based on quality metrics"""
        try:
            # Extract quality scores
            rule_score = rule_result.get('quality_score', 0)
            ml_score = ml_result.get('quality_score', 0)
            
            # Extract article counts
            rule_articles = len(rule_result.get('all_articles', []))
            ml_articles = len(ml_result.get('all_articles', []))
            
            # Selection criteria
            selection_criteria = {
                'rule_based': {
                    'quality_score': rule_score,
                    'article_count': rule_articles,
                    'has_error': 'error' in rule_result,
                    'method': 'rule_based'
                },
                'ml_enhanced': {
                    'quality_score': ml_score,
                    'article_count': ml_articles,
                    'has_error': 'error' in ml_result,
                    'method': 'ml_enhanced'
                }
            }
            
            # Select best result
            if selection_criteria['rule_based']['has_error'] and not selection_criteria['ml_enhanced']['has_error']:
                selected = ml_result
                selected['selected_method'] = 'ml_enhanced'
                selected['selection_reason'] = 'rule_based_failed'
            elif selection_criteria['ml_enhanced']['has_error'] and not selection_criteria['rule_based']['has_error']:
                selected = rule_result
                selected['selected_method'] = 'rule_based'
                selected['selection_reason'] = 'ml_enhanced_failed'
            elif rule_score > ml_score + 0.1:  # Rule-based significantly better
                selected = rule_result
                selected['selected_method'] = 'rule_based'
                selected['selection_reason'] = 'higher_quality_score'
            elif ml_score > rule_score + 0.1:  # ML significantly better
                selected = ml_result
                selected['selected_method'] = 'ml_enhanced'
                selected['selection_reason'] = 'higher_quality_score'
            elif ml_articles > rule_articles:  # ML found more articles
                selected = ml_result
                selected['selected_method'] = 'ml_enhanced'
                selected['selection_reason'] = 'more_articles'
            else:  # Default to rule-based
                selected = rule_result
                selected['selected_method'] = 'rule_based'
                selected['selection_reason'] = 'default'
            
            # Add comparison information
            selected['quality_comparison'] = {
                'rule_based_score': rule_score,
                'ml_enhanced_score': ml_score,
                'rule_based_articles': rule_articles,
                'ml_enhanced_articles': ml_articles,
                'selection_criteria': selection_criteria
            }
            
            self.logger.info(f"Selected {selected['selected_method']} result (reason: {selected['selection_reason']})")
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting best result: {e}")
            # Fallback to rule-based result
            return rule_result
    
    def _apply_post_processing_corrections(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing corrections to the selected result"""
        try:
            if 'error' in result:
                return result
            
            corrected_result = result.copy()
            corrections_applied = []
            
            # Apply corrections based on quality issues
            quality_report = result.get('quality_report')
            if quality_report and quality_report.overall_score < 0.8:
                
                # Fix missing articles if ML parser is available
                if self.ml_parser and hasattr(self.ml_parser, 'fix_missing_articles'):
                    original_count = len(corrected_result.get('all_articles', []))
                    corrected_result = self.ml_parser.fix_missing_articles(corrected_result)
                    new_count = len(corrected_result.get('all_articles', []))
                    if new_count > original_count:
                        corrections_applied.append(f"Recovered {new_count - original_count} missing articles")
                
                # Remove duplicates
                if self.ml_parser and hasattr(self.ml_parser, 'remove_duplicate_articles'):
                    original_count = len(corrected_result.get('all_articles', []))
                    corrected_result = self.ml_parser.remove_duplicate_articles(corrected_result)
                    duplicates_removed = corrected_result.get('duplicates_removed', 0)
                    if duplicates_removed > 0:
                        corrections_applied.append(f"Removed {duplicates_removed} duplicate articles")
                
                # Re-validate after corrections
                if self.quality_validator and corrections_applied:
                    updated_report = self.quality_validator.validate_parsing_quality(corrected_result)
                    if updated_report:
                        corrected_result['quality_report'] = updated_report
                        corrected_result['quality_score'] = updated_report.overall_score
                        corrected_result['auto_corrected'] = True
                        corrected_result['corrections_applied'] = corrections_applied
                        
                        self.logger.info(f"Applied {len(corrections_applied)} corrections. New quality score: {updated_report.overall_score:.3f}")
            
            return corrected_result
            
        except Exception as e:
            self.logger.error(f"Error in post-processing corrections: {e}")
            return result
    
    def _create_fallback_result(self, law_content: str, law_name: str) -> Dict[str, Any]:
        """Create a fallback result when parsers are not available"""
        return {
            'law_name': law_name,
            'parsing_method': 'fallback',
            'parsing_timestamp': datetime.now().isoformat(),
            'all_articles': [],
            'main_articles': [],
            'supplementary_articles': [],
            'total_articles': 0,
            'quality_score': 0.0,
            'error': 'No parsers available',
            'manual_review_required': True
        }
    
    def _create_error_result(self, law_content: str, law_name: str, error_message: str) -> Dict[str, Any]:
        """Create an error result"""
        return {
            'law_name': law_name,
            'parsing_method': 'error',
            'parsing_timestamp': datetime.now().isoformat(),
            'all_articles': [],
            'main_articles': [],
            'supplementary_articles': [],
            'total_articles': 0,
            'quality_score': 0.0,
            'error': error_message,
            'manual_review_required': True
        }
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics and capabilities"""
        return {
            'rule_based_available': self.rule_based_parser is not None,
            'ml_enhanced_available': self.ml_parser is not None,
            'quality_validator_available': self.quality_validator is not None,
            'auto_correction_available': self.ml_parser is not None and hasattr(self.ml_parser, 'fix_missing_articles'),
            'parsing_methods': ['rule_based', 'ml_enhanced', 'hybrid'],
            'quality_thresholds': {
                'excellent': 0.9,
                'good': 0.8,
                'acceptable': 0.7,
                'needs_review': 0.6,
                'poor': 0.5
            }
        }


def parse_law_with_hybrid_parser(law_content: str, law_name: str = "Unknown") -> Dict[str, Any]:
    """
    Convenience function to parse law content with hybrid parser
    
    Args:
        law_content: Law content to parse
        law_name: Name of the law
        
    Returns:
        Dict[str, Any]: Parsed result
    """
    parser = HybridArticleParser()
    return parser.parse_law_content(law_content, law_name)


if __name__ == "__main__":
    # Test the hybrid parser
    test_content = """
    제1조(목적) 이 법은 민사에 관한 기본법이다.
    
    제2조(적용 범위) 이 법은 다음 각 호의 어느 하나에 해당하는 경우에 적용한다.
    1. 민사관계
    2. 상사관계
    
    제3조(해석) 민법은 공정과 신의에 따라 해석한다.
    
    부칙제1조(시행일) 이 법은 공포한 날부터 시행한다.
    """
    
    parser = HybridArticleParser()
    result = parser.parse_law_content(test_content, "민법")
    
    print("=== Hybrid Parsing Result ===")
    print(f"Law: {result['law_name']}")
    print(f"Method: {result['parsing_method']}")
    print(f"Quality Score: {result['final_result'].get('quality_score', 0):.3f}")
    print(f"Total Articles: {result['final_result'].get('total_articles', 0)}")
    print(f"Manual Review Required: {result['manual_review_required']}")
    
    if result['final_result'].get('all_articles'):
        print("\nArticles:")
        for article in result['final_result']['all_articles']:
            print(f"- {article.get('article_number', 'Unknown')}: {article.get('article_title', 'No title')}")
