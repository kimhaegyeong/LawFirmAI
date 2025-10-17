#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Validator

This module provides comprehensive data quality validation for parsed law data,
including article count consistency, title extraction accuracy, and legal structure completeness.

Usage:
    validator = DataQualityValidator()
    quality_report = validator.validate_parsing_quality(law_data)
    quality_score = validator.calculate_quality_score(law_data)
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality validation report"""
    overall_score: float
    article_count_score: float
    title_extraction_score: float
    article_sequence_score: float
    structure_completeness_score: float
    issues: List[str]
    suggestions: List[str]
    validation_timestamp: str
    validation_results: Dict[str, Any]  # 추가: 상세 검증 결과


class DataQualityValidator:
    """Comprehensive data quality validator for law parsing results"""
    
    def __init__(self):
        """Initialize the quality validator"""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_article_count = 1
        self.max_article_count = 1000
        self.min_title_coverage = 0.7  # 70% of articles should have titles
        self.max_sequence_gap = 5  # Maximum gap in article numbering
        
        # Legal structure patterns
        self.legal_patterns = {
            'article_start': r'제\s*\d+\s*조',
            'supplementary_start': r'부칙',
            'paragraph_start': r'제\s*\d+\s*항',
            'subparagraph_start': r'제\s*\d+\s*호',
            'law_title': r'법|규칙|령|규정|조치|지침'
        }
    
    def validate_parsing_quality(self, law_data: Dict[str, Any]) -> QualityReport:
        """
        Validate parsing quality of law data
        
        Args:
            law_data: Parsed law data dictionary
            
        Returns:
            QualityReport: Comprehensive quality validation report
        """
        try:
            self.logger.info(f"Validating quality for law: {law_data.get('law_name', 'Unknown')}")
            
            # Extract articles for validation
            articles = self._extract_articles(law_data)
            
            # Calculate individual quality scores
            article_count_score = self._validate_article_count(articles)
            title_extraction_score = self._validate_title_extraction(articles)
            article_sequence_score = self._validate_article_sequence(articles)
            structure_completeness_score = self._validate_structure_completeness(law_data, articles)
            
            # Calculate overall score (weighted average)
            overall_score = (
                article_count_score * 0.3 +
                title_extraction_score * 0.25 +
                article_sequence_score * 0.25 +
                structure_completeness_score * 0.2
            )
            
            # Generate issues and suggestions
            issues = self._identify_issues(articles, law_data)
            suggestions = self._generate_suggestions(issues, articles)
            
            # Create detailed validation results
            validation_results = {
                'article_count': len(articles),
                'article_count_score': article_count_score,
                'title_extraction_score': title_extraction_score,
                'article_sequence_score': article_sequence_score,
                'structure_completeness_score': structure_completeness_score,
                'overall_score': overall_score,
                'issues_count': len(issues),
                'suggestions_count': len(suggestions)
            }
            
            report = QualityReport(
                overall_score=overall_score,
                article_count_score=article_count_score,
                title_extraction_score=title_extraction_score,
                article_sequence_score=article_sequence_score,
                structure_completeness_score=structure_completeness_score,
                issues=issues,
                suggestions=suggestions,
                validation_timestamp=datetime.now().isoformat(),
                validation_results=validation_results
            )
            
            self.logger.info(f"Quality validation completed. Overall score: {overall_score:.3f}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error during quality validation: {e}")
            return QualityReport(
                overall_score=0.0,
                article_count_score=0.0,
                title_extraction_score=0.0,
                article_sequence_score=0.0,
                structure_completeness_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Fix parsing errors before validation"],
                validation_timestamp=datetime.now().isoformat()
            )
    
    def calculate_quality_score(self, law_data: Dict[str, Any]) -> float:
        """
        Calculate overall quality score for law data
        
        Args:
            law_data: Parsed law data dictionary
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        report = self.validate_parsing_quality(law_data)
        return report.overall_score
    
    def suggest_improvements(self, law_data: Dict[str, Any]) -> List[str]:
        """
        Generate improvement suggestions for law data
        
        Args:
            law_data: Parsed law data dictionary
            
        Returns:
            List[str]: List of improvement suggestions
        """
        report = self.validate_parsing_quality(law_data)
        return report.suggestions
    
    def _extract_articles(self, law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract articles from law data"""
        articles = []
        
        # Try different possible article structures
        if 'articles' in law_data:
            articles = law_data['articles']
        elif 'all_articles' in law_data:
            articles = law_data['all_articles']
        elif 'parsed_articles' in law_data:
            articles = law_data['parsed_articles']
        elif isinstance(law_data.get('parsed_result'), dict) and 'all_articles' in law_data['parsed_result']:
            articles = law_data['parsed_result']['all_articles']
        
        # Ensure articles is a list
        if not isinstance(articles, list):
            articles = []
        
        return articles
    
    def _validate_article_count(self, articles: List[Dict[str, Any]]) -> float:
        """Validate article count consistency"""
        article_count = len(articles)
        
        if article_count == 0:
            return 0.0
        elif article_count < self.min_article_count:
            return 0.3
        elif article_count > self.max_article_count:
            return 0.5
        elif 1 <= article_count <= 10:
            return 1.0
        elif 11 <= article_count <= 50:
            return 0.9
        elif 51 <= article_count <= 100:
            return 0.8
        else:
            return 0.7
    
    def _validate_title_extraction(self, articles: List[Dict[str, Any]]) -> float:
        """Validate title extraction completeness"""
        if not articles:
            return 0.0
        
        articles_with_titles = 0
        total_articles = len(articles)
        
        for article in articles:
            # Check various possible title fields
            title_fields = ['title', 'article_title', 'heading', 'name']
            has_title = False
            
            for field in title_fields:
                if field in article and article[field] and str(article[field]).strip():
                    has_title = True
                    break
            
            if has_title:
                articles_with_titles += 1
        
        title_coverage = articles_with_titles / total_articles if total_articles > 0 else 0
        
        if title_coverage >= self.min_title_coverage:
            return 1.0
        elif title_coverage >= 0.5:
            return 0.7
        elif title_coverage >= 0.3:
            return 0.4
        else:
            return 0.1
    
    def _validate_article_sequence(self, articles: List[Dict[str, Any]]) -> float:
        """Validate article number sequence"""
        if not articles:
            return 0.0
        
        article_numbers = []
        
        for article in articles:
            # Extract article number from various possible fields
            number_fields = ['number', 'article_number', 'num', 'id']
            article_number = None
            
            for field in number_fields:
                if field in article and article[field]:
                    try:
                        # Extract number from string like "제1조", "1조", "1"
                        number_text = str(article[field])
                        number_match = re.search(r'\d+', number_text)
                        if number_match:
                            article_number = int(number_match.group())
                            break
                    except (ValueError, TypeError):
                        continue
            
            if article_number is not None:
                article_numbers.append(article_number)
        
        if not article_numbers:
            return 0.3  # No article numbers found
        
        # Check sequence consistency
        article_numbers.sort()
        gaps = []
        for i in range(1, len(article_numbers)):
            gap = article_numbers[i] - article_numbers[i-1]
            gaps.append(gap)
        
        if not gaps:
            return 1.0
        
        max_gap = max(gaps)
        avg_gap = sum(gaps) / len(gaps)
        
        # Score based on gap consistency
        if max_gap <= 1:
            return 1.0
        elif max_gap <= 2:
            return 0.8
        elif max_gap <= self.max_sequence_gap:
            return 0.6
        else:
            return 0.3
    
    def _validate_structure_completeness(self, law_data: Dict[str, Any], articles: List[Dict[str, Any]]) -> float:
        """Validate legal structure completeness"""
        score = 0.0
        checks = 0
        
        # Check for law title
        law_name = law_data.get('law_name', '')
        if law_name and any(pattern in law_name for pattern in ['법', '규칙', '령', '규정']):
            score += 1.0
        checks += 1
        
        # Check for articles
        if articles:
            score += 1.0
        checks += 1
        
        # Check for supplementary articles (부칙)
        has_supplementary = False
        for article in articles:
            article_text = str(article.get('content', '')) + str(article.get('text', ''))
            if '부칙' in article_text:
                has_supplementary = True
                break
        
        if has_supplementary:
            score += 0.5
        checks += 1
        
        # Check for proper article structure
        proper_structure_count = 0
        for article in articles:
            content = str(article.get('content', '')) + str(article.get('text', ''))
            if re.search(self.legal_patterns['article_start'], content):
                proper_structure_count += 1
        
        if articles:
            structure_ratio = proper_structure_count / len(articles)
            score += structure_ratio
        checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def _identify_issues(self, articles: List[Dict[str, Any]], law_data: Dict[str, Any]) -> List[str]:
        """Identify specific issues in the parsed data"""
        issues = []
        
        # Article count issues
        article_count = len(articles)
        if article_count == 0:
            issues.append("No articles found in parsed data")
        elif article_count < self.min_article_count:
            issues.append(f"Too few articles: {article_count}")
        elif article_count > self.max_article_count:
            issues.append(f"Too many articles: {article_count}")
        
        # Title extraction issues
        articles_without_titles = 0
        for article in articles:
            title_fields = ['title', 'article_title', 'heading', 'name']
            has_title = any(field in article and article[field] and str(article[field]).strip() 
                          for field in title_fields)
            if not has_title:
                articles_without_titles += 1
        
        if articles_without_titles > 0:
            issues.append(f"{articles_without_titles} articles missing titles")
        
        # Article sequence issues
        article_numbers = []
        for article in articles:
            number_fields = ['number', 'article_number', 'num', 'id']
            for field in number_fields:
                if field in article and article[field]:
                    try:
                        number_text = str(article[field])
                        number_match = re.search(r'\d+', number_text)
                        if number_match:
                            article_numbers.append(int(number_match.group()))
                            break
                    except (ValueError, TypeError):
                        continue
        
        if article_numbers:
            article_numbers.sort()
            gaps = [article_numbers[i] - article_numbers[i-1] for i in range(1, len(article_numbers))]
            if gaps and max(gaps) > self.max_sequence_gap:
                issues.append(f"Large gaps in article numbering: {max(gaps)}")
        
        # Structure issues
        law_name = law_data.get('law_name', '')
        if not law_name or not any(pattern in law_name for pattern in ['법', '규칙', '령', '규정']):
            issues.append("Law name does not follow standard legal naming convention")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str], articles: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions based on identified issues"""
        suggestions = []
        
        for issue in issues:
            if "No articles found" in issue:
                suggestions.append("Check parsing logic for article extraction")
            elif "Too few articles" in issue:
                suggestions.append("Review article boundary detection algorithm")
            elif "Too many articles" in issue:
                suggestions.append("Implement article filtering to remove noise")
            elif "missing titles" in issue:
                suggestions.append("Improve title extraction from article content")
            elif "gaps in article numbering" in issue:
                suggestions.append("Check for missing articles in sequence")
            elif "naming convention" in issue:
                suggestions.append("Validate law name format during parsing")
        
        # General suggestions based on quality
        if articles:
            title_coverage = sum(1 for article in articles 
                               if any(field in article and article[field] and str(article[field]).strip()
                                    for field in ['title', 'article_title', 'heading', 'name'])) / len(articles)
            
            if title_coverage < 0.5:
                suggestions.append("Consider implementing ML-based title extraction")
            
            if len(articles) > 50:
                suggestions.append("Consider chunking large laws for better processing")
        
        return suggestions


def validate_law_data_quality(law_data: Dict[str, Any]) -> QualityReport:
    """
    Convenience function to validate law data quality
    
    Args:
        law_data: Parsed law data dictionary
        
    Returns:
        QualityReport: Quality validation report
    """
    validator = DataQualityValidator()
    return validator.validate_parsing_quality(law_data)


if __name__ == "__main__":
    # Test the validator with sample data
    sample_law_data = {
        'law_name': '민법',
        'articles': [
            {'number': '1', 'title': '민법의 목적', 'content': '이 법은 민사에 관한 기본법이다.'},
            {'number': '2', 'title': '민법의 적용', 'content': '민법은 민사에 관하여 적용한다.'},
            {'number': '3', 'title': '민법의 해석', 'content': '민법은 공정과 신의에 따라 해석한다.'}
        ]
    }
    
    validator = DataQualityValidator()
    report = validator.validate_parsing_quality(sample_law_data)
    
    print(f"Overall Quality Score: {report.overall_score:.3f}")
    print(f"Issues: {report.issues}")
    print(f"Suggestions: {report.suggestions}")
