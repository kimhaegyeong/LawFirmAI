"""
Text Normalizer for Assembly Law Data

This module normalizes and cleans legal text by removing duplicate whitespace,
standardizing legal terminology, converting special characters, and
standardizing date formats.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Text normalizer for cleaning and standardizing legal text"""
    
    def __init__(self):
        """Initialize the text normalizer"""
        # Date format patterns
        self.date_patterns = [
            (r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.', r'\1-\2-\3'),  # 2025.10.2. -> 2025-10-2
            (r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', r'\1-\2-\3'),  # 2025년 10월 2일 -> 2025-10-2
            (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', r'\1-\2-\3'),  # 2025.10.2 -> 2025-10-2
        ]
        
        # Article patterns
        self.article_patterns = [
            (r'제(\d+)조', r'제\1조'),
            (r'제(\d+)항', r'제\1항'),
            (r'제(\d+)호', r'제\1호'),
            (r'제(\d+)목', r'제\1목'),
        ]
        
        # Legal terminology normalization
        self.legal_terms = {
            '같은 법': 'parent_law_reference',
            '이 법': 'self_reference',
            '동법': 'same_law_reference',
            '상위법': 'superior_law',
            '관련법': 'related_law',
            '시행령': 'enforcement_decree',
            '시행규칙': 'enforcement_rule',
            '부령': 'ministry_ordinance',
            '대통령령': 'presidential_decree',
            '총리령': 'prime_minister_decree'
        }
        
        # UI elements to remove
        self.ui_patterns = [
            r'조문버튼선택체크',
            r'펼치기접기',
            r'선택체크',
            r'펼치기',
            r'접기',
            r'버튼',
            r'선택',
            r'체크'
        ]
        self.whitespace_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'\n\s*\n', '\n'),  # Multiple newlines to single newline
            (r'^\s+|\s+$', ''),  # Trim whitespace
        ]
        
        # Special character patterns
        self.special_char_patterns = [
            (r'「', '"'),
            (r'」', '"'),
            (r'『', '"'),
            (r'』', '"'),
            (r'〈', '<'),
            (r'〉', '>'),
            (r'〔', '['),
            (r'〕', ']'),
        ]
        
        # Amendment markers
        self.amendment_patterns = [
            (r'<개정\s+[^>]+>', ''),  # Remove amendment markers
            (r'<신설>', ''),
            (r'<폐지>', ''),
            (r'<일부개정>', ''),
            (r'<전부개정>', ''),
        ]
    
    def normalize(self, text: str) -> str:
        """
        Normalize text content
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Normalized text
        """
        try:
            if not text or not isinstance(text, str):
                return ''
            
            # Start with the original text
            normalized_text = text
            
            # Remove UI elements
            normalized_text = self._remove_ui_elements(normalized_text)
            
            # Remove amendment markers
            normalized_text = self._remove_amendment_markers(normalized_text)
            
            # Normalize special characters
            normalized_text = self._normalize_special_characters(normalized_text)
            
            # Normalize date formats
            normalized_text = self._normalize_dates(normalized_text)
            
            # Normalize article patterns
            normalized_text = self._normalize_articles(normalized_text)
            
            # Normalize legal terminology
            normalized_text = self._normalize_legal_terms(normalized_text)
            
            # Clean whitespace
            normalized_text = self._clean_whitespace(normalized_text)
            
            return normalized_text.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return text if text else ''
    
    def _remove_ui_elements(self, text: str) -> str:
        """
        Remove UI elements from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with UI elements removed
        """
        try:
            normalized_text = text
            
            for pattern in self.ui_patterns:
                normalized_text = re.sub(pattern, '', normalized_text)
            
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error removing UI elements: {e}")
            return text
    
    def _remove_amendment_markers(self, text: str) -> str:
        """
        Remove amendment markers from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with amendment markers removed
        """
        for pattern, replacement in self.amendment_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _normalize_special_characters(self, text: str) -> str:
        """
        Normalize special characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized special characters
        """
        for pattern, replacement in self.special_char_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _normalize_dates(self, text: str) -> str:
        """
        Normalize date formats
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized dates
        """
        for pattern, replacement in self.date_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _normalize_articles(self, text: str) -> str:
        """
        Normalize article patterns
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized article patterns
        """
        for pattern, replacement in self.article_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _normalize_legal_terms(self, text: str) -> str:
        """
        Normalize legal terminology
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized legal terms
        """
        # This is a placeholder - in practice, you might want to keep
        # the original terms but add normalized versions for search
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Clean up whitespace
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with cleaned whitespace
        """
        for pattern, replacement in self.whitespace_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def normalize_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize article content
        
        Args:
            article (Dict[str, Any]): Article dictionary
            
        Returns:
            Dict[str, Any]: Normalized article
        """
        try:
            normalized_article = article.copy()
            
            # Normalize main content
            if 'article_content' in normalized_article:
                normalized_article['article_content'] = self.normalize(
                    normalized_article['article_content']
                )
            
            # Normalize sub-articles
            if 'sub_articles' in normalized_article:
                normalized_sub_articles = []
                for sub_article in normalized_article['sub_articles']:
                    normalized_sub = sub_article.copy()
                    if 'content' in normalized_sub:
                        normalized_sub['content'] = self.normalize(
                            normalized_sub['content']
                        )
                    normalized_sub_articles.append(normalized_sub)
                normalized_article['sub_articles'] = normalized_sub_articles
            
            return normalized_article
            
        except Exception as e:
            logger.error(f"Error normalizing article: {e}")
            return article
    
    # extract_keywords와 generate_searchable_text 메서드 제거됨 - 더 이상 사용하지 않음
    
    def validate_normalization(self, original_text: str, normalized_text: str) -> Dict[str, Any]:
        """
        Validate normalization results
        
        Args:
            original_text (str): Original text
            normalized_text (str): Normalized text
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'original_length': len(original_text) if original_text else 0,
            'normalized_length': len(normalized_text) if normalized_text else 0,
            'length_change': 0,
            'has_amendment_markers': bool(re.search(r'<개정|신설|폐지>', original_text or '')),
            'has_special_chars': bool(re.search(r'「|」|『|』', original_text or '')),
            'has_multiple_spaces': bool(re.search(r'\s{2,}', original_text or '')),
            'normalization_score': 0.0
        }
        
        # Calculate length change
        if validation_results['original_length'] > 0:
            validation_results['length_change'] = (
                validation_results['normalized_length'] - validation_results['original_length']
            ) / validation_results['original_length']
        
        # Calculate normalization score
        improvements = 0
        total_checks = 3
        
        if not validation_results['has_amendment_markers']:
            improvements += 1
        if not validation_results['has_special_chars']:
            improvements += 1
        if not validation_results['has_multiple_spaces']:
            improvements += 1
        
        validation_results['normalization_score'] = improvements / total_checks
        
        return validation_results
