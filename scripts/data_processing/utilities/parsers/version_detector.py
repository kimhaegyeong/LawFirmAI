"""
Version Detection System for Assembly Law Data

This module detects the version of raw law data based on structure analysis
and provides version-specific parsing capabilities.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DataVersionDetector:
    """Raw ?°ì´??êµ¬ì¡°ë¥?ë¶„ì„?˜ì—¬ ?Œì‹± ë²„ì „???ë™ ê°ì?"""
    
    def __init__(self):
        """Initialize version detector with pattern definitions"""
        self.version_patterns = {
            'v1.0': {
                'required_fields': ['law_name', 'law_content', 'content_html'],
                'optional_fields': ['row_number', 'category', 'law_type'],
                'date_patterns': ['YYYY.M.D.', 'YYYY??M??D??],
                'html_structure': 'basic_html',
                'weight': 1.0
            },
            'v1.1': {
                'required_fields': ['law_name', 'law_content', 'content_html', 'promulgation_number'],
                'optional_fields': ['row_number', 'category', 'law_type', 'enforcement_date'],
                'date_patterns': ['YYYY.M.D.', 'YYYY??M??D??, 'YYYY-MM-DD'],
                'html_structure': 'enhanced_html',
                'weight': 1.0
            },
            'v1.2': {
                'required_fields': ['law_name', 'law_content', 'content_html', 'promulgation_number', 'amendment_type'],
                'optional_fields': ['row_number', 'category', 'law_type', 'enforcement_date', 'cont_id'],
                'date_patterns': ['YYYY.M.D.', 'YYYY??M??D??, 'YYYY-MM-DD', 'YYYY.MM.DD'],
                'html_structure': 'structured_html',
                'weight': 1.0
            }
        }
        
        # Date pattern regexes
        self.date_regexes = {
            'YYYY.M.D.': re.compile(r'\d{4}\.\d{1,2}\.\d{1,2}\.'),
            'YYYY??M??D??: re.compile(r'\d{4}??s*\d{1,2}??s*\d{1,2}??),
            'YYYY-MM-DD': re.compile(r'\d{4}-\d{2}-\d{2}'),
            'YYYY.MM.DD': re.compile(r'\d{4}\.\d{2}\.\d{2}')
        }
    
    def detect_version(self, raw_data: Dict[str, Any]) -> str:
        """
        Raw ?°ì´??êµ¬ì¡°ë¥?ë¶„ì„?˜ì—¬ ë²„ì „ ê°ì?
        
        Args:
            raw_data (Dict[str, Any]): Raw law data dictionary
            
        Returns:
            str: Detected version (v1.0, v1.1, v1.2)
        """
        try:
            # ê°?ë²„ì „ë³??ìˆ˜ ê³„ì‚°
            version_scores = {}
            
            for version, patterns in self.version_patterns.items():
                field_score = self._analyze_fields(raw_data, patterns)
                date_score = self._analyze_date_formats(raw_data, patterns)
                html_score = self._analyze_html_structure(raw_data, patterns)
                
                # ê°€ì¤??‰ê· ?¼ë¡œ ìµœì¢… ?ìˆ˜ ê³„ì‚°
                total_score = (
                    field_score * 0.4 +
                    date_score * 0.3 +
                    html_score * 0.3
                ) * patterns['weight']
                
                version_scores[version] = total_score
            
            # ê°€???’ì? ?ìˆ˜??ë²„ì „ ë°˜í™˜
            detected_version = max(version_scores, key=version_scores.get)
            
            logger.debug(f"Version detection scores: {version_scores}")
            logger.debug(f"Detected version: {detected_version}")
            
            return detected_version
            
        except Exception as e:
            logger.error(f"Error detecting version: {e}")
            return 'v1.2'  # Default to latest version
    
    def get_confidence(self, raw_data: Dict[str, Any], version: str) -> float:
        """
        ?¹ì • ë²„ì „???€??? ë¢°??ê³„ì‚°
        
        Args:
            raw_data (Dict[str, Any]): Raw law data dictionary
            version (str): Version to check confidence for
            
        Returns:
            float: Confidence score (0.0 - 1.0)
        """
        if version not in self.version_patterns:
            return 0.0
        
        patterns = self.version_patterns[version]
        
        field_score = self._analyze_fields(raw_data, patterns)
        date_score = self._analyze_date_formats(raw_data, patterns)
        html_score = self._analyze_html_structure(raw_data, patterns)
        
        confidence = (field_score * 0.4 + date_score * 0.3 + html_score * 0.3)
        return min(1.0, max(0.0, confidence))
    
    def _analyze_fields(self, raw_data: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """
        ?„ë“œ ì¡´ì¬ ?¬ë?ë¡?ë²„ì „ ë¶„ì„
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            patterns (Dict[str, Any]): Version patterns
            
        Returns:
            float: Field analysis score (0.0 - 1.0)
        """
        required_fields = patterns['required_fields']
        optional_fields = patterns['optional_fields']
        
        # ?„ìˆ˜ ?„ë“œ ?ìˆ˜ (ê°€ì¤‘ì¹˜ ?’ìŒ)
        required_count = sum(1 for field in required_fields if field in raw_data and raw_data[field])
        required_score = required_count / len(required_fields) if required_fields else 0.0
        
        # ? íƒ ?„ë“œ ?ìˆ˜ (ê°€ì¤‘ì¹˜ ??Œ)
        optional_count = sum(1 for field in optional_fields if field in raw_data and raw_data[field])
        optional_score = optional_count / len(optional_fields) if optional_fields else 0.0
        
        # ?„ìˆ˜ ?„ë“œ 70%, ? íƒ ?„ë“œ 30% ê°€ì¤‘ì¹˜
        return required_score * 0.7 + optional_score * 0.3
    
    def _analyze_date_formats(self, raw_data: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """
        ? ì§œ ?•ì‹?¼ë¡œ ë²„ì „ ë¶„ì„
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            patterns (Dict[str, Any]): Version patterns
            
        Returns:
            float: Date format analysis score (0.0 - 1.0)
        """
        date_fields = ['promulgation_date', 'enforcement_date']
        supported_patterns = patterns['date_patterns']
        
        total_matches = 0
        total_fields = 0
        
        for field in date_fields:
            if field in raw_data and raw_data[field]:
                total_fields += 1
                date_value = str(raw_data[field])
                
                # ì§€?ë˜???¨í„´ ì¤??˜ë‚˜?¼ë„ ë§¤ì¹˜?˜ë©´ ?ìˆ˜ ?ë“
                for pattern_name in supported_patterns:
                    if pattern_name in self.date_regexes:
                        if self.date_regexes[pattern_name].search(date_value):
                            total_matches += 1
                            break
        
        return total_matches / total_fields if total_fields > 0 else 0.0
    
    def _analyze_html_structure(self, raw_data: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """
        HTML êµ¬ì¡°ë¡?ë²„ì „ ë¶„ì„
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            patterns (Dict[str, Any]): Version patterns
            
        Returns:
            float: HTML structure analysis score (0.0 - 1.0)
        """
        html_content = raw_data.get('content_html', '')
        if not html_content:
            return 0.0
        
        structure_type = patterns['html_structure']
        
        if structure_type == 'basic_html':
            # ê¸°ë³¸ HTML êµ¬ì¡° (html, body ?œê·¸ë§?
            score = 1.0 if '<html>' in html_content and '<body>' in html_content else 0.0
            
        elif structure_type == 'enhanced_html':
            # ?¥ìƒ??HTML êµ¬ì¡° (div, span ?œê·¸ ?¬í•¨)
            has_div = '<div' in html_content
            has_span = '<span' in html_content
            has_enhanced = has_div or has_span
            score = 1.0 if has_enhanced else 0.0
            
        elif structure_type == 'structured_html':
            # êµ¬ì¡°?”ëœ HTML (article ?œê·¸, data ?ì„± ?¬í•¨)
            has_article = '<article' in html_content
            has_data_attr = 'data-article' in html_content or 'data-law' in html_content
            has_structured = has_article or has_data_attr
            score = 1.0 if has_structured else 0.0
            
        else:
            score = 0.0
        
        return score
    
    def get_version_info(self, version: str) -> Dict[str, Any]:
        """
        ë²„ì „ ?•ë³´ ë°˜í™˜
        
        Args:
            version (str): Version identifier
            
        Returns:
            Dict[str, Any]: Version information
        """
        if version not in self.version_patterns:
            return {}
        
        patterns = self.version_patterns[version]
        
        return {
            'version': version,
            'required_fields': patterns['required_fields'],
            'optional_fields': patterns['optional_fields'],
            'supported_date_formats': patterns['date_patterns'],
            'html_structure': patterns['html_structure'],
            'weight': patterns['weight']
        }
    
    def get_supported_versions(self) -> List[str]:
        """
        ì§€?ë˜??ë²„ì „ ëª©ë¡ ë°˜í™˜
        
        Returns:
            List[str]: List of supported versions
        """
        return list(self.version_patterns.keys())
    
    def validate_data_compatibility(self, raw_data: Dict[str, Any], version: str) -> Dict[str, Any]:
        """
        ?°ì´?°ì? ë²„ì „???¸í™˜??ê²€ì¦?
        
        Args:
            raw_data (Dict[str, Any]): Raw data dictionary
            version (str): Version to validate against
            
        Returns:
            Dict[str, Any]: Compatibility validation results
        """
        if version not in self.version_patterns:
            return {
                'compatible': False,
                'error': f'Unknown version: {version}',
                'missing_fields': [],
                'extra_fields': []
            }
        
        patterns = self.version_patterns[version]
        required_fields = patterns['required_fields']
        
        # ?„ìˆ˜ ?„ë“œ ê²€??
        missing_fields = []
        for field in required_fields:
            if field not in raw_data or not raw_data[field]:
                missing_fields.append(field)
        
        # ?¸í™˜???ë‹¨
        compatible = len(missing_fields) == 0
        
        return {
            'compatible': compatible,
            'missing_fields': missing_fields,
            'confidence': self.get_confidence(raw_data, version),
            'validation_timestamp': datetime.now().isoformat()
        }
