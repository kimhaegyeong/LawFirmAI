"""
Metadata Extractor for Assembly Law Data

This module extracts metadata from law data including enforcement dates,
amendment history, legal references, and ministry information.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extractor for law metadata including dates, amendments, and references"""
    
    def __init__(self):
        """Initialize the metadata extractor"""
        # Date patterns
        self.enforcement_date_pattern = re.compile(r'\[?œí–‰\s+([^\]]+)\]')
        
        # Amendment patterns - more flexible
        self.amendment_pattern = re.compile(r'([^,]+),\s*([^,]+),\s*([^,]+)')
        self.amendment_type_pattern = re.compile(r'(?¼ë?ê°œì •|?„ë?ê°œì •|? ì„¤|?ì?)')
        
        # Ministry patterns
        self.ministry_patterns = [
            r'([ê°€-??+ë¶€??',
            r'([ê°€-??+ë¶€)',
            r'([ê°€-??+ì²?',
            r'([ê°€-??+??',
            r'([ê°€-??+?„ì›??',
            r'([ê°€-??+ì²?'
        ]
        
        # Law type patterns
        self.law_type_patterns = {
            'ë²•ë¥ ': r'ë²•ë¥ ',
            '?œí–‰??: r'?œí–‰??,
            '?œí–‰ê·œì¹™': r'?œí–‰ê·œì¹™',
            'ë¶€??: r'ë¶€??,
            '?€?µë ¹??: r'?€?µë ¹??,
            'ì´ë¦¬??: r'ì´ë¦¬??,
            'ë¶€ê³ ì‹œ': r'ë¶€ê³ ì‹œ',
            'ë¶€?ˆë ¹': r'ë¶€?ˆë ¹'
        }
        
        # Reference patterns
        self.reference_patterns = [
            r'??[^??+)??,
            r'ê°™ì?\s+ë²?,
            r'?™ë²•',
            r'??s+ë²?,
            r'?ìœ„ë²?,
            r'ê´€?¨ë²•'
        ]
    
    def extract(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from law data
        
        Args:
            law_data (Dict[str, Any]): Raw law data dictionary
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        try:
            metadata = {}
            
            # Extract enforcement information
            metadata['enforcement_info'] = self._extract_enforcement_info(
                law_data.get('law_content', '')
            )
            
            # Extract amendment information
            metadata['amendment_info'] = self._extract_amendment_info(law_data)
            
            # Extract ministry information
            metadata['ministry'] = self._extract_ministry(law_data)
            
            # Extract law type
            metadata['law_type'] = self._extract_law_type(law_data)
            
            # Extract references
            metadata['references'] = self._extract_references(
                law_data.get('law_content', '')
            )
            
            # Extract parent law
            metadata['parent_law'] = self._extract_parent_law(law_data)
            
            # Extract related laws
            metadata['related_laws'] = self._extract_related_laws(
                law_data.get('law_content', '')
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def _extract_enforcement_info(self, content: str) -> Dict[str, Any]:
        """
        Extract enforcement date information
        
        Args:
            content (str): Law content text
            
        Returns:
            Dict[str, Any]: Enforcement information
        """
        enforcement_info = {
            'date': None,
            'text': None,
            'parsed_date': None
        }
        
        # Find enforcement date pattern
        match = self.enforcement_date_pattern.search(content)
        if match:
            enforcement_text = match.group(1).strip()
            enforcement_info['text'] = f"[?œí–‰ {enforcement_text}]"
            enforcement_info['date'] = enforcement_text
            
            # Try to parse the date
            parsed_date = self._parse_date(enforcement_text)
            if parsed_date:
                enforcement_info['parsed_date'] = parsed_date.isoformat()
        
        return enforcement_info
    
    def _extract_amendment_info(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract amendment information
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Amendment information
        """
        amendment_info = {
            'number': None,
            'date': None,
            'type': None,
            'ministry': None,
            'parsed_date': None
        }
        
        # Extract from promulgation_number if available
        promulgation_number = law_data.get('promulgation_number', '')
        if promulgation_number:
            amendment_info['number'] = promulgation_number
            
            # Extract ministry from promulgation number
            for pattern in self.ministry_patterns:
                match = re.search(pattern, promulgation_number)
                if match:
                    amendment_info['ministry'] = match.group(1)
                    break
        
        # Extract amendment type
        amendment_type = law_data.get('amendment_type', '')
        if amendment_type:
            amendment_info['type'] = amendment_type
        
        # Extract amendment date
        promulgation_date = law_data.get('promulgation_date', '')
        if promulgation_date:
            amendment_info['date'] = promulgation_date
            parsed_date = self._parse_date(promulgation_date)
            if parsed_date:
                amendment_info['parsed_date'] = parsed_date.isoformat()
        
        return amendment_info
    
    def _extract_ministry(self, law_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract ministry/department information
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Optional[str]: Ministry name
        """
        # Try to extract from promulgation number first
        promulgation_number = law_data.get('promulgation_number', '')
        if promulgation_number:
            for pattern in self.ministry_patterns:
                match = re.search(pattern, promulgation_number)
                if match:
                    return match.group(1)
        
        # Try to extract from law name
        law_name = law_data.get('law_name', '')
        if law_name:
            for pattern in self.ministry_patterns:
                match = re.search(pattern, law_name)
                if match:
                    return match.group(1)
        
        return None
    
    def _extract_law_type(self, law_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract law type from law data
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Optional[str]: Law type
        """
        # Check law_type field first
        law_type = law_data.get('law_type')
        if law_type:
            return law_type
        
        # Extract from law name
        law_name = law_data.get('law_name', '')
        for law_type_name, pattern in self.law_type_patterns.items():
            if re.search(pattern, law_name):
                return law_type_name
        
        return None
    
    def _extract_references(self, content: str) -> List[Dict[str, str]]:
        """
        Extract legal references from content
        
        Args:
            content (str): Law content text
            
        Returns:
            List[Dict[str, str]]: List of references with types
        """
        references = []
        
        # Extract quoted law names
        quoted_pattern = re.compile(r'??[^??+)??)
        for match in quoted_pattern.finditer(content):
            law_name = match.group(1)
            ref_type = self._classify_reference_type(law_name)
            references.append({
                'type': ref_type,
                'name': law_name,
                'context': self._get_reference_context(content, match.start())
            })
        
        # Extract other reference types
        for pattern in self.reference_patterns[1:]:  # Skip quoted pattern
            if re.search(pattern, content):
                references.append({
                    'type': 'reference',
                    'name': pattern,
                    'context': 'general'
                })
        
        return references
    
    def _extract_parent_law(self, law_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract parent law from law data
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Optional[str]: Parent law name
        """
        law_name = law_data.get('law_name', '')
        
        # Look for ?œí–‰??or ?œí–‰ê·œì¹™ patterns
        if '?œí–‰?? in law_name:
            parent_match = re.search(r'([^?œí–‰??+)?œí–‰??, law_name)
            if parent_match:
                return parent_match.group(1).strip()
        elif '?œí–‰ê·œì¹™' in law_name:
            parent_match = re.search(r'([^?œí–‰ê·œì¹™]+)?œí–‰ê·œì¹™', law_name)
            if parent_match:
                return parent_match.group(1).strip()
        
        return None
    
    def _extract_related_laws(self, content: str) -> List[str]:
        """
        Extract related laws from content
        
        Args:
            content (str): Law content text
            
        Returns:
            List[str]: List of related law names
        """
        related_laws = []
        
        # Extract all quoted law names
        quoted_pattern = re.compile(r'??[^??+)??)
        for match in quoted_pattern.finditer(content):
            law_name = match.group(1)
            if law_name not in related_laws:
                related_laws.append(law_name)
        
        return related_laws
    
    def _classify_reference_type(self, law_name: str) -> str:
        """
        Classify the type of legal reference
        
        Args:
            law_name (str): Law name
            
        Returns:
            str: Reference type
        """
        if '?œí–‰?? in law_name:
            return 'enforcement_decree'
        elif '?œí–‰ê·œì¹™' in law_name:
            return 'enforcement_rule'
        elif 'ë²•ë¥ ' in law_name or law_name.endswith('ë²?):
            return 'parent_law'
        elif 'ë¶€?? in law_name or '?€?µë ¹?? in law_name:
            return 'regulation'
        else:
            return 'related_law'
    
    def _get_reference_context(self, content: str, position: int) -> str:
        """
        Get context around a reference
        
        Args:
            content (str): Full content
            position (int): Reference position
            
        Returns:
            str: Context string
        """
        start = max(0, position - 50)
        end = min(len(content), position + 50)
        context = content[start:end]
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string into datetime object
        
        Args:
            date_str (str): Date string
            
        Returns:
            Optional[datetime]: Parsed datetime or None
        """
        date_str = date_str.strip()
        
        # Common Korean date formats
        date_formats = [
            '%Y.%m.%d.',      # 2025.10.2.
            '%Y.%m.%d',       # 2025.10.2
            '%Y??%m??%d??,  # 2025??10??2??
            '%Y-%m-%d',       # 2025-10-02
            '%Y/%m/%d',       # 2025/10/02
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try to extract year, month, day from various patterns
        year_match = re.search(r'(\d{4})', date_str)
        month_match = re.search(r'(\d{1,2})', date_str)
        day_match = re.search(r'(\d{1,2})', date_str)
        
        if year_match and month_match and day_match:
            try:
                year = int(year_match.group(1))
                month = int(month_match.group(1))
                day = int(day_match.group(1))
                return datetime(year, month, day)
            except ValueError:
                pass
        
        return None
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted metadata
        
        Args:
            metadata (Dict[str, Any]): Extracted metadata
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'has_enforcement_date': bool(metadata.get('enforcement_info', {}).get('date')),
            'has_amendment_info': bool(metadata.get('amendment_info', {}).get('number')),
            'has_ministry': bool(metadata.get('ministry')),
            'has_law_type': bool(metadata.get('law_type')),
            'has_references': len(metadata.get('references', [])) > 0,
            'has_parent_law': bool(metadata.get('parent_law')),
            'completeness_score': 0.0
        }
        
        # Calculate completeness score
        total_fields = len(validation_results) - 1  # Exclude completeness_score
        completed_fields = sum(1 for v in validation_results.values() 
                             if isinstance(v, bool) and v)
        
        if total_fields > 0:
            validation_results['completeness_score'] = completed_fields / total_fields
        
        return validation_results
