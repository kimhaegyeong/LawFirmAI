"""
Legal Hierarchy Classifier for Korean Law Data

This module classifies Korean law data based on the legal hierarchy structure
including Constitution, Laws, Presidential Decrees, Ministry Ordinances, etc.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LegalHierarchyClassifier:
    """ë²•ë¥  ?„ê³„ êµ¬ì¡° ê¸°ë°˜ ë¶„ë¥˜ê¸?""
    
    def __init__(self):
        """Initialize hierarchy classifier with Korean legal system patterns"""
        self.hierarchy_patterns = {
            'constitution': {
                'keywords': ['?Œë²•', '?Œë²•?¬íŒ??, 'ê¸°ë³¸ê¶?, 'êµ??ê¸°ê?', '?Œë²•?¬íŒ'],
                'patterns': [r'?Œë²•\s*??d+ì¡?, r'?Œë²•?¬íŒ??, r'ê¸°ë³¸ê¶?, r'êµ??ê¸°ê?'],
                'level': 1,
                'weight': 1.0
            },
            'law': {
                'keywords': ['ë²•ë¥ ', 'êµ?šŒ', '?œì •', 'ê³µí¬', 'ë²?],
                'patterns': [r'ë²•ë¥ \s*??d+??, r'êµ?šŒ?ì„œ\s*?œì •', r'[ê°€-??+ë²?s*??d+ì¡?],
                'level': 2,
                'weight': 1.0
            },
            'presidential_decree': {
                'keywords': ['?€?µë ¹??, '?œí–‰??],
                'patterns': [r'?€?µë ¹??s*??d+??, r'?œí–‰??],
                'level': 3,
                'weight': 1.0
            },
            'prime_minister_decree': {
                'keywords': ['ì´ë¦¬??],
                'patterns': [r'ì´ë¦¬??s*??d+??],
                'level': 4,
                'weight': 1.0
            },
            'ministry_ordinance': {
                'keywords': ['ë¶€??, '?œí–‰ê·œì¹™'],
                'patterns': [r'[ê°€-??+ë¶€??s*??d+??, r'?œí–‰ê·œì¹™'],
                'level': 5,
                'weight': 1.0
            },
            'local_ordinance': {
                'keywords': ['ì¡°ë?', 'ê·œì¹™', '?œÂ·ë„', '?œìž¥', 'êµ°ìˆ˜', 'êµ¬ì²­??],
                'patterns': [r'[ê°€-??+??s*ì¡°ë?', r'[ê°€-??+??s*ì¡°ë?', r'[ê°€-??+êµ?s*ì¡°ë?'],
                'level': 6,
                'weight': 1.0
            }
        }
        
        # ë²•ë¥ ëª??¨í„´ (???•í™•??ë¶„ë¥˜ë¥??„í•´)
        self.law_name_patterns = {
            'constitution': re.compile(r'.*?Œë²•.*'),
            'law': re.compile(r'.*ë²?'),
            'presidential_decree': re.compile(r'.*?œí–‰??'),
            'ministry_ordinance': re.compile(r'.*?œí–‰ê·œì¹™$|.*ë¶€??'),
            'local_ordinance': re.compile(r'.*ì¡°ë?$|.*ê·œì¹™$')
        }
    
    def classify_law_hierarchy(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ë²•ë¥  ?„ê³„ ë¶„ë¥˜
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Hierarchy classification results
        """
        try:
            law_name = law_data.get('law_name', '')
            promulgation_number = law_data.get('promulgation_number', '')
            law_content = law_data.get('law_content', '')
            
            hierarchy_info = {
                'hierarchy_level': 0,
                'hierarchy_type': 'unknown',
                'hierarchy_confidence': 0.0,
                'parent_laws': [],
                'subordinate_laws': [],
                'related_hierarchy': [],
                'classification_method': 'unknown',
                'classification_timestamp': datetime.now().isoformat()
            }
            
            # ê°?ë°©ë²•ë³?ë¶„ë¥˜ ?˜í–‰
            name_classification = self._classify_by_name(law_name)
            promulgation_classification = self._classify_by_promulgation(promulgation_number)
            content_classification = self._classify_by_content(law_content)
            
            # ì¢…í•© ë¶„ë¥˜
            final_classification = self._combine_classifications(
                name_classification, promulgation_classification, content_classification
            )
            
            hierarchy_info.update(final_classification)
            
            # ê´€??ë²•ë¥  ë¶„ì„
            hierarchy_info['parent_laws'] = self._extract_parent_laws(law_content)
            hierarchy_info['subordinate_laws'] = self._extract_subordinate_laws(law_content)
            hierarchy_info['related_hierarchy'] = self._extract_related_hierarchy(law_content)
            
            return hierarchy_info
            
        except Exception as e:
            logger.error(f"Error classifying law hierarchy: {e}")
            return {
                'hierarchy_level': 0,
                'hierarchy_type': 'unknown',
                'hierarchy_confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_by_name(self, law_name: str) -> Dict[str, Any]:
        """ë²•ë¥ ëª?ê¸°ë°˜ ë¶„ë¥˜"""
        if not law_name:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'name'}
        
        # ë²•ë¥ ëª??¨í„´ ë§¤ì¹­
        for hierarchy_type, pattern in self.law_name_patterns.items():
            if pattern.match(law_name):
                hierarchy_info = self.hierarchy_patterns.get(hierarchy_type, {})
                return {
                    'hierarchy_type': hierarchy_type,
                    'hierarchy_level': hierarchy_info.get('level', 0),
                    'confidence': 0.9,
                    'method': 'name_pattern'
                }
        
        # ?¤ì›Œ??ê¸°ë°˜ ë¶„ë¥˜
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in law_name:
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.8,
                        'method': 'name_keyword'
                    }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'name'}
    
    def _classify_by_promulgation(self, promulgation_number: str) -> Dict[str, Any]:
        """ê³µí¬ë²ˆí˜¸ ê¸°ë°˜ ë¶„ë¥˜"""
        if not promulgation_number:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'promulgation'}
        
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for pattern in patterns['patterns']:
                if re.search(pattern, promulgation_number):
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.95,
                        'method': 'promulgation_pattern'
                    }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'promulgation'}
    
    def _classify_by_content(self, law_content: str) -> Dict[str, Any]:
        """?´ìš© ê¸°ë°˜ ë¶„ë¥˜"""
        if not law_content:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'content'}
        
        hierarchy_scores = {}
        
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            score = 0
            matched_patterns = []
            
            # ?¤ì›Œ??ë§¤ì¹­
            for keyword in patterns['keywords']:
                count = law_content.count(keyword)
                score += count
                if count > 0:
                    matched_patterns.append(f"keyword:{keyword}")
            
            # ?¨í„´ ë§¤ì¹­
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, law_content)
                score += len(matches) * 2  # ?¨í„´ ë§¤ì¹˜?????’ì? ê°€ì¤‘ì¹˜
                if matches:
                    matched_patterns.append(f"pattern:{pattern}")
            
            hierarchy_scores[hierarchy_type] = {
                'score': score,
                'matched_patterns': matched_patterns
            }
        
        # ê°€???’ì? ?ìˆ˜???„ê³„ ? íƒ
        if hierarchy_scores:
            best_type = max(hierarchy_scores, key=lambda x: hierarchy_scores[x]['score'])
            best_score = hierarchy_scores[best_type]['score']
            
            if best_score > 0:
                return {
                    'hierarchy_type': best_type,
                    'hierarchy_level': self.hierarchy_patterns[best_type]['level'],
                    'confidence': min(0.7, best_score / 10),
                    'method': 'content_analysis',
                    'matched_patterns': hierarchy_scores[best_type]['matched_patterns']
                }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'content'}
    
    def _combine_classifications(self, name_class: Dict, promulgation_class: Dict, 
                               content_class: Dict) -> Dict[str, Any]:
        """ë¶„ë¥˜ ê²°ê³¼ ì¢…í•©"""
        classifications = [name_class, promulgation_class, content_class]
        
        # ? íš¨??ë¶„ë¥˜ë§??„í„°ë§?
        valid_classifications = [c for c in classifications if c['hierarchy_type'] != 'unknown']
        
        if not valid_classifications:
            return {
                'hierarchy_type': 'unknown',
                'hierarchy_level': 0,
                'hierarchy_confidence': 0.0,
                'classification_method': 'combined'
            }
        
        # ê°€ì¤??‰ê· ?¼ë¡œ ìµœì¢… ë¶„ë¥˜ ê²°ì •
        hierarchy_votes = {}
        
        for classification in valid_classifications:
            hierarchy_type = classification['hierarchy_type']
            confidence = classification['confidence']
            
            if hierarchy_type not in hierarchy_votes:
                hierarchy_votes[hierarchy_type] = {
                    'total_confidence': 0.0,
                    'count': 0,
                    'methods': []
                }
            
            hierarchy_votes[hierarchy_type]['total_confidence'] += confidence
            hierarchy_votes[hierarchy_type]['count'] += 1
            hierarchy_votes[hierarchy_type]['methods'].append(classification.get('method', 'unknown'))
        
        # ê°€???’ì? ?‰ê·  ? ë¢°?„ì˜ ?„ê³„ ? íƒ
        best_hierarchy = max(hierarchy_votes, 
                           key=lambda x: hierarchy_votes[x]['total_confidence'] / hierarchy_votes[x]['count'])
        
        best_info = hierarchy_votes[best_hierarchy]
        hierarchy_info = self.hierarchy_patterns[best_hierarchy]
        
        return {
            'hierarchy_type': best_hierarchy,
            'hierarchy_level': hierarchy_info['level'],
            'hierarchy_confidence': best_info['total_confidence'] / best_info['count'],
            'classification_method': 'combined',
            'classification_methods': best_info['methods']
        }
    
    def _extract_parent_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """?ìœ„ë²?ì¶”ì¶œ"""
        parent_laws = []
        
        # ë²•ë¥ ëª?ì¶”ì¶œ ?¨í„´
        law_patterns = [
            r'??[^??+ë²???,  # ?°ì˜´?œë¡œ ?˜ëŸ¬?¸ì¸ ë²•ë¥ ëª?
            r'([ê°€-??+ë²?\s*??d+ì¡?,  # ë²•ë¥ ëª?+ ì¡°ë¬¸
            r'([ê°€-??+ë²?\s*ë°?,  # ë²•ë¥ ëª?+ ë°?
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                # ?ìœ„ë²??¬ë? ?ë‹¨
                if self._is_parent_law(law_name):
                    parent_laws.append({
                        'law_name': law_name,
                        'reference_type': 'parent_law',
                        'hierarchy_level': 2,  # ë²•ë¥ ?€ ?¼ë°˜?ìœ¼ë¡??ˆë²¨ 2
                        'extraction_method': 'pattern_matching'
                    })
        
        # ì¤‘ë³µ ?œê±°
        unique_parent_laws = []
        seen_names = set()
        
        for parent_law in parent_laws:
            if parent_law['law_name'] not in seen_names:
                unique_parent_laws.append(parent_law)
                seen_names.add(parent_law['law_name'])
        
        return unique_parent_laws
    
    def _extract_subordinate_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """?˜ìœ„ë²?ì¶”ì¶œ"""
        subordinate_laws = []
        
        # ?˜ìœ„ë²??¨í„´
        subordinate_patterns = [
            r'([ê°€-??+?œí–‰??',  # ?œí–‰??
            r'([ê°€-??+?œí–‰ê·œì¹™)',  # ?œí–‰ê·œì¹™
            r'([ê°€-??+ë¶€??',  # ë¶€??
        ]
        
        for pattern in subordinate_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                subordinate_laws.append({
                    'law_name': law_name,
                    'reference_type': 'subordinate_law',
                    'hierarchy_level': 3 if '?œí–‰?? in law_name else 5,
                    'extraction_method': 'pattern_matching'
                })
        
        return subordinate_laws
    
    def _extract_related_hierarchy(self, law_content: str) -> List[Dict[str, Any]]:
        """ê´€???„ê³„ ë²•ë¥  ì¶”ì¶œ"""
        related_hierarchy = []
        
        # ê´€??ë²•ë¥  ?¨í„´
        related_patterns = [
            r'([ê°€-??+ë²?\s*ë°?s*([ê°€-??+ë²?',  # Aë²?ë°?Bë²?
            r'([ê°€-??+ë²?\s*?ëŠ”\s*([ê°€-??+ë²?',  # Aë²??ëŠ” Bë²?
            r'([ê°€-??+ë²?\s*,\s*([ê°€-??+ë²?',  # Aë²? Bë²?
        ]
        
        for pattern in related_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                if isinstance(match, tuple):
                    for law_name in match:
                        if law_name:
                            related_hierarchy.append({
                                'law_name': law_name,
                                'reference_type': 'related_hierarchy',
                                'hierarchy_level': 2,  # ê´€??ë²•ë¥ ?€ ?¼ë°˜?ìœ¼ë¡?ë²•ë¥  ?ˆë²¨
                                'extraction_method': 'pattern_matching'
                            })
        
        return related_hierarchy
    
    def _is_parent_law(self, law_name: str) -> bool:
        """?ìœ„ë²??¬ë? ?ë‹¨"""
        # ?œí–‰?? ?œí–‰ê·œì¹™, ë¶€???±ì? ?˜ìœ„ë²?
        subordinate_indicators = ['?œí–‰??, '?œí–‰ê·œì¹™', 'ë¶€??, '?€?µë ¹??, 'ì´ë¦¬??]
        
        for indicator in subordinate_indicators:
            if indicator in law_name:
                return False
        
        # ë²•ë¥ ëª…ì´ 'ë²??¼ë¡œ ?ë‚˜??ê²½ìš° ?ìœ„ë²•ìœ¼ë¡?ê°„ì£¼
        return law_name.endswith('ë²?)
    
    def get_hierarchy_info(self, hierarchy_type: str) -> Dict[str, Any]:
        """?„ê³„ ?•ë³´ ë°˜í™˜"""
        if hierarchy_type not in self.hierarchy_patterns:
            return {}
        
        patterns = self.hierarchy_patterns[hierarchy_type]
        
        return {
            'hierarchy_type': hierarchy_type,
            'hierarchy_level': patterns['level'],
            'keywords': patterns['keywords'],
            'patterns': patterns['patterns'],
            'weight': patterns['weight']
        }
    
    def get_supported_hierarchies(self) -> List[str]:
        """ì§€?ë˜???„ê³„ ëª©ë¡ ë°˜í™˜"""
        return list(self.hierarchy_patterns.keys())
    
    def validate_hierarchy_classification(self, law_data: Dict[str, Any], 
                                        hierarchy_type: str) -> Dict[str, Any]:
        """?„ê³„ ë¶„ë¥˜ ê²€ì¦?""
        classification_result = self.classify_law_hierarchy(law_data)
        
        return {
            'is_valid': classification_result['hierarchy_type'] == hierarchy_type,
            'confidence': classification_result['hierarchy_confidence'],
            'actual_type': classification_result['hierarchy_type'],
            'expected_type': hierarchy_type,
            'validation_timestamp': datetime.now().isoformat()
        }
