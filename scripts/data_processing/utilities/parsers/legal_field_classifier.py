"""
Legal Field Classifier for Korean Law Data

This module classifies Korean law data based on legal fields
such as Constitutional Law, Civil Law, Criminal Law, Commercial Law, etc.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LegalFieldClassifier:
    """Î≤ïÎ•† Î∂ÑÏïº Î∂ÑÎ•òÍ∏?""
    
    def __init__(self):
        """Initialize field classifier with Korean legal field patterns"""
        self.field_patterns = {
            'constitutional_law': {
                'keywords': ['?åÎ≤ï', 'Í∏∞Î≥∏Í∂?, 'Íµ??Í∏∞Í?', '?åÎ≤ï?¨Ìåê??, '?ÑÌóå', '?åÎ≤ï?¨Ìåê', 'Íµ????Í∂åÎ¶¨', 'Íµ?????òÎ¨¥'],
                'patterns': [r'?åÎ≤ï\s*??d+Ï°?, r'Í∏∞Î≥∏Í∂?, r'Íµ??Í∏∞Í?', r'?åÎ≤ï?¨Ìåê??, r'?ÑÌóåÎ≤ïÎ•†?¨Ìåê'],
                'subfields': ['constitutional_rights', 'state_organization', 'constitutional_review'],
                'weight': 1.0
            },
            'civil_law': {
                'keywords': ['ÎØºÎ≤ï', '?¨ÏÇ∞Í∂?, 'Í≥ÑÏïΩ', 'Í∞ÄÏ°?, '?ÅÏÜç', 'Ï±ÑÍ∂å', 'Î¨ºÍ∂å', '?êÌï¥Î∞∞ÏÉÅ', 'Î∂àÎ≤ï?âÏúÑ', 'Ï±ÑÎ¨¥'],
                'patterns': [r'ÎØºÎ≤ï\s*??d+Ï°?, r'Í≥ÑÏïΩ', r'?¨ÏÇ∞Í∂?, r'Í∞ÄÏ°±Í?Í≥?, r'?ÅÏÜç', r'Ï±ÑÍ∂å', r'Î¨ºÍ∂å'],
                'subfields': ['property_law', 'contract_law', 'family_law', 'inheritance_law', 'tort_law'],
                'weight': 1.0
            },
            'criminal_law': {
                'keywords': ['?ïÎ≤ï', 'Î≤îÏ£Ñ', '?ïÎ≤å', 'Î≤åÍ∏à', 'ÏßïÏó≠', 'Í∏àÍ≥†', '?¨Ìòï', '?êÏú†??, '?ïÏÇ¨Ï≤òÎ≤å'],
                'patterns': [r'?ïÎ≤ï\s*??d+Ï°?, r'Î≤îÏ£Ñ', r'?ïÎ≤å', r'Î≤åÍ∏à', r'ÏßïÏó≠', r'Í∏àÍ≥†'],
                'subfields': ['general_criminal_law', 'special_criminal_law', 'criminal_procedure'],
                'weight': 1.0
            },
            'commercial_law': {
                'keywords': ['?ÅÎ≤ï', '?åÏÇ¨', '?ÅÍ±∞??, '?¥Ïùå', '?òÌëú', 'Î≥¥Ìóò', '?¥ÏÉÅ', '??≥µ', '?ÅÌñâ??],
                'patterns': [r'?ÅÎ≤ï\s*??d+Ï°?, r'?åÏÇ¨Î≤?, r'?ÅÍ±∞??, r'?¥ÏùåÎ≤?, r'?òÌëúÎ≤?, r'Î≥¥ÌóòÎ≤?],
                'subfields': ['company_law', 'commercial_transaction', 'insurance_law', 'maritime_law', 'aviation_law'],
                'weight': 1.0
            },
            'administrative_law': {
                'keywords': ['?âÏ†ïÎ≤?, '?âÏ†ïÏ≤òÎ∂Ñ', '?àÍ?', '?πÏù∏', '?†Í≥†', '?âÏ†ï?àÏ∞®', '?âÏ†ï?åÏÜ°', '?âÏ†ï?¨Ìåê'],
                'patterns': [r'?âÏ†ïÎ≤?, r'?àÍ?', r'?πÏù∏', r'?†Í≥†', r'?âÏ†ï?àÏ∞®', r'?âÏ†ï?åÏÜ°'],
                'subfields': ['administrative_procedure', 'administrative_disposition', 'administrative_litigation'],
                'weight': 1.0
            },
            'labor_law': {
                'keywords': ['?∏ÎèôÎ≤?, 'Í∑ºÎ°ú', '?ÑÍ∏à', 'Í∑ºÎ°ú?úÍ∞Ñ', '?∞ÏóÖ?àÏ†Ñ', 'Í≥†Ïö©', 'Í∑ºÎ°úÍ∏∞Ï?', '?∞ÏóÖ?¨Ìï¥'],
                'patterns': [r'?∏ÎèôÎ≤?, r'Í∑ºÎ°úÍ∏∞Ï?Î≤?, r'?ÑÍ∏à', r'Í∑ºÎ°ú?úÍ∞Ñ', r'?∞ÏóÖ?àÏ†ÑÎ≥¥Í±¥Î≤?],
                'subfields': ['labor_standards', 'industrial_safety', 'employment_security', 'workers_compensation'],
                'weight': 1.0
            },
            'economic_law': {
                'keywords': ['Í≤ΩÏ†úÎ≤?, 'Í≥µÏ†ïÍ±∞Îûò', '?ÖÏ†êÍ∑úÏ†ú', '?åÎπÑ??, 'Í∏àÏúµ', 'Ï¶ùÍ∂å', '?êÎ≥∏?úÏû•', 'Í≤ΩÏüÅÎ≤?],
                'patterns': [r'Í≥µÏ†ïÍ±∞ÎûòÎ≤?, r'?ÖÏ†êÍ∑úÏ†ú', r'?åÎπÑ?êÎ≥¥??, r'Í∏àÏúµÎ≤?, r'Ï¶ùÍ∂åÎ≤?],
                'subfields': ['fair_trade', 'consumer_protection', 'financial_law', 'securities_law', 'competition_law'],
                'weight': 1.0
            },
            'procedural_law': {
                'keywords': ['?åÏÜ°Î≤?, 'ÎØºÏÇ¨?åÏÜ°', '?ïÏÇ¨?åÏÜ°', '?âÏ†ï?åÏÜ°', '?¨Ìåê', '?åÏÜ°?àÏ∞®', 'Î≤ïÏõê', '?êÏÇ¨'],
                'patterns': [r'?åÏÜ°Î≤?, r'ÎØºÏÇ¨?åÏÜ°Î≤?, r'?ïÏÇ¨?åÏÜ°Î≤?, r'?âÏ†ï?åÏÜ°Î≤?, r'?¨Ìåê'],
                'subfields': ['civil_procedure', 'criminal_procedure', 'administrative_procedure'],
                'weight': 1.0
            },
            'tax_law': {
                'keywords': ['?∏Î≤ï', '?åÎìù??, 'Î≤ïÏù∏??, 'Î∂ÄÍ∞ÄÍ∞ÄÏπòÏÑ∏', '?ÅÏÜç??, 'Ï¶ùÏó¨??, 'Íµ?Ñ∏', 'ÏßÄÎ∞©ÏÑ∏'],
                'patterns': [r'?∏Î≤ï', r'?åÎìù?∏Î≤ï', r'Î≤ïÏù∏?∏Î≤ï', r'Î∂ÄÍ∞ÄÍ∞ÄÏπòÏÑ∏Î≤?, r'?ÅÏÜç?∏Î≤ï'],
                'subfields': ['income_tax', 'corporate_tax', 'value_added_tax', 'inheritance_tax'],
                'weight': 1.0
            },
            'environmental_law': {
                'keywords': ['?òÍ≤ΩÎ≤?, '?ÄÍ∏∞ÌôòÍ≤?, '?òÏßà?òÍ≤Ω', '?êÍ∏∞Î¨?, '?òÍ≤Ω?ÅÌñ•?âÍ?', '?òÍ≤Ω?§Ïóº'],
                'patterns': [r'?òÍ≤ΩÎ≤?, r'?ÄÍ∏∞ÌôòÍ≤ΩÎ≥¥?ÑÎ≤ï', r'?òÏßà?òÍ≤ΩÎ≥¥Ï†ÑÎ≤?, r'?êÍ∏∞Î¨ºÍ?Î¶¨Î≤ï'],
                'subfields': ['air_quality', 'water_quality', 'waste_management', 'environmental_assessment'],
                'weight': 1.0
            },
            'intellectual_property_law': {
                'keywords': ['ÏßÄ?ÅÏû¨?∞Í∂å', '?πÌóà', '?ÅÌëú', '?Ä?ëÍ∂å', '?îÏûê??, '?ÅÏóÖÎπÑÎ?', '?πÌóàÎ≤?],
                'patterns': [r'?πÌóàÎ≤?, r'?ÅÌëúÎ≤?, r'?Ä?ëÍ∂åÎ≤?, r'?îÏûê?∏Î≥¥?∏Î≤ï', r'ÏßÄ?ÅÏû¨?∞Í∂å'],
                'subfields': ['patent_law', 'trademark_law', 'copyright_law', 'design_law'],
                'weight': 1.0
            },
            'health_law': {
                'keywords': ['?òÎ£åÎ≤?, 'Î≥¥Í±¥Î≤?, '?òÎ£åÍ∏∞Í?', '?òÎ£å??, '?òÎ£å?âÏúÑ', 'Î≥¥Í±¥Î≥µÏ?'],
                'patterns': [r'?òÎ£åÎ≤?, r'Î≥¥Í±¥Î≤?, r'?òÎ£åÍ∏∞Í?', r'?òÎ£å??, r'Î≥¥Í±¥Î≥µÏ?'],
                'subfields': ['medical_law', 'public_health', 'healthcare_institutions', 'medical_practitioners'],
                'weight': 1.0
            }
        }
        
        # Î≤ïÎ•†Î™?Í∏∞Î∞ò Î∂ÑÏïº Îß§Ìïë
        self.law_name_field_mapping = {
            'ÎØºÎ≤ï': 'civil_law',
            '?ïÎ≤ï': 'criminal_law',
            '?ÅÎ≤ï': 'commercial_law',
            '?åÎ≤ï': 'constitutional_law',
            '?âÏ†ïÎ≤?: 'administrative_law',
            '?∏ÎèôÎ≤?: 'labor_law',
            'Í∑ºÎ°úÍ∏∞Ï?Î≤?: 'labor_law',
            'Í≥µÏ†ïÍ±∞ÎûòÎ≤?: 'economic_law',
            '?åÎπÑ?êÎ≥¥?∏Î≤ï': 'economic_law',
            '?åÏÜ°Î≤?: 'procedural_law',
            'ÎØºÏÇ¨?åÏÜ°Î≤?: 'procedural_law',
            '?ïÏÇ¨?åÏÜ°Î≤?: 'procedural_law',
            '?âÏ†ï?åÏÜ°Î≤?: 'procedural_law',
            '?∏Î≤ï': 'tax_law',
            '?åÎìù?∏Î≤ï': 'tax_law',
            'Î≤ïÏù∏?∏Î≤ï': 'tax_law',
            '?òÍ≤ΩÎ≤?: 'environmental_law',
            '?πÌóàÎ≤?: 'intellectual_property_law',
            '?ÅÌëúÎ≤?: 'intellectual_property_law',
            '?Ä?ëÍ∂åÎ≤?: 'intellectual_property_law',
            '?òÎ£åÎ≤?: 'health_law',
            'Î≥¥Í±¥Î≤?: 'health_law'
        }
    
    def classify_legal_field(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Î≤ïÎ•† Î∂ÑÏïº Î∂ÑÎ•ò
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Field classification results
        """
        try:
            law_name = law_data.get('law_name', '')
            law_content = law_data.get('law_content', '')
            
            field_info = {
                'primary_field': 'unknown',
                'secondary_fields': [],
                'subfields': [],
                'field_confidence': 0.0,
                'field_keywords': [],
                'related_laws': [],
                'classification_method': 'unknown',
                'classification_timestamp': datetime.now().isoformat()
            }
            
            # Í∞?Î∞©Î≤ïÎ≥?Î∂ÑÎ•ò ?òÌñâ
            name_classification = self._classify_by_name(law_name)
            content_classification = self._classify_by_content(law_content)
            
            # Ï¢ÖÌï© Î∂ÑÎ•ò
            final_classification = self._combine_classifications(
                name_classification, content_classification
            )
            
            field_info.update(final_classification)
            
            # Í¥Ä??Î≤ïÎ•† Î∂ÑÏÑù
            field_info['related_laws'] = self._extract_related_laws(law_content)
            
            return field_info
            
        except Exception as e:
            logger.error(f"Error classifying legal field: {e}")
            return {
                'primary_field': 'unknown',
                'field_confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_by_name(self, law_name: str) -> Dict[str, Any]:
        """Î≤ïÎ•†Î™?Í∏∞Î∞ò Î∂ÑÎ•ò"""
        if not law_name:
            return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'name'}
        
        # ÏßÅÏ†ë Îß§Ìïë ?ïÏù∏
        for law_keyword, field in self.law_name_field_mapping.items():
            if law_keyword in law_name:
                field_info = self.field_patterns.get(field, {})
                return {
                    'primary_field': field,
                    'confidence': 0.95,
                    'method': 'name_mapping',
                    'matched_keyword': law_keyword,
                    'subfields': field_info.get('subfields', [])
                }
        
        # ?®ÌÑ¥ Í∏∞Î∞ò Î∂ÑÎ•ò
        for field, patterns in self.field_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in law_name:
                    return {
                        'primary_field': field,
                        'confidence': 0.8,
                        'method': 'name_keyword',
                        'matched_keyword': keyword,
                        'subfields': patterns['subfields']
                    }
        
        return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'name'}
    
    def _classify_by_content(self, law_content: str) -> Dict[str, Any]:
        """?¥Ïö© Í∏∞Î∞ò Î∂ÑÎ•ò"""
        if not law_content:
            return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'content'}
        
        field_scores = {}
        field_keywords = {}
        
        for field, patterns in self.field_patterns.items():
            score = 0
            matched_keywords = []
            
            # ?§Ïõå??Îß§Ïπ≠
            for keyword in patterns['keywords']:
                count = law_content.count(keyword)
                score += count
                if count > 0:
                    matched_keywords.append(f"{keyword}({count})")
            
            # ?®ÌÑ¥ Îß§Ïπ≠
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, law_content)
                score += len(matches) * 2  # ?®ÌÑ¥ Îß§Ïπò?????íÏ? Í∞ÄÏ§ëÏπò
                if matches:
                    matched_keywords.append(f"pattern:{pattern}({len(matches)})")
            
            if score > 0:
                field_scores[field] = score
                field_keywords[field] = matched_keywords
        
        # Í∞Ä???íÏ? ?êÏàò??Î∂ÑÏïº ?†ÌÉù
        if field_scores:
            best_field = max(field_scores, key=field_scores.get)
            best_score = field_scores[best_field]
            
            return {
                'primary_field': best_field,
                'confidence': min(0.7, best_score / 10),
                'method': 'content_analysis',
                'field_keywords': field_keywords[best_field],
                'subfields': self.field_patterns[best_field]['subfields']
            }
        
        return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'content'}
    
    def _combine_classifications(self, name_class: Dict, content_class: Dict) -> Dict[str, Any]:
        """Î∂ÑÎ•ò Í≤∞Í≥º Ï¢ÖÌï©"""
        classifications = [name_class, content_class]
        
        # ?†Ìö®??Î∂ÑÎ•òÎß??ÑÌÑ∞Îß?
        valid_classifications = [c for c in classifications if c['primary_field'] != 'unknown']
        
        if not valid_classifications:
            return {
                'primary_field': 'unknown',
                'field_confidence': 0.0,
                'classification_method': 'combined'
            }
        
        # Í∞ÄÏ§??âÍ∑†?ºÎ°ú ÏµúÏ¢Ö Î∂ÑÏïº Í≤∞Ï†ï
        field_votes = {}
        
        for classification in valid_classifications:
            field = classification['primary_field']
            confidence = classification['confidence']
            
            if field not in field_votes:
                field_votes[field] = {
                    'total_confidence': 0.0,
                    'count': 0,
                    'methods': [],
                    'keywords': []
                }
            
            field_votes[field]['total_confidence'] += confidence
            field_votes[field]['count'] += 1
            field_votes[field]['methods'].append(classification.get('method', 'unknown'))
            
            # ?§Ïõå???òÏßë
            if 'field_keywords' in classification:
                field_votes[field]['keywords'].extend(classification['field_keywords'])
        
        # Í∞Ä???íÏ? ?âÍ∑† ?†Î¢∞?ÑÏùò Î∂ÑÏïº ?†ÌÉù
        best_field = max(field_votes, 
                        key=lambda x: field_votes[x]['total_confidence'] / field_votes[x]['count'])
        
        best_info = field_votes[best_field]
        field_info = self.field_patterns[best_field]
        
        # Î≥¥Ï°∞ Î∂ÑÏïº??Í≤∞Ï†ï
        secondary_fields = []
        sorted_fields = sorted(field_votes.items(), 
                             key=lambda x: x[1]['total_confidence'] / x[1]['count'], 
                             reverse=True)
        
        for field, info in sorted_fields[1:3]:  # ?ÅÏúÑ 2Í∞?Î≥¥Ï°∞ Î∂ÑÏïº
            if info['total_confidence'] / info['count'] > 0.3:  # ÏµúÏÜå ?†Î¢∞???ÑÍ≥ÑÍ∞?
                secondary_fields.append(field)
        
        return {
            'primary_field': best_field,
            'secondary_fields': secondary_fields,
            'field_confidence': best_info['total_confidence'] / best_info['count'],
            'classification_method': 'combined',
            'classification_methods': best_info['methods'],
            'field_keywords': best_info['keywords'],
            'subfields': field_info['subfields']
        }
    
    def _extract_related_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """Í¥Ä??Î≤ïÎ•† Ï∂îÏ∂ú"""
        related_laws = []
        
        # Í¥Ä??Î≤ïÎ•† ?®ÌÑ¥
        related_patterns = [
            r'??[^??+Î≤???,  # ?∞Ïò¥?úÎ°ú ?òÎü¨?∏Ïù∏ Î≤ïÎ•†Î™?
            r'([Í∞Ä-??+Î≤?\s*??d+Ï°?,  # Î≤ïÎ•†Î™?+ Ï°∞Î¨∏
            r'([Í∞Ä-??+Î≤?\s*Î∞?,  # Î≤ïÎ•†Î™?+ Î∞?
            r'([Í∞Ä-??+Î≤?\s*?êÎäî',  # Î≤ïÎ•†Î™?+ ?êÎäî
        ]
        
        for pattern in related_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                # Î∂ÑÏïº Îß§Ìïë ?ïÏù∏
                field = self.law_name_field_mapping.get(law_name, 'unknown')
                
                related_laws.append({
                    'law_name': law_name,
                    'field': field,
                    'extraction_method': 'pattern_matching'
                })
        
        # Ï§ëÎ≥µ ?úÍ±∞
        unique_related_laws = []
        seen_names = set()
        
        for related_law in related_laws:
            if related_law['law_name'] not in seen_names:
                unique_related_laws.append(related_law)
                seen_names.add(related_law['law_name'])
        
        return unique_related_laws
    
    def get_field_info(self, field: str) -> Dict[str, Any]:
        """Î∂ÑÏïº ?ïÎ≥¥ Î∞òÌôò"""
        if field not in self.field_patterns:
            return {}
        
        patterns = self.field_patterns[field]
        
        return {
            'field': field,
            'keywords': patterns['keywords'],
            'patterns': patterns['patterns'],
            'subfields': patterns['subfields'],
            'weight': patterns['weight']
        }
    
    def get_supported_fields(self) -> List[str]:
        """ÏßÄ?êÎêò??Î∂ÑÏïº Î™©Î°ù Î∞òÌôò"""
        return list(self.field_patterns.keys())
    
    def get_subfields(self, field: str) -> List[str]:
        """?πÏ†ï Î∂ÑÏïº???òÏúÑ Î∂ÑÏïº Î™©Î°ù Î∞òÌôò"""
        if field not in self.field_patterns:
            return []
        
        return self.field_patterns[field]['subfields']
    
    def validate_field_classification(self, law_data: Dict[str, Any], 
                                    expected_field: str) -> Dict[str, Any]:
        """Î∂ÑÏïº Î∂ÑÎ•ò Í≤ÄÏ¶?""
        classification_result = self.classify_legal_field(law_data)
        
        return {
            'is_valid': classification_result['primary_field'] == expected_field,
            'confidence': classification_result['field_confidence'],
            'actual_field': classification_result['primary_field'],
            'expected_field': expected_field,
            'secondary_fields': classification_result.get('secondary_fields', []),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_field_statistics(self, law_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Î∂ÑÏïºÎ≥??µÍ≥Ñ ?ùÏÑ±"""
        field_counts = {}
        field_confidences = {}
        
        for law_data in law_data_list:
            classification = self.classify_legal_field(law_data)
            field = classification['primary_field']
            confidence = classification['field_confidence']
            
            if field not in field_counts:
                field_counts[field] = 0
                field_confidences[field] = []
            
            field_counts[field] += 1
            field_confidences[field].append(confidence)
        
        # ?âÍ∑† ?†Î¢∞??Í≥ÑÏÇ∞
        field_stats = {}
        for field, counts in field_counts.items():
            confidences = field_confidences[field]
            field_stats[field] = {
                'count': counts,
                'percentage': counts / len(law_data_list) * 100,
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0
            }
        
        return {
            'total_laws': len(law_data_list),
            'field_statistics': field_stats,
            'generated_at': datetime.now().isoformat()
        }
