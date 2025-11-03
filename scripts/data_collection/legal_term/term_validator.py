# -*- coding: utf-8 -*-
"""
?©ì–´ ê²€ì¦ê¸°

?˜ì§‘??ë²•ë¥  ?©ì–´???ˆì§ˆ??ê²€ì¦í•˜ê³?ê°œì„ ?¬í•­???œì•ˆ?©ë‹ˆ??
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class TermValidator:
    """?©ì–´ ê²€ì¦ê¸° ?´ë˜??""
    
    def __init__(self, dictionary=None):
        """ê²€ì¦ê¸° ì´ˆê¸°??""
        self.dictionary = dictionary
        self.validation_rules = self._load_validation_rules()
        
        logger.info("TermValidator initialized")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """ê²€ì¦?ê·œì¹™ ë¡œë“œ"""
        return {
            'term_name': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[ê°€-?£a-zA-Z0-9\s\-_]+$',
                'required': True
            },
            'definition': {
                'min_length': 10,
                'max_length': 1000,
                'required': True
            },
            'category': {
                'allowed_values': [
                    'ë¯¼ì‚¬ë²?, '?•ì‚¬ë²?, '?ì‚¬ë²?, '?¸ë™ë²?, '?‰ì •ë²?,
                    '?˜ê²½ë²?, '?Œë¹„?ë²•', 'ì§€?ì¬?°ê¶Œë²?, 'ê¸ˆìœµë²?, 'ê¸°í?'
                ],
                'required': True
            },
            'frequency': {
                'min_value': 0,
                'max_value': 10000,
                'required': True
            }
        }
    
    def validate_all_terms(self, dictionary) -> Dict[str, Any]:
        """ëª¨ë“  ?©ì–´ ê²€ì¦?""
        if not dictionary:
            logger.error("?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??")
            return {'is_valid': False, 'issues': ['?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??']}
        
        self.dictionary = dictionary
        
        logger.info(f"?©ì–´ ê²€ì¦??œì‘ - {len(dictionary.terms)}ê°??©ì–´")
        
        validation_results = {
            'total_terms': len(dictionary.terms),
            'valid_terms': 0,
            'invalid_terms': 0,
            'issues': [],
            'term_issues': {},
            'statistics': {},
            'recommendations': []
        }
        
        # ê°œë³„ ?©ì–´ ê²€ì¦?
        for term_id, term_data in dictionary.terms.items():
            term_validation = self.validate_single_term(term_data)
            
            if term_validation['is_valid']:
                validation_results['valid_terms'] += 1
            else:
                validation_results['invalid_terms'] += 1
                validation_results['term_issues'][term_id] = term_validation['issues']
                validation_results['issues'].extend(term_validation['issues'])
        
        # ?„ì²´ ?µê³„ ?ì„±
        validation_results['statistics'] = self._generate_validation_statistics(dictionary)
        
        # ê°œì„  ê¶Œì¥?¬í•­ ?ì„±
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        validation_results['is_valid'] = validation_results['invalid_terms'] == 0
        
        logger.info(f"?©ì–´ ê²€ì¦??„ë£Œ - {validation_results['valid_terms']}ê°?? íš¨, {validation_results['invalid_terms']}ê°?ë¬´íš¨")
        
        return validation_results
    
    def validate_single_term(self, term_data: Dict[str, Any]) -> Dict[str, Any]:
        """ê°œë³„ ?©ì–´ ê²€ì¦?""
        issues = []
        
        # ?©ì–´ëª?ê²€ì¦?
        term_name = term_data.get('term_name', '')
        if not term_name:
            issues.append("?©ì–´ëª…ì´ ?†ìŠµ?ˆë‹¤.")
        else:
            if len(term_name) < self.validation_rules['term_name']['min_length']:
                issues.append(f"?©ì–´ëª…ì´ ?ˆë¬´ ì§§ìŠµ?ˆë‹¤: {len(term_name)}??(ìµœì†Œ {self.validation_rules['term_name']['min_length']}??")
            
            if len(term_name) > self.validation_rules['term_name']['max_length']:
                issues.append(f"?©ì–´ëª…ì´ ?ˆë¬´ ê¹ë‹ˆ?? {len(term_name)}??(ìµœë? {self.validation_rules['term_name']['max_length']}??")
            
            if not re.match(self.validation_rules['term_name']['pattern'], term_name):
                issues.append(f"?©ì–´ëª…ì— ? íš¨?˜ì? ?Šì? ë¬¸ìê°€ ?¬í•¨?˜ì–´ ?ˆìŠµ?ˆë‹¤: {term_name}")
        
        # ?•ì˜ ê²€ì¦?
        definition = term_data.get('definition', '')
        if not definition:
            issues.append("?•ì˜ê°€ ?†ìŠµ?ˆë‹¤.")
        else:
            if len(definition) < self.validation_rules['definition']['min_length']:
                issues.append(f"?•ì˜ê°€ ?ˆë¬´ ì§§ìŠµ?ˆë‹¤: {len(definition)}??(ìµœì†Œ {self.validation_rules['definition']['min_length']}??")
            
            if len(definition) > self.validation_rules['definition']['max_length']:
                issues.append(f"?•ì˜ê°€ ?ˆë¬´ ê¹ë‹ˆ?? {len(definition)}??(ìµœë? {self.validation_rules['definition']['max_length']}??")
        
        # ì¹´í…Œê³ ë¦¬ ê²€ì¦?
        category = term_data.get('category', '')
        if not category:
            issues.append("ì¹´í…Œê³ ë¦¬ê°€ ?†ìŠµ?ˆë‹¤.")
        elif category not in self.validation_rules['category']['allowed_values']:
            issues.append(f"? íš¨?˜ì? ?Šì? ì¹´í…Œê³ ë¦¬?…ë‹ˆ?? {category}")
        
        # ë¹ˆë„ ê²€ì¦?
        frequency = term_data.get('frequency', 0)
        if not isinstance(frequency, (int, float)):
            issues.append("ë¹ˆë„ê°€ ?«ìê°€ ?„ë‹™?ˆë‹¤.")
        elif frequency < self.validation_rules['frequency']['min_value']:
            issues.append(f"ë¹ˆë„ê°€ ?ˆë¬´ ??Šµ?ˆë‹¤: {frequency} (ìµœì†Œ {self.validation_rules['frequency']['min_value']})")
        elif frequency > self.validation_rules['frequency']['max_value']:
            issues.append(f"ë¹ˆë„ê°€ ?ˆë¬´ ?’ìŠµ?ˆë‹¤: {frequency} (ìµœë? {self.validation_rules['frequency']['max_value']})")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'term_name': term_name,
            'category': category
        }
    
    def _generate_validation_statistics(self, dictionary) -> Dict[str, Any]:
        """ê²€ì¦??µê³„ ?ì„±"""
        stats = {
            'category_distribution': defaultdict(int),
            'frequency_distribution': defaultdict(int),
            'definition_length_distribution': defaultdict(int),
            'term_name_length_distribution': defaultdict(int),
            'duplicate_terms': [],
            'empty_definitions': 0,
            'low_frequency_terms': 0
        }
        
        term_names = []
        
        for term_data in dictionary.terms.values():
            # ì¹´í…Œê³ ë¦¬ ë¶„í¬
            category = term_data.get('category', 'ê¸°í?')
            stats['category_distribution'][category] += 1
            
            # ë¹ˆë„ ë¶„í¬
            frequency = term_data.get('frequency', 0)
            freq_range = f"{(frequency // 10) * 10}-{(frequency // 10) * 10 + 9}"
            stats['frequency_distribution'][freq_range] += 1
            
            if frequency < 5:
                stats['low_frequency_terms'] += 1
            
            # ?•ì˜ ê¸¸ì´ ë¶„í¬
            definition = term_data.get('definition', '')
            if not definition:
                stats['empty_definitions'] += 1
            else:
                def_len_range = f"{(len(definition) // 50) * 50}-{(len(definition) // 50) * 50 + 49}"
                stats['definition_length_distribution'][def_len_range] += 1
            
            # ?©ì–´ëª?ê¸¸ì´ ë¶„í¬
            term_name = term_data.get('term_name', '')
            if term_name:
                name_len_range = f"{(len(term_name) // 5) * 5}-{(len(term_name) // 5) * 5 + 4}"
                stats['term_name_length_distribution'][name_len_range] += 1
                term_names.append(term_name)
        
        # ì¤‘ë³µ ?©ì–´ ì°¾ê¸°
        term_name_counts = Counter(term_names)
        stats['duplicate_terms'] = [name for name, count in term_name_counts.items() if count > 1]
        
        # ?•ì…”?ˆë¦¬ë¡?ë³€??
        for key in ['category_distribution', 'frequency_distribution', 'definition_length_distribution', 'term_name_length_distribution']:
            stats[key] = dict(stats[key])
        
        return stats
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """ê°œì„  ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []
        
        # ì¤‘ë³µ ?©ì–´ ?•ë¦¬ ê¶Œì¥
        if validation_results['statistics']['duplicate_terms']:
            recommendations.append(f"ì¤‘ë³µ???©ì–´ëª?{len(validation_results['statistics']['duplicate_terms'])}ê°œë? ?•ë¦¬?˜ì„¸?? {validation_results['statistics']['duplicate_terms'][:5]}")
        
        # ë¹??•ì˜ ?•ë¦¬ ê¶Œì¥
        if validation_results['statistics']['empty_definitions'] > 0:
            recommendations.append(f"?•ì˜ê°€ ?†ëŠ” ?©ì–´ {validation_results['statistics']['empty_definitions']}ê°œë? ?•ë¦¬?˜ì„¸??")
        
        # ??? ë¹ˆë„ ?©ì–´ ê²€??ê¶Œì¥
        if validation_results['statistics']['low_frequency_terms'] > 0:
            recommendations.append(f"ë¹ˆë„ê°€ ??? ?©ì–´ {validation_results['statistics']['low_frequency_terms']}ê°œë? ê²€? í•˜?¸ìš”.")
        
        # ì¹´í…Œê³ ë¦¬ ë¶ˆê· ??ê¶Œì¥
        category_dist = validation_results['statistics']['category_distribution']
        if category_dist:
            max_category = max(category_dist, key=category_dist.get)
            min_category = min(category_dist, key=category_dist.get)
            if category_dist[max_category] > category_dist[min_category] * 3:
                recommendations.append(f"ì¹´í…Œê³ ë¦¬ ë¶„í¬ê°€ ë¶ˆê· ?•í•©?ˆë‹¤. '{min_category}' ì¹´í…Œê³ ë¦¬ ?©ì–´ë¥?ì¶”ê??˜ì„¸??")
        
        return recommendations
    
    def validate_synonym_groups(self, dictionary) -> Dict[str, Any]:
        """?™ì˜??ê·¸ë£¹ ê²€ì¦?""
        if not dictionary:
            return {'is_valid': False, 'issues': ['?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??']}
        
        issues = []
        group_stats = {
            'total_groups': len(dictionary.synonym_groups),
            'groups_with_issues': 0,
            'empty_groups': 0,
            'invalid_confidence': 0
        }
        
        for group_id, group_data in dictionary.synonym_groups.items():
            group_issues = []
            
            # ?œì? ?©ì–´ ê²€ì¦?
            standard_term = group_data.get('standard_term', '')
            if not standard_term:
                group_issues.append("?œì? ?©ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
            elif not dictionary.get_term(standard_term):
                group_issues.append(f"?œì? ?©ì–´ '{standard_term}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
            
            # ë³€?•ì–´ ê²€ì¦?
            variants = group_data.get('variants', [])
            if not variants:
                group_issues.append("ë³€?•ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
                group_stats['empty_groups'] += 1
            else:
                for variant in variants:
                    if not dictionary.get_term(variant):
                        group_issues.append(f"ë³€?•ì–´ '{variant}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
            
            # ? ë¢°??ê²€ì¦?
            confidence = group_data.get('confidence', 0)
            if not 0 <= confidence <= 1:
                group_issues.append(f"? ë¢°?„ê? ? íš¨?˜ì? ?ŠìŠµ?ˆë‹¤: {confidence}")
                group_stats['invalid_confidence'] += 1
            
            if group_issues:
                issues.extend([f"{group_id}: {issue}" for issue in group_issues])
                group_stats['groups_with_issues'] += 1
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'statistics': group_stats
        }
    
    def suggest_term_improvements(self, term_data: Dict[str, Any]) -> List[str]:
        """ê°œë³„ ?©ì–´ ê°œì„  ?œì•ˆ"""
        suggestions = []
        
        term_name = term_data.get('term_name', '')
        definition = term_data.get('definition', '')
        category = term_data.get('category', '')
        
        # ?©ì–´ëª?ê°œì„  ?œì•ˆ
        if term_name:
            if len(term_name) < 3:
                suggestions.append("?©ì–´ëª…ì„ ??êµ¬ì²´?ìœ¼ë¡?ë§Œë“œ?¸ìš”.")
            
            if ' ' in term_name:
                suggestions.append("?©ì–´ëª…ì—??ê³µë°±???œê±°?˜ëŠ” ê²ƒì„ ê³ ë ¤?˜ì„¸??")
            
            if not re.match(r'^[ê°€-??+', term_name):
                suggestions.append("?©ì–´ëª…ì„ ?œê?ë¡??œì‘?˜ë„ë¡??˜ì„¸??")
        
        # ?•ì˜ ê°œì„  ?œì•ˆ
        if definition:
            if len(definition) < 20:
                suggestions.append("?•ì˜ë¥????ì„¸???‘ì„±?˜ì„¸??")
            
            if definition.endswith('.'):
                suggestions.append("?•ì˜?ì„œ ë§ˆì¹¨?œë? ?œê±°?˜ëŠ” ê²ƒì„ ê³ ë ¤?˜ì„¸??")
            
            if '?? in definition and len(definition) < 50:
                suggestions.append("'?????¬ìš©???ŒëŠ” êµ¬ì²´?ì¸ ?ˆì‹œë¥?ì¶”ê??˜ì„¸??")
        
        # ì¹´í…Œê³ ë¦¬ ê°œì„  ?œì•ˆ
        if category == 'ê¸°í?':
            suggestions.append("??êµ¬ì²´?ì¸ ì¹´í…Œê³ ë¦¬ë¥?ì§€?•í•˜?¸ìš”.")
        
        return suggestions
    
    def generate_quality_report(self, dictionary) -> Dict[str, Any]:
        """?ˆì§ˆ ë³´ê³ ???ì„±"""
        if not dictionary:
            return {'error': '?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??'}
        
        # ?„ì²´ ê²€ì¦?
        validation_results = self.validate_all_terms(dictionary)
        synonym_validation = self.validate_synonym_groups(dictionary)
        
        # ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
        total_terms = validation_results['total_terms']
        valid_terms = validation_results['valid_terms']
        quality_score = (valid_terms / total_terms * 100) if total_terms > 0 else 0
        
        # ë³´ê³ ???ì„±
        report = {
            'generated_at': datetime.now().isoformat(),
            'quality_score': round(quality_score, 2),
            'total_terms': total_terms,
            'valid_terms': valid_terms,
            'invalid_terms': validation_results['invalid_terms'],
            'validation_results': validation_results,
            'synonym_validation': synonym_validation,
            'statistics': validation_results['statistics'],
            'recommendations': validation_results['recommendations']
        }
        
        return report
    
    def export_validation_report(self, dictionary, file_path: str) -> bool:
        """ê²€ì¦?ë³´ê³ ???´ë³´?´ê¸°"""
        try:
            import json
            
            report = self.generate_quality_report(dictionary)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ê²€ì¦?ë³´ê³ ???´ë³´?´ê¸° ?„ë£Œ: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"ê²€ì¦?ë³´ê³ ???´ë³´?´ê¸° ?¤íŒ¨: {e}")
            return False


def main():
    """ë©”ì¸ ?¨ìˆ˜ - ê²€ì¦ê¸° ?ŒìŠ¤??""
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?ŒìŠ¤?¸ìš© ?¬ì „ ?ì„±
    from source.data.legal_term_dictionary import LegalTermDictionary
    
    dictionary = LegalTermDictionary()
    
    # ?ŒìŠ¤???©ì–´ ì¶”ê?
    test_terms = [
        {
            'term_id': 'T001',
            'term_name': 'ê³„ì•½',
            'definition': '?¹ì‚¬??ê°??˜ì‚¬?œì‹œ???©ì¹˜',
            'category': 'ë¯¼ì‚¬ë²?,
            'frequency': 100
        },
        {
            'term_id': 'T002',
            'term_name': 'ê³„ì•½??,
            'definition': 'ê³„ì•½???´ìš©??ë¬¸ì„œë¡??‘ì„±??ê²?,
            'category': 'ë¯¼ì‚¬ë²?,
            'frequency': 80
        },
        {
            'term_id': 'T003',
            'term_name': '?í•´ë°°ìƒ',
            'definition': 'ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´??ë°°ìƒ',
            'category': 'ë¯¼ì‚¬ë²?,
            'frequency': 90
        }
    ]
    
    for term in test_terms:
        dictionary.add_term(term)
    
    # ê²€ì¦ê¸° ì´ˆê¸°??
    validator = TermValidator(dictionary)
    
    # ?„ì²´ ê²€ì¦?
    validation_results = validator.validate_all_terms(dictionary)
    print(f"ê²€ì¦?ê²°ê³¼: {validation_results}")
    
    # ?ˆì§ˆ ë³´ê³ ???ì„±
    report = validator.generate_quality_report(dictionary)
    print(f"?ˆì§ˆ ë³´ê³ ?? {report}")


if __name__ == "__main__":
    main()
