# -*- coding: utf-8 -*-
"""
?™ì˜??ê´€ë¦¬ì

ë²•ë¥  ?©ì–´???™ì˜??ê·¸ë£¹??ê´€ë¦¬í•˜ê³??ë™?¼ë¡œ ?ì„±?©ë‹ˆ??
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class SynonymManager:
    """?™ì˜??ê´€ë¦¬ì ?´ë˜??""
    
    def __init__(self, dictionary=None):
        """?™ì˜??ê´€ë¦¬ì ì´ˆê¸°??""
        self.dictionary = dictionary
        self.synonym_patterns = self._load_synonym_patterns()
        
        logger.info("SynonymManager initialized")
    
    def _load_synonym_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """?™ì˜???¨í„´ ë¡œë“œ"""
        return {
            'contract_terms': [
                {
                    'standard': 'ê³„ì•½',
                    'patterns': [r'ê³„ì•½??, r'ê³„ì•½ê´€ê³?, r'ê³„ì•½ì²´ê²°'],
                    'confidence': 0.95
                }
            ],
            'damage_terms': [
                {
                    'standard': '?í•´ë°°ìƒ',
                    'patterns': [r'?í•´ë³´ìƒ', r'?í•´ë°°ìƒì±…ì„'],
                    'confidence': 0.90
                },
                {
                    'standard': '?í•´',
                    'patterns': [r'?¼í•´', r'?ì‹¤'],
                    'confidence': 0.90
                },
                {
                    'standard': 'ë°°ìƒ',
                    'patterns': [r'ë³´ìƒ', r'ë°°ìƒê¸?],
                    'confidence': 0.85
                }
            ],
            'legal_terms': [
                {
                    'standard': 'ë²•ë¥ ',
                    'patterns': [r'ë²•ë ¹', r'ë²•ê·œ'],
                    'confidence': 0.85
                },
                {
                    'standard': 'ì¡°ë¬¸',
                    'patterns': [r'ë²•ì¡°ë¬?, r'ì¡°í•­'],
                    'confidence': 0.90
                }
            ],
            'court_terms': [
                {
                    'standard': 'ë²•ì›',
                    'patterns': [r'?¬íŒ??, r'ë²•ì •'],
                    'confidence': 0.80
                }
            ],
            'case_terms': [
                {
                    'standard': '?¬ê±´',
                    'patterns': [r'?¬ê±´ë²ˆí˜¸', r'?¬ê±´ëª?],
                    'confidence': 0.80
                }
            ],
            'party_terms': [
                {
                    'standard': '?¹ì‚¬??,
                    'patterns': [r'ê³„ì•½?¹ì‚¬??, r'ê³„ì•½??],
                    'confidence': 0.90
                }
            ],
            'tort_terms': [
                {
                    'standard': 'ë¶ˆë²•?‰ìœ„',
                    'patterns': [r'ë¶ˆë²•?‰ìœ„ì±…ì„'],
                    'confidence': 0.85
                }
            ]
        }
    
    def create_synonym_groups_from_patterns(self, dictionary) -> Dict[str, Any]:
        """?¨í„´ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„±"""
        if not dictionary:
            logger.error("?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info("?¨í„´ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?œì‘")
        
        for category, patterns in self.synonym_patterns.items():
            for pattern_data in patterns:
                try:
                    result = self._create_group_from_pattern(pattern_data)
                    if result['success']:
                        results['created'] += 1
                        logger.info(f"?™ì˜??ê·¸ë£¹ ?ì„±: {pattern_data['standard']} -> {result['variants']}")
                    else:
                        results['skipped'] += 1
                        logger.warning(f"?™ì˜??ê·¸ë£¹ ê±´ë„ˆ?€: {pattern_data['standard']} - {result['reason']}")
                except Exception as e:
                    results['errors'] += 1
                    logger.error(f"?™ì˜??ê·¸ë£¹ ?ì„± ?¤íŒ¨ ({pattern_data['standard']}): {e}")
        
        logger.info(f"?¨í„´ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?„ë£Œ - {results['created']}ê°??ì„±, {results['skipped']}ê°?ê±´ë„ˆ?€, {results['errors']}ê°??¤ë¥˜")
        return results
    
    def _create_group_from_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """ê°œë³„ ?¨í„´?ì„œ ?™ì˜??ê·¸ë£¹ ?ì„±"""
        standard_term = pattern_data['standard']
        patterns = pattern_data['patterns']
        confidence = pattern_data['confidence']
        
        # ?œì? ?©ì–´ê°€ ?¬ì „???ˆëŠ”ì§€ ?•ì¸
        if not self.dictionary.get_term(standard_term):
            return {
                'success': False,
                'reason': f"?œì? ?©ì–´ '{standard_term}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤."
            }
        
        # ?¨í„´??ë§¤ì¹­?˜ëŠ” ?©ì–´??ì°¾ê¸°
        matching_terms = []
        for pattern in patterns:
            matches = self._find_terms_by_pattern(pattern)
            matching_terms.extend(matches)
        
        # ì¤‘ë³µ ?œê±°
        matching_terms = list(set(matching_terms))
        
        if not matching_terms:
            return {
                'success': False,
                'reason': f"?œì? ?©ì–´ '{standard_term}'???€??ë§¤ì¹­ ?©ì–´ê°€ ?†ìŠµ?ˆë‹¤."
            }
        
        # ?™ì˜??ê·¸ë£¹ ?ì„±
        group_id = f"{standard_term}_group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = self.dictionary.create_synonym_group(
            group_id,
            standard_term,
            matching_terms,
            confidence
        )
        
        if success:
            return {
                'success': True,
                'variants': matching_terms,
                'group_id': group_id
            }
        else:
            return {
                'success': False,
                'reason': "?¬ì „???™ì˜??ê·¸ë£¹ ì¶”ê? ?¤íŒ¨"
            }
    
    def _find_terms_by_pattern(self, pattern: str) -> List[str]:
        """?¨í„´??ë§¤ì¹­?˜ëŠ” ?©ì–´??ì°¾ê¸°"""
        matching_terms = []
        
        for term_name in self.dictionary.term_index.keys():
            if re.search(pattern, term_name, re.IGNORECASE):
                matching_terms.append(term_name)
        
        return matching_terms
    
    def create_synonym_groups_from_frequency(self, dictionary, min_frequency: int = 5) -> Dict[str, Any]:
        """ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„±"""
        if not dictionary:
            logger.error("?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info(f"ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?œì‘ (ìµœì†Œ ë¹ˆë„: {min_frequency})")
        
        # ë¹ˆë„ê°€ ?’ì? ?©ì–´??ì°¾ê¸°
        frequent_terms = [
            term for term, freq in dictionary.frequency_index.items()
            if freq >= min_frequency
        ]
        
        # ? ì‚¬???©ì–´??ê·¸ë£¹??
        term_groups = self._group_similar_terms(frequent_terms)
        
        for group in term_groups:
            if len(group) < 2:  # ìµœì†Œ 2ê°??´ìƒ???©ì–´ê°€ ?ˆì–´??ê·¸ë£¹ ?ì„±
                continue
            
            try:
                # ê°€??ë¹ˆë„ê°€ ?’ì? ?©ì–´ë¥??œì? ?©ì–´ë¡?? íƒ
                standard_term = max(group, key=lambda x: dictionary.frequency_index.get(x, 0))
                variants = [term for term in group if term != standard_term]
                
                group_id = f"freq_group_{standard_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                confidence = 0.7  # ë¹ˆë„ ê¸°ë°˜?´ë?ë¡???? ? ë¢°??
                
                success = dictionary.create_synonym_group(
                    group_id,
                    standard_term,
                    variants,
                    confidence
                )
                
                if success:
                    results['created'] += 1
                    logger.info(f"ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„±: {standard_term} -> {variants}")
                else:
                    results['skipped'] += 1
                    logger.warning(f"ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ê±´ë„ˆ?€: {standard_term}")
                    
            except Exception as e:
                results['errors'] += 1
                logger.error(f"ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?¤íŒ¨: {e}")
        
        logger.info(f"ë¹ˆë„ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?„ë£Œ - {results['created']}ê°??ì„±, {results['skipped']}ê°?ê±´ë„ˆ?€, {results['errors']}ê°??¤ë¥˜")
        return results
    
    def _group_similar_terms(self, terms: List[str]) -> List[List[str]]:
        """? ì‚¬???©ì–´?¤ì„ ê·¸ë£¹??""
        groups = []
        used_terms = set()
        
        for term in terms:
            if term in used_terms:
                continue
            
            # ?„ì¬ ?©ì–´?€ ? ì‚¬???©ì–´??ì°¾ê¸°
            similar_terms = [term]
            
            for other_term in terms:
                if other_term == term or other_term in used_terms:
                    continue
                
                # ê°„ë‹¨??? ì‚¬??ê³„ì‚° (ë¬¸ì???¬í•¨ ê´€ê³?
                if self._calculate_similarity(term, other_term) > 0.7:
                    similar_terms.append(other_term)
                    used_terms.add(other_term)
            
            if len(similar_terms) > 1:
                groups.append(similar_terms)
            
            used_terms.add(term)
        
        return groups
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """???©ì–´ ê°„ì˜ ? ì‚¬??ê³„ì‚°"""
        # ê°„ë‹¨??ë¬¸ì??? ì‚¬??ê³„ì‚°
        if term1 in term2 or term2 in term1:
            return 0.8
        
        # ê³µí†µ ë¬¸ì ë¹„ìœ¨ ê³„ì‚°
        common_chars = set(term1) & set(term2)
        total_chars = set(term1) | set(term2)
        
        if len(total_chars) == 0:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    def create_manual_synonym_groups(self, dictionary, synonym_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?˜ë™ ?•ì˜???™ì˜??ê·¸ë£¹ ?ì„±"""
        if not dictionary:
            logger.error("?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info(f"?˜ë™ ?•ì˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?œì‘ - {len(synonym_definitions)}ê°?ê·¸ë£¹")
        
        for group_data in synonym_definitions:
            try:
                standard_term = group_data['standard_term']
                variants = group_data['variants']
                confidence = group_data.get('confidence', 0.9)
                group_id = group_data.get('group_id', f"manual_{standard_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # ?œì? ?©ì–´ê°€ ?¬ì „???ˆëŠ”ì§€ ?•ì¸
                if not dictionary.get_term(standard_term):
                    results['skipped'] += 1
                    logger.warning(f"?œì? ?©ì–´ '{standard_term}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
                    continue
                
                # ?¤ì œë¡?ì¡´ì¬?˜ëŠ” ë³€?•ì–´ë§??„í„°ë§?
                existing_variants = []
                for variant in variants:
                    if dictionary.get_term(variant):
                        existing_variants.append(variant)
                
                if not existing_variants:
                    results['skipped'] += 1
                    logger.warning(f"?œì? ?©ì–´ '{standard_term}'??ë³€?•ì–´ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
                    continue
                
                # ?™ì˜??ê·¸ë£¹ ?ì„±
                success = dictionary.create_synonym_group(
                    group_id,
                    standard_term,
                    existing_variants,
                    confidence
                )
                
                if success:
                    results['created'] += 1
                    logger.info(f"?˜ë™ ?™ì˜??ê·¸ë£¹ ?ì„±: {group_id} ({standard_term} -> {existing_variants})")
                else:
                    results['skipped'] += 1
                    logger.warning(f"?˜ë™ ?™ì˜??ê·¸ë£¹ ?ì„± ?¤íŒ¨: {group_id}")
                    
            except Exception as e:
                results['errors'] += 1
                logger.error(f"?˜ë™ ?™ì˜??ê·¸ë£¹ ?ì„± ?¤íŒ¨ ({group_data.get('group_id', 'unknown')}): {e}")
        
        logger.info(f"?˜ë™ ?•ì˜ ?™ì˜??ê·¸ë£¹ ?ì„± ?„ë£Œ - {results['created']}ê°??ì„±, {results['skipped']}ê°?ê±´ë„ˆ?€, {results['errors']}ê°??¤ë¥˜")
        return results
    
    def get_synonym_statistics(self, dictionary) -> Dict[str, Any]:
        """?™ì˜??ê·¸ë£¹ ?µê³„ ì¡°íšŒ"""
        if not dictionary:
            return {}
        
        total_groups = len(dictionary.synonym_groups)
        total_variants = sum(len(group['variants']) for group in dictionary.synonym_groups.values())
        
        # ? ë¢°?„ë³„ ë¶„í¬
        confidence_distribution = defaultdict(int)
        for group in dictionary.synonym_groups.values():
            confidence = group.get('confidence', 0)
            confidence_range = f"{int(confidence * 10) * 10}%"
            confidence_distribution[confidence_range] += 1
        
        return {
            'total_groups': total_groups,
            'total_variants': total_variants,
            'average_variants_per_group': total_variants / total_groups if total_groups > 0 else 0,
            'confidence_distribution': dict(confidence_distribution)
        }
    
    def validate_synonym_groups(self, dictionary) -> Dict[str, Any]:
        """?™ì˜??ê·¸ë£¹ ? íš¨??ê²€??""
        if not dictionary:
            return {'is_valid': False, 'issues': ['?¬ì „???œê³µ?˜ì? ?Šì•˜?µë‹ˆ??']}
        
        issues = []
        
        for group_id, group_data in dictionary.synonym_groups.items():
            # ?œì? ?©ì–´ê°€ ?¬ì „???ˆëŠ”ì§€ ?•ì¸
            standard_term = group_data.get('standard_term', '')
            if not dictionary.get_term(standard_term):
                issues.append(f"?™ì˜??ê·¸ë£¹ {group_id}???œì? ?©ì–´ '{standard_term}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
            
            # ë³€?•ì–´?¤ì´ ?¬ì „???ˆëŠ”ì§€ ?•ì¸
            variants = group_data.get('variants', [])
            for variant in variants:
                if not dictionary.get_term(variant):
                    issues.append(f"?™ì˜??ê·¸ë£¹ {group_id}??ë³€?•ì–´ '{variant}'ê°€ ?¬ì „???†ìŠµ?ˆë‹¤.")
            
            # ? ë¢°??ë²”ìœ„ ?•ì¸
            confidence = group_data.get('confidence', 0)
            if not 0 <= confidence <= 1:
                issues.append(f"?™ì˜??ê·¸ë£¹ {group_id}??? ë¢°?„ê? ? íš¨?˜ì? ?ŠìŠµ?ˆë‹¤: {confidence}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }


def main():
    """ë©”ì¸ ?¨ìˆ˜ - ?™ì˜??ê´€ë¦??ŒìŠ¤??""
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
        {'term_id': 'T001', 'term_name': 'ê³„ì•½', 'definition': 'ê³„ì•½???•ì˜', 'category': 'ë¯¼ì‚¬ë²?, 'frequency': 100},
        {'term_id': 'T002', 'term_name': 'ê³„ì•½??, 'definition': 'ê³„ì•½?œì˜ ?•ì˜', 'category': 'ë¯¼ì‚¬ë²?, 'frequency': 80},
        {'term_id': 'T003', 'term_name': '?í•´ë°°ìƒ', 'definition': '?í•´ë°°ìƒ???•ì˜', 'category': 'ë¯¼ì‚¬ë²?, 'frequency': 90},
        {'term_id': 'T004', 'term_name': '?í•´ë³´ìƒ', 'definition': '?í•´ë³´ìƒ???•ì˜', 'category': 'ë¯¼ì‚¬ë²?, 'frequency': 70},
    ]
    
    for term in test_terms:
        dictionary.add_term(term)
    
    # ?™ì˜??ê´€ë¦¬ì ì´ˆê¸°??
    synonym_manager = SynonymManager(dictionary)
    
    # ?¨í„´ ê¸°ë°˜ ?™ì˜??ê·¸ë£¹ ?ì„±
    results = synonym_manager.create_synonym_groups_from_patterns(dictionary)
    print(f"?¨í„´ ê¸°ë°˜ ê²°ê³¼: {results}")
    
    # ?µê³„ ì¡°íšŒ
    stats = synonym_manager.get_synonym_statistics(dictionary)
    print(f"?™ì˜???µê³„: {stats}")
    
    # ? íš¨??ê²€??
    validation = synonym_manager.validate_synonym_groups(dictionary)
    print(f"? íš¨??ê²€?? {validation}")


if __name__ == "__main__":
    main()
