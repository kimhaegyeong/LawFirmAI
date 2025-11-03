#!/usr/bin/env python3
"""
ë²•ë¥  ?©ì–´ ?¬ì „ ?•ì¥???„í•œ ?©ì–´ ì¶”ì¶œ ?¤í¬ë¦½íŠ¸
ê¸°ì¡´ ë²•ë ¹ ë°??ë? ?°ì´?°ì—??ë²•ë¥  ?©ì–´ë¥??ë™?¼ë¡œ ì¶”ì¶œ?˜ê³  ë¶„ë¥˜?©ë‹ˆ??
"""

import os
import json
import re
import logging
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LegalTerm:
    """ë²•ë¥  ?©ì–´ ?°ì´???´ë˜??""
    term: str
    category: str
    domain: str
    frequency: int
    sources: List[str]
    synonyms: List[str]
    related_terms: List[str]
    context: List[str]
    confidence: float

class LegalTermExtractor:
    """ë²•ë¥  ?©ì–´ ì¶”ì¶œê¸?""
    
    def __init__(self):
        self.extracted_terms: Dict[str, LegalTerm] = {}
        self.domain_patterns = self._initialize_domain_patterns()
        self.legal_patterns = self._initialize_legal_patterns()
        self.stop_words = self._initialize_stop_words()
        
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """?„ë©”?¸ë³„ ?¨í„´ ì´ˆê¸°??""
        return {
            "?•ì‚¬ë²?: [
                r"ë²”ì£„", r"ì²˜ë²Œ", r"?•ë²Œ", r"êµ¬ì†", r"ê¸°ì†Œ", r"ê³µì†Œ", r"?¼ê³ ", r"ê²€??,
                r"ë³€?¸ì‚¬", r"?¬íŒ", r"?ê²°", r"?•ì‚¬?Œì†¡", r"ë¶ˆë²•?‰ìœ„", r"ê³¼ì‹¤", r"ê³ ì˜"
            ],
            "ë¯¼ì‚¬ë²?: [
                r"ê³„ì•½", r"?í•´ë°°ìƒ", r"?Œìœ ê¶?, r"ì±„ê¶Œ", r"ì±„ë¬´", r"?´í–‰", r"?„ë°˜",
                r"?´ì?", r"?´ì œ", r"ë¬´íš¨", r"ì·¨ì†Œ", r"ë¯¼ì‚¬?Œì†¡", r"?Œì¥", r"?µë???
            ],
            "ê°€ì¡±ë²•": [
                r"?¼ì¸", r"?´í˜¼", r"?ì†", r"?‘ìœ¡", r"?„ìë£?, r"?¬ì‚°ë¶„í• ", r"?‘ìœ¡ê¶?,
                r"ë©´ì ‘êµì„­ê¶?, r"?‘ìœ¡ë¹?, r"ê°€?•ë²•??, r"ê°€ì¡±ë²•", r"?´í˜¼ë²?
            ],
            "?ì‚¬ë²?: [
                r"?Œì‚¬", r"ì£¼ì‹", r"?´ìŒ", r"?˜í‘œ", r"?í–‰??, r"?Œì‚¬ë²?, r"?ë²•",
                r"ì£¼ì£¼", r"?´ì‚¬", r"ê°ì‚¬", r"?ë³¸ê¸?, r"ì£¼ì‹?Œì‚¬"
            ],
            "?¸ë™ë²?: [
                r"ê·¼ë¡œ", r"ê·¼ë¡œ??, r"ê·¼ë¡œê³„ì•½", r"?„ê¸ˆ", r"ê·¼ë¡œ?œê°„", r"?´ê³ ",
                r"ë¶€?¹í•´ê³?, r"?¸ë™?„ì›??, r"ê·¼ë¡œê¸°ì?ë²?, r"?¸ë™ì¡°í•©ë²?
            ],
            "ë¶€?™ì‚°ë²?: [
                r"ë¶€?™ì‚°", r"? ì?", r"ê±´ë¬¼", r"?±ê¸°", r"?Œìœ ê¶Œì´??, r"ë§¤ë§¤",
                r"?„ë?ì°?, r"?„ì„¸", r"?”ì„¸", r"?±ê¸°ë¶€?±ë³¸", r"ë¶€?™ì‚°?±ê¸°ë²?
            ],
            "?¹í—ˆë²?: [
                r"?¹í—ˆ", r"?¹í—ˆê¶?, r"?¹í—ˆì¶œì›", r"?¹í—ˆ?±ë¡", r"?¹í—ˆì¹¨í•´", r"ë°œëª…",
                r"?¹í—ˆì²?, r"?¹í—ˆ?¬íŒ??, r"?¹í—ˆë²?, r"?¹í—ˆ?¬ì‚¬"
            ],
            "?‰ì •ë²?: [
                r"?‰ì •ì²˜ë¶„", r"?‰ì •?Œì†¡", r"?‰ì •ë²?, r"?ˆê?", r"?¸ê?", r"?¹ì¸",
                r"? ê³ ", r"? ì²­", r"?‰ì •ê¸°ê?", r"ê³µë¬´??, r"?‰ì •?ˆì°¨"
            ]
        }
    
    def _initialize_legal_patterns(self) -> List[str]:
        """ë²•ë¥  ?©ì–´ ?¨í„´ ì´ˆê¸°??""
        return [
            # ì¡°ë¬¸ ?¨í„´
            r"??d+ì¡?,
            r"??d+??,
            r"??d+??,
            r"??d+??,
            r"??d+??,
            r"??d+??,
            
            # ë²•ë¥ ëª??¨í„´
            r"[ê°€-??+ë²?,
            r"[ê°€-??+ê·œì¹™",
            r"[ê°€-??+??,
            r"[ê°€-??+?œí–‰??,
            r"[ê°€-??+?œí–‰ê·œì¹™",
            
            # ê¶Œë¦¬/?˜ë¬´ ?¨í„´
            r"[ê°€-??+ê¶?,
            r"[ê°€-??+?˜ë¬´",
            r"[ê°€-??+ì±…ì„",
            r"[ê°€-??+?˜ë¬´",
            
            # ?ˆì°¨ ?¨í„´
            r"[ê°€-??+?ˆì°¨",
            r"[ê°€-??+? ì²­",
            r"[ê°€-??+? ê³ ",
            r"[ê°€-??+?ˆê?",
            r"[ê°€-??+?¸ê?",
            r"[ê°€-??+?¹ì¸",
            
            # ê¸°ê? ?¨í„´
            r"[ê°€-??+??,
            r"[ê°€-??+ì²?,
            r"[ê°€-??+ë¶€",
            r"[ê°€-??+?„ì›??,
            r"[ê°€-??+ë²•ì›",
            
            # ?‰ìœ„ ?¨í„´
            r"[ê°€-??+?‰ìœ„",
            r"[ê°€-??+ì²˜ë¶„",
            r"[ê°€-??+ê²°ì •",
            r"[ê°€-??+ëª…ë ¹",
            r"[ê°€-??+ì§€??
        ]
    
    def _initialize_stop_words(self) -> Set[str]:
        """ë¶ˆìš©??ì´ˆê¸°??""
        return {
            "ê²?, "??, "??, "ë°?, "?ëŠ”", "ê·?, "??, "?€", "??, "ê°€", "??, "ë¥?,
            "??, "?ì„œ", "ë¡?, "?¼ë¡œ", "?€", "ê³?, "??, "?€", "??, "ë§?, "ë¶€??,
            "ê¹Œì?", "ê¹Œì???, "?ì˜", "?ë???, "?ê???, "?ë”°ë¥?, "?ì˜??
        }
    
    def extract_terms_from_laws(self, law_data_dir: str) -> Dict[str, LegalTerm]:
        """ë²•ë ¹ ?°ì´?°ì—???©ì–´ ì¶”ì¶œ"""
        logger.info(f"ë²•ë ¹ ?°ì´?°ì—???©ì–´ ì¶”ì¶œ ?œì‘: {law_data_dir}")
        
        extracted_terms = defaultdict(lambda: {
            'frequency': 0,
            'sources': [],
            'domains': set(),
            'contexts': set()
        })
        
        # ë²•ë ¹ ?Œì¼??ì²˜ë¦¬
        for root, dirs, files in os.walk(law_data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # ë²•ë ¹ ?°ì´?°ì—???©ì–´ ì¶”ì¶œ
                        if 'laws' in data:
                            for law in data['laws']:
                                self._extract_from_law(law, extracted_terms, file_path)
                                
                    except Exception as e:
                        logger.error(f"?Œì¼ ì²˜ë¦¬ ?¤ë¥˜ {file_path}: {e}")
                        continue
        
        # LegalTerm ê°ì²´ë¡?ë³€??
        legal_terms = {}
        for term, data in extracted_terms.items():
            if len(term) >= 2 and term not in self.stop_words:  # ìµœì†Œ ê¸¸ì´ ë°?ë¶ˆìš©???„í„°ë§?
                legal_terms[term] = LegalTerm(
                    term=term,
                    category=self._categorize_term(term),
                    domain=self._determine_domain(term, data['domains']),
                    frequency=data['frequency'],
                    sources=data['sources'],
                    synonyms=[],
                    related_terms=[],
                    context=list(data['contexts']),
                    confidence=self._calculate_confidence(term, data)
                )
        
        logger.info(f"ë²•ë ¹?ì„œ ì¶”ì¶œ???©ì–´ ?? {len(legal_terms)}")
        return legal_terms
    
    def extract_terms_from_precedents(self, precedent_data_dir: str) -> Dict[str, LegalTerm]:
        """?ë? ?°ì´?°ì—???©ì–´ ì¶”ì¶œ"""
        logger.info(f"?ë? ?°ì´?°ì—???©ì–´ ì¶”ì¶œ ?œì‘: {precedent_data_dir}")
        
        extracted_terms = defaultdict(lambda: {
            'frequency': 0,
            'sources': [],
            'domains': set(),
            'contexts': set()
        })
        
        # ?ë? ?Œì¼??ì²˜ë¦¬
        for root, dirs, files in os.walk(precedent_data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # ?ë? ?°ì´?°ì—???©ì–´ ì¶”ì¶œ
                        if 'cases' in data:
                            for case in data['cases']:
                                self._extract_from_precedent(case, extracted_terms, file_path)
                                
                    except Exception as e:
                        logger.error(f"?Œì¼ ì²˜ë¦¬ ?¤ë¥˜ {file_path}: {e}")
                        continue
        
        # LegalTerm ê°ì²´ë¡?ë³€??
        legal_terms = {}
        for term, data in extracted_terms.items():
            if len(term) >= 2 and term not in self.stop_words:
                legal_terms[term] = LegalTerm(
                    term=term,
                    category=self._categorize_term(term),
                    domain=self._determine_domain(term, data['domains']),
                    frequency=data['frequency'],
                    sources=data['sources'],
                    synonyms=[],
                    related_terms=[],
                    context=list(data['contexts']),
                    confidence=self._calculate_confidence(term, data)
                )
        
        logger.info(f"?ë??ì„œ ì¶”ì¶œ???©ì–´ ?? {len(legal_terms)}")
        return legal_terms
    
    def _extract_from_law(self, law: Dict[str, Any], extracted_terms: Dict, file_path: str):
        """ê°œë³„ ë²•ë ¹?ì„œ ?©ì–´ ì¶”ì¶œ"""
        # ë²•ë ¹ëª…ì—???©ì–´ ì¶”ì¶œ
        if 'law_name' in law and law['law_name']:
            self._extract_terms_from_text(law['law_name'], extracted_terms, file_path, "ë²•ë ¹ëª?)
        
        # ì¡°ë¬¸?ì„œ ?©ì–´ ì¶”ì¶œ
        if 'articles' in law:
            for article in law['articles']:
                if 'article_title' in article and article['article_title']:
                    self._extract_terms_from_text(article['article_title'], extracted_terms, file_path, "ì¡°ë¬¸?œëª©")
                
                if 'article_content' in article and article['article_content']:
                    self._extract_terms_from_text(article['article_content'], extracted_terms, file_path, "ì¡°ë¬¸?´ìš©")
    
    def _extract_from_precedent(self, case: Dict[str, Any], extracted_terms: Dict, file_path: str):
        """ê°œë³„ ?ë??ì„œ ?©ì–´ ì¶”ì¶œ"""
        # ?¬ê±´ëª…ì—???©ì–´ ì¶”ì¶œ
        if 'case_name' in case and case['case_name']:
            self._extract_terms_from_text(case['case_name'], extracted_terms, file_path, "?¬ê±´ëª?)
        
        # ?ì‹œ?¬í•­?ì„œ ?©ì–´ ì¶”ì¶œ
        if 'sections' in case:
            for section in case['sections']:
                if section.get('has_content', False) and section.get('section_content'):
                    section_type = section.get('section_type_korean', 'ê¸°í?')
                    self._extract_terms_from_text(section['section_content'], extracted_terms, file_path, section_type)
    
    def _extract_terms_from_text(self, text: str, extracted_terms: Dict, file_path: str, context: str):
        """?ìŠ¤?¸ì—???©ì–´ ì¶”ì¶œ"""
        if not text or not isinstance(text, str):
            return
        
        # ?¨í„´ ë§¤ì¹­?¼ë¡œ ?©ì–´ ì¶”ì¶œ
        for pattern in self.legal_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # ê·¸ë£¹???ˆëŠ” ê²½ìš° ì²?ë²ˆì§¸ ê·¸ë£¹ ?¬ìš©
                
                if match and len(match) >= 2:
                    extracted_terms[match]['frequency'] += 1
                    extracted_terms[match]['sources'].append(file_path)
                    extracted_terms[match]['contexts'].add(context)
                    
                    # ?„ë©”??ë¶„ë¥˜
                    domain = self._classify_domain(match)
                    if domain:
                        extracted_terms[match]['domains'].add(domain)
        
        # ?¼ë°˜?ì¸ ë²•ë¥  ?©ì–´ ì¶”ì¶œ (2-4ê¸€???œê?)
        general_terms = re.findall(r'[ê°€-??{2,4}', text)
        for term in general_terms:
            if term not in self.stop_words and self._is_legal_term(term):
                extracted_terms[term]['frequency'] += 1
                extracted_terms[term]['sources'].append(file_path)
                extracted_terms[term]['contexts'].add(context)
                
                domain = self._classify_domain(term)
                if domain:
                    extracted_terms[term]['domains'].add(domain)
    
    def _classify_domain(self, term: str) -> str:
        """?©ì–´???„ë©”??ë¶„ë¥˜"""
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, term):
                    return domain
        return "ê¸°í?"
    
    def _is_legal_term(self, term: str) -> bool:
        """ë²•ë¥  ?©ì–´ ?¬ë? ?ë‹¨"""
        legal_indicators = [
            'ë²?, 'ê·œì¹™', '??, 'ê¶?, '?˜ë¬´', 'ì±…ì„', '?ˆì°¨', '? ì²­', '? ê³ ',
            '?ˆê?', '?¸ê?', '?¹ì¸', '??, 'ì²?, 'ë¶€', '?„ì›??, 'ë²•ì›',
            '?‰ìœ„', 'ì²˜ë¶„', 'ê²°ì •', 'ëª…ë ¹', 'ì§€??, '?Œì†¡', '?¬íŒ', '?ê²°'
        ]
        
        return any(indicator in term for indicator in legal_indicators)
    
    def _categorize_term(self, term: str) -> str:
        """?©ì–´ ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜"""
        if 'ë²? in term or 'ê·œì¹™' in term or '?? in term:
            return "ë²•ë¥ ëª?
        elif 'ì¡? in term or '?? in term or '?? in term:
            return "ì¡°ë¬¸"
        elif 'ê¶? in term:
            return "ê¶Œë¦¬"
        elif '?˜ë¬´' in term or 'ì±…ì„' in term:
            return "?˜ë¬´"
        elif '?ˆì°¨' in term or '? ì²­' in term or '? ê³ ' in term:
            return "?ˆì°¨"
        elif '?? in term or 'ì²? in term or 'ë¶€' in term:
            return "ê¸°ê?"
        elif '?Œì†¡' in term or '?¬íŒ' in term or '?ê²°' in term:
            return "?Œì†¡"
        else:
            return "?¼ë°˜"
    
    def _determine_domain(self, term: str, domains: Set[str]) -> str:
        """ì£¼ìš” ?„ë©”??ê²°ì •"""
        if not domains:
            return "ê¸°í?"
        
        # ê°€??ë¹ˆë²ˆ???„ë©”??ë°˜í™˜
        domain_counts = Counter(domains)
        return domain_counts.most_common(1)[0][0]
    
    def _calculate_confidence(self, term: str, data: Dict) -> float:
        """?©ì–´ ? ë¢°??ê³„ì‚°"""
        confidence = 0.0
        
        # ë¹ˆë„??ê¸°ë°˜ ?ìˆ˜ (0-0.4)
        frequency_score = min(data['frequency'] / 10.0, 0.4)
        confidence += frequency_score
        
        # ?ŒìŠ¤ ?¤ì–‘???ìˆ˜ (0-0.3)
        source_diversity = min(len(set(data['sources'])) / 5.0, 0.3)
        confidence += source_diversity
        
        # ì»¨í…?¤íŠ¸ ?¤ì–‘???ìˆ˜ (0-0.2)
        context_diversity = min(len(data['contexts']) / 3.0, 0.2)
        confidence += context_diversity
        
        # ?„ë©”??ëª…í™•???ìˆ˜ (0-0.1)
        if len(data['domains']) == 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def merge_and_deduplicate(self, law_terms: Dict[str, LegalTerm], precedent_terms: Dict[str, LegalTerm]) -> Dict[str, LegalTerm]:
        """?©ì–´ ?µí•© ë°?ì¤‘ë³µ ?œê±°"""
        logger.info("?©ì–´ ?µí•© ë°?ì¤‘ë³µ ?œê±° ?œì‘")
        
        merged_terms = {}
        
        # ë²•ë ¹ ?©ì–´ ì¶”ê?
        for term, legal_term in law_terms.items():
            merged_terms[term] = legal_term
        
        # ?ë? ?©ì–´ ?µí•©
        for term, legal_term in precedent_terms.items():
            if term in merged_terms:
                # ì¤‘ë³µ???©ì–´??ê²½ìš° ë¹ˆë„?˜ì? ?ŒìŠ¤ ?µí•©
                existing_term = merged_terms[term]
                existing_term.frequency += legal_term.frequency
                existing_term.sources.extend(legal_term.sources)
                existing_term.context.extend(legal_term.context)
                existing_term.confidence = max(existing_term.confidence, legal_term.confidence)
            else:
                merged_terms[term] = legal_term
        
        # ?ˆì§ˆ ?„í„°ë§?(? ë¢°??0.3 ?´ìƒ, ë¹ˆë„??2 ?´ìƒ)
        filtered_terms = {
            term: legal_term for term, legal_term in merged_terms.items()
            if legal_term.confidence >= 0.3 and legal_term.frequency >= 2
        }
        
        logger.info(f"?µí•© ???©ì–´ ?? {len(merged_terms)}")
        logger.info(f"?ˆì§ˆ ?„í„°ë§????©ì–´ ?? {len(filtered_terms)}")
        
        return filtered_terms
    
    def generate_semantic_relations(self, terms: Dict[str, LegalTerm]) -> Dict[str, Dict[str, List[str]]]:
        """?˜ë???ê´€ê³??ì„±"""
        logger.info("?˜ë???ê´€ê³??ì„± ?œì‘")
        
        semantic_relations = {}
        
        # ?„ë©”?¸ë³„ ê·¸ë£¹??
        domain_groups = defaultdict(list)
        for term, legal_term in terms.items():
            domain_groups[legal_term.domain].append(term)
        
        # ê°??„ë©”?¸ë³„ë¡??˜ë???ê´€ê³??ì„±
        for domain, domain_terms in domain_groups.items():
            if len(domain_terms) < 3:
                continue
            
            # ?ìœ„ ë¹ˆë„???©ì–´?¤ì„ ?€???©ì–´ë¡?? íƒ
            domain_term_freq = [(term, terms[term].frequency) for term in domain_terms]
            domain_term_freq.sort(key=lambda x: x[1], reverse=True)
            
            representative_terms = [term for term, freq in domain_term_freq[:5]]
            
            if representative_terms:
                main_term = representative_terms[0]
                synonyms = representative_terms[1:3] if len(representative_terms) > 1 else []
                related_terms = domain_terms[:10]  # ê´€???©ì–´
                
                semantic_relations[main_term] = {
                    "synonyms": synonyms,
                    "related": related_terms,
                    "context": [domain]
                }
        
        logger.info(f"?ì„±???˜ë???ê´€ê³??? {len(semantic_relations)}")
        return semantic_relations
    
    def save_results(self, terms: Dict[str, LegalTerm], semantic_relations: Dict[str, Dict[str, List[str]]], output_dir: str):
        """ê²°ê³¼ ?€??""
        os.makedirs(output_dir, exist_ok=True)
        
        # ?©ì–´ ?¬ì „ ?€??
        terms_dict = {}
        for term, legal_term in terms.items():
            terms_dict[term] = {
                "term": legal_term.term,
                "category": legal_term.category,
                "domain": legal_term.domain,
                "frequency": legal_term.frequency,
                "sources": legal_term.sources,
                "synonyms": legal_term.synonyms,
                "related_terms": legal_term.related_terms,
                "context": legal_term.context,
                "confidence": legal_term.confidence
            }
        
        terms_file = os.path.join(output_dir, "extracted_legal_terms.json")
        with open(terms_file, 'w', encoding='utf-8') as f:
            json.dump(terms_dict, f, ensure_ascii=False, indent=2)
        
        # ?˜ë???ê´€ê³??€??
        relations_file = os.path.join(output_dir, "semantic_relations.json")
        with open(relations_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_relations, f, ensure_ascii=False, indent=2)
        
        # ?µê³„ ë³´ê³ ???ì„±
        self._generate_statistics_report(terms, semantic_relations, output_dir)
        
        logger.info(f"ê²°ê³¼ ?€???„ë£Œ: {output_dir}")
    
    def _generate_statistics_report(self, terms: Dict[str, LegalTerm], semantic_relations: Dict[str, Dict[str, List[str]]], output_dir: str):
        """?µê³„ ë³´ê³ ???ì„±"""
        stats = {
            "extraction_summary": {
                "total_terms": len(terms),
                "total_semantic_relations": len(semantic_relations),
                "extraction_date": datetime.now().isoformat()
            },
            "domain_distribution": {},
            "category_distribution": {},
            "confidence_distribution": {},
            "frequency_distribution": {}
        }
        
        # ?„ë©”?¸ë³„ ë¶„í¬
        domain_counts = Counter(term.domain for term in terms.values())
        stats["domain_distribution"] = dict(domain_counts)
        
        # ì¹´í…Œê³ ë¦¬ë³?ë¶„í¬
        category_counts = Counter(term.category for term in terms.values())
        stats["category_distribution"] = dict(category_counts)
        
        # ? ë¢°??ë¶„í¬
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for term in terms.values():
            if term.confidence >= 0.7:
                confidence_ranges["high"] += 1
            elif term.confidence >= 0.4:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        stats["confidence_distribution"] = confidence_ranges
        
        # ë¹ˆë„??ë¶„í¬
        frequency_ranges = {"high": 0, "medium": 0, "low": 0}
        for term in terms.values():
            if term.frequency >= 10:
                frequency_ranges["high"] += 1
            elif term.frequency >= 5:
                frequency_ranges["medium"] += 1
            else:
                frequency_ranges["low"] += 1
        stats["frequency_distribution"] = frequency_ranges
        
        # ë³´ê³ ???€??
        report_file = os.path.join(output_dir, "extraction_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("ë²•ë¥  ?©ì–´ ì¶”ì¶œ ?œì‘")
    
    # ?°ì´???”ë ‰? ë¦¬ ?¤ì •
    law_data_dir = "data/processed/assembly/law"
    precedent_data_dir = "data/processed/assembly/precedent"
    output_dir = "data/extracted_terms"
    
    # ì¶”ì¶œê¸?ì´ˆê¸°??
    extractor = LegalTermExtractor()
    
    try:
        # ë²•ë ¹?ì„œ ?©ì–´ ì¶”ì¶œ
        law_terms = extractor.extract_terms_from_laws(law_data_dir)
        
        # ?ë??ì„œ ?©ì–´ ì¶”ì¶œ
        precedent_terms = extractor.extract_terms_from_precedents(precedent_data_dir)
        
        # ?©ì–´ ?µí•© ë°?ì¤‘ë³µ ?œê±°
        merged_terms = extractor.merge_and_deduplicate(law_terms, precedent_terms)
        
        # ?˜ë???ê´€ê³??ì„±
        semantic_relations = extractor.generate_semantic_relations(merged_terms)
        
        # ê²°ê³¼ ?€??
        extractor.save_results(merged_terms, semantic_relations, output_dir)
        
        logger.info("ë²•ë¥  ?©ì–´ ì¶”ì¶œ ?„ë£Œ")
        
    except Exception as e:
        logger.error(f"?©ì–´ ì¶”ì¶œ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        raise

if __name__ == "__main__":
    main()
