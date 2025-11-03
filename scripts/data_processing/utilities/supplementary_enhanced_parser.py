#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë¶€ì¹??Œì‹±??ê°œì„ ??ì¡°ë¬¸ ?Œì„œ
?€?œë?êµ?ë²•ë¥  ë¶€ì¹??‘ì„± ê·œì¹™???°ë¥¸ ?•í™•??ë¶€ì¹??¸ì‹ ë°??Œì‹±
"""

import json
import sys
import os
import re
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple, Optional

# Windows ì½˜ì†”?ì„œ UTF-8 ?¸ì½”???¤ì •
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# ê¸°ì¡´ ?Œì„œ ëª¨ë“ˆ ê²½ë¡œ ì¶”ê?
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplementaryEnhancedParser(ImprovedArticleParser):
    """ë¶€ì¹??Œì‹±??ê°œì„ ??ì¡°ë¬¸ ?Œì„œ"""
    
    def __init__(self):
        super().__init__()
        
        # ë¶€ì¹?ê´€???•ê·œ???¨í„´??
        self.supplementary_patterns = [
            # ê¸°ë³¸ ë¶€ì¹??¨í„´
            r'ë¶€ì¹?s*<[^>]*>',
            r'ë¶€ì¹?s*$',
            r'ë¶€ì¹?s*?¼ì¹˜ê¸°ì ‘ê¸?,
            
            # ë¶€ì¹?ì¡°ë¬¸ ?¨í„´ (??ì¡? ??ì¡???
            r'??\d+)ì¡?s*\([^)]*\)',
            r'??\d+)ì¡?s*[ê°€-??',
            
            # ?œí–‰??ê´€???¨í„´
            r'??s*[ë²•ë ¹ê·œì¹™]\s*?€\s*ê³µí¬??s*? ë???s*?œí–‰?œë‹¤',
            r'??s*[ë²•ë ¹ê·œì¹™]\s*?€\s*ê³µí¬\s*??s*\d+ê°œì›”??s*ê²½ê³¼??s*? ë???s*?œí–‰?œë‹¤',
            r'??s*[ë²•ë ¹ê·œì¹™]\s*?€\s*\d{4}??s*\d{1,2}??s*\d{1,2}?¼ë???s*?œí–‰?œë‹¤',
            
            # ê²½ê³¼ì¡°ì¹˜ ê´€???¨í„´
            r'ê²½ê³¼ì¡°ì¹˜',
            r'?ìš©ë¡€',
            r'ì¢…ì „??s*ê·œì •',
            r'??s*[ë²•ë ¹ê·œì¹™]\s*?œí–‰\s*?¹ì‹œ',
        ]
        
        # ë¶€ì¹??œì‘???˜í??´ëŠ” ?¤ì›Œ?œë“¤
        self.supplementary_keywords = [
            'ë¶€ì¹?, '?œí–‰??, 'ê²½ê³¼ì¡°ì¹˜', '?ìš©ë¡€', 'ì¤€ë¹„í–‰??, 
            'ì¢…ì „ ì²˜ë¶„', '?¤ë¥¸ ë²•ë¥ ??ê°œì •', '?¤ë¥¸ ë²•ë¥ ???ì?'
        ]
    
    def _identify_supplementary_section(self, content: str) -> tuple:
        """ë¶€ì¹??¹ì…˜ ?ë³„"""
        lines = content.split('\n')
        supplementary_start = -1
        supplementary_end = len(lines)
        
        # ë¶€ì¹??œì‘??ì°¾ê¸°
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # ë¶€ì¹??œì‘ ?¨í„´ ?•ì¸
            if re.search(r'ë¶€ì¹?s*<[^>]*>', line_clean) or \
               re.search(r'ë¶€ì¹?s*$', line_clean) or \
               re.search(r'ë¶€ì¹?s*?¼ì¹˜ê¸°ì ‘ê¸?, line_clean):
                supplementary_start = i
                break
        
        if supplementary_start == -1:
            return None, None
        
        # ë¶€ì¹??ì  ì°¾ê¸° (ë¬¸ì„œ ?ê¹Œì§€)
        return supplementary_start, supplementary_end
    
    def _is_supplementary_line(self, line: str) -> bool:
        """ë¶€ì¹?ê´€???¼ì¸?¸ì? ?•ì¸"""
        # ë¹??¼ì¸?€ ë¶€ì¹™ì— ?¬í•¨?????ˆìŒ
        if not line.strip():
            return True
        
        # ë¶€ì¹??¤ì›Œ???•ì¸
        for keyword in self.supplementary_keywords:
            if keyword in line:
                return True
        
        # ë¶€ì¹?ì¡°ë¬¸ ?¨í„´ ?•ì¸
        if re.match(r'??d+ì¡?s*\([^)]*\)', line) or \
           re.match(r'??d+ì¡?s*[ê°€-??', line):
            return True
        
        # ?œí–‰??ê´€???¨í„´ ?•ì¸
        if re.search(r'?œí–‰?œë‹¤', line) or \
           re.search(r'ê³µí¬??s*? ë???, line) or \
           re.search(r'ê²½ê³¼??s*? ë???, line):
            return True
        
        return False
    
    def _is_supplementary_article(self, line: str) -> bool:
        """ë¶€ì¹?ì¡°ë¬¸?¸ì? ?•ì¸"""
        # ë¶€ì¹?ì¡°ë¬¸???¹ì§•?ì¸ ?¨í„´??
        supplementary_patterns = [
            r'?œí–‰??,
            r'ê²½ê³¼ì¡°ì¹˜',
            r'?ìš©ë¡€',
            r'ì¤€ë¹„í–‰??,
            r'ì¢…ì „\s*ì²˜ë¶„',
            r'?¤ë¥¸\s*ë²•ë¥ ??s*ê°œì •',
            r'?¤ë¥¸\s*ë²•ë¥ ??s*?ì?',
            r'ë²Œì¹™\s*?ìš©',
            r'ê³¼íƒœë£?s*?ìš©'
        ]
        
        for pattern in supplementary_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def _parse_supplementary_articles(self, content: str) -> list:
        """ë¶€ì¹?ì¡°ë¬¸ ?Œì‹±"""
        supplementary_start, supplementary_end = self._identify_supplementary_section(content)
        
        if supplementary_start is None:
            return []
        
        lines = content.split('\n')
        supplementary_content = '\n'.join(lines[supplementary_start:supplementary_end])
        
        print(f"ë¶€ì¹??¹ì…˜ ë°œê²¬: {supplementary_start}~{supplementary_end}")
        print(f"ë¶€ì¹??´ìš©: {supplementary_content[:200]}...")
        
        # ë¶€ì¹?ì¡°ë¬¸ ì¶”ì¶œ
        articles = []
        
        # ë¶€ì¹?ì¡°ë¬¸ ?¨í„´?¼ë¡œ ë§¤ì¹­ (???•í™•???¨í„´)
        article_pattern = r'??\d+)ì¡?s*\(([^)]*)\)\s*(.*?)(?=??d+ì¡?s*\(|$)'
        matches = re.finditer(article_pattern, supplementary_content, re.DOTALL)
        
        for match in matches:
            article_number = f"ë¶€ì¹™ì œ{match.group(1)}ì¡?
            article_title = match.group(2).strip()
            article_content = match.group(3).strip()
            
            # ?´ìš© ?•ë¦¬
            article_content = self._clean_content(article_content)
            
            if article_content:
                articles.append({
                    'article_number': article_number,
                    'article_title': article_title,
                    'article_content': article_content,
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(article_content.split()),
                    'char_count': len(article_content),
                    'is_supplementary': True
                })
        
        # ì¡°ë¬¸???†ëŠ” ?¨ìˆœ ë¶€ì¹?ì²˜ë¦¬
        if not articles and supplementary_content.strip():
            # ë¶€ì¹??¤ë” ?œê±°
            clean_content = re.sub(r'ë¶€ì¹?s*<[^>]*>?¼ì¹˜ê¸°ì ‘ê¸?, '', supplementary_content)
            clean_content = re.sub(r'ë¶€ì¹?s*<[^>]*>', '', clean_content)
            clean_content = clean_content.strip()
            
            if clean_content:
                articles.append({
                    'article_number': 'ë¶€ì¹?,
                    'article_title': '',
                    'article_content': clean_content,
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(clean_content.split()),
                    'char_count': len(clean_content),
                    'is_supplementary': True
                })
        
        print(f"ë¶€ì¹?ì¡°ë¬¸ ?? {len(articles)}")
        return articles
    
    def parse_law_document(self, content: str) -> dict:
        """ë²•ë¥  ë¬¸ì„œ ?Œì‹± (ë¶€ì¹??¬í•¨)"""
        try:
            # ?´ìš© ?•ë¦¬
            cleaned_content = self._clean_content(content)
            
            # ê¸°ë³¸ ì¡°ë¬¸ ?Œì‹± (ë¶€ì¹??œì™¸)
            main_content, _ = self._separate_main_and_supplementary(cleaned_content)
            main_articles = self._parse_articles_from_text(main_content)
            
            # ë¶€ì¹??Œì‹±
            supplementary_articles = self._parse_supplementary_articles(cleaned_content)
            
            # ëª¨ë“  ì¡°ë¬¸ ?µí•©
            all_articles = main_articles + supplementary_articles
            
            return {
                'parsing_status': 'success',
                'total_articles': len(all_articles),
                'main_articles': len(main_articles),
                'supplementary_articles': len(supplementary_articles),
                'all_articles': all_articles,
                'main_article_list': main_articles,
                'supplementary_article_list': supplementary_articles,
                'parsing_metadata': {
                    'main_content_length': len(main_content),
                    'total_content_length': len(cleaned_content)
                }
            }
            
        except Exception as e:
            logger.error(f"ë²•ë¥  ë¬¸ì„œ ?Œì‹± ì¤??¤ë¥˜ ë°œìƒ: {e}")
            return {
                'parsing_status': 'error',
                'error_message': str(e),
                'total_articles': 0,
                'main_articles': 0,
                'supplementary_articles': 0,
                'all_articles': [],
                'main_article_list': [],
                'supplementary_article_list': []
            }

def test_enhanced_supplementary_parsing():
    """ê°œì„ ??ë¶€ì¹??Œì‹± ?ŒìŠ¤??""
    
    # ?ŒìŠ¤?¸ìš© ë²•ë¥  ë¬¸ì„œ (ë¶€ì¹??¬í•¨)
    test_content = """??ì¡?ëª©ì ) ??ë²•ì? ?€?œë?êµ?˜ ë²•ì¹˜ì£¼ì˜ë¥?êµ¬í˜„?˜ê¸° ?„í•˜???„ìš”???¬í•­??ê·œì •?¨ì„ ëª©ì ?¼ë¡œ ?œë‹¤.

??ì¡??•ì˜) ??ë²•ì—???¬ìš©?˜ëŠ” ?©ì–´???•ì˜???¤ìŒê³?ê°™ë‹¤.
1. "ë²•ë¥ "?´ë? êµ?šŒ?ì„œ ?œì •??ë²•ì„ ë§í•œ??
2. "ëª…ë ¹"?´ë? ?‰ì •ë¶€?ì„œ ?œì •??ê·œì¹™??ë§í•œ??

??ì¡??ìš© ë²”ìœ„) ??ë²•ì? ?€?œë?êµ??í†  ?´ì—???ìš©?œë‹¤.

ë¶€ì¹?<ë²•ë¥  ??0000?? 2025. 1. 15.>

??ì¡??œí–‰?? ??ë²•ì? ê³µí¬ ??6ê°œì›”??ê²½ê³¼??? ë????œí–‰?œë‹¤.

??ì¡?ê²½ê³¼ì¡°ì¹˜) ??ë²??œí–‰ ?¹ì‹œ ì¢…ì „??ê·œì •???°ë¼ ?‰í•œ ì²˜ë¶„?€ ??ë²•ì— ?°ë¼ ?‰í•œ ê²ƒìœ¼ë¡?ë³¸ë‹¤.

??ì¡??¤ë¥¸ ë²•ë¥ ??ê°œì •) ê·¼ë¡œê¸°ì?ë²??¼ë?ë¥??¤ìŒê³?ê°™ì´ ê°œì •?œë‹¤.
??6ì¡°ì œ1??ì¤?"8?œê°„"??"7?œê°„"?¼ë¡œ ?œë‹¤."""
    
    print("=== ê°œì„ ??ë¶€ì¹??Œì‹± ?ŒìŠ¤??===")
    print("?ë³¸ ?´ìš©:")
    print(test_content)
    print("\n" + "="*80 + "\n")
    
    # ê°œì„ ???Œì„œë¡??ŒìŠ¤??
    parser = SupplementaryEnhancedParser()
    result = parser.parse_law_document(test_content)
    
    print("?Œì‹± ê²°ê³¼:")
    print(f"ì´?ì¡°ë¬¸ ?? {result['total_articles']}")
    print(f"ë³¸ì¹™ ì¡°ë¬¸ ?? {result['main_articles']}")
    print(f"ë¶€ì¹?ì¡°ë¬¸ ?? {result['supplementary_articles']}")
    print()
    
    for i, article in enumerate(result['all_articles']):
        print(f"ì¡°ë¬¸ {i+1}:")
        print(f"  ë²ˆí˜¸: {article['article_number']}")
        print(f"  ?œëª©: {article.get('article_title', 'N/A')}")
        print(f"  ë¶€ì¹??¬ë?: {article.get('is_supplementary', False)}")
        print(f"  ?´ìš©: {article['article_content'][:100]}...")
        print()

def test_real_law_with_supplementary():
    """?¤ì œ ë²•ë¥  ë¬¸ì„œë¡?ë¶€ì¹??Œì‹± ?ŒìŠ¤??""
    
    # ?¤ì œ ë²•ë¥  ë¬¸ì„œ ë¡œë“œ
    test_file = "data/processed/assembly/law/ml_enhanced/20251013/_?€?œë?êµ?ë²•ì›?????œì •??ê´€??ê·œì¹™_assembly_law_1951.json"
    
    if not Path(test_file).exists():
        print(f"?ŒìŠ¤???Œì¼???†ìŠµ?ˆë‹¤: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë³¸ ?´ìš© ?¬êµ¬??(ë¶€ì¹??¬í•¨)
    original_content = """??ì¡?ëª©ì ) ??ê·œì¹™?€ ?€?œë?êµ?ë²•ì›???¬ë²•ì£¼ê¶Œ???Œë³µ??? ì„ ê¸°ë…?˜ê¸° ?„í•˜???ë??œë?êµ?ë²•ì›??? ã€ì„ ?œì •?˜ê³ , ?¬ë²•?…ë¦½ê³?ë²•ì¹˜ì£¼ì˜??ì¤‘ìš”?±ì„ ?Œë¦¬ë©?ê·??˜ì˜ë¥?ê¸°ë…?˜ê¸° ?„í•œ ?‰ì‚¬ ?±ì„ ì§„í–‰?¨ì— ?ˆì–´ ?„ìš”???¬í•­??ê·œì •?¨ì„ ëª©ì ?¼ë¡œ ?œë‹¤.
??ì¡??•ì˜ ë°?ëª…ì¹­) ????ì¡°ì—???¬ë²•ì£¼ê¶Œ???Œë³µ??? ì´???¨ì?, ?¼ì œ???¬ë²•ì£¼ê¶Œ??ë¹¼ì•—ê²¼ë‹¤ê°€ ?€?œë?êµ?´ 1948??9??13??ë¯¸êµ°?•ìœ¼ë¡œë????¬ë²•ê¶Œì„ ?´ì–‘ë°›ìŒ?¼ë¡œ???Œë²•ê¸°ê????€?œë?êµ?ë²•ì›???¤ì§ˆ?ìœ¼ë¡??˜ë¦½??? ì„ ?˜ë??œë‹¤.
???ë??œë?êµ?ë²•ì›??? ã€ì? ë§¤ë…„ 9??13?¼ë¡œ ?œë‹¤.
??ì¡?ê¸°ë…??ë°??‰ì‚¬) ??ë²•ì›?€ ?ë??œë?êµ?ë²•ì›??? ã€ì— ê¸°ë…?ê³¼ ê·¸ì— ë¶€?˜ë˜???‰ì‚¬ë¥??¤ì‹œ?????ˆë‹¤.
??ì¡??¬ìƒ) ???€ë²•ì›?¥ì? ??ì¡°ì œ1??— ê·œì •??ê¸°ë…?¼ì˜ ?˜ì‹?ì„œ ?¬ë²•ë¶€??ë°œì „ ?ëŠ” ë²•ë¥ ë¬¸í™”???¥ìƒ??ê³µí—Œ???‰ì ???œë ·???¬ëŒ?ê²Œ ?¬ìƒ?????ˆë‹¤.
ë¶€ì¹?<??605?? 2015.6.29.>?¼ì¹˜ê¸°ì ‘ê¸?
??ê·œì¹™?€ ê³µí¬??? ë????œí–‰?œë‹¤."""
    
    print("=== ?¤ì œ ë²•ë¥  ë¬¸ì„œ ë¶€ì¹??Œì‹± ?ŒìŠ¤??===")
    
    # ê°œì„ ???Œì„œë¡??ŒìŠ¤??
    parser = SupplementaryEnhancedParser()
    result = parser.parse_law_document(original_content)
    
    print("?Œì‹± ê²°ê³¼:")
    print(f"ì´?ì¡°ë¬¸ ?? {result['total_articles']}")
    print(f"ë³¸ì¹™ ì¡°ë¬¸ ?? {result['main_articles']}")
    print(f"ë¶€ì¹?ì¡°ë¬¸ ?? {result['supplementary_articles']}")
    print()
    
    for i, article in enumerate(result['all_articles']):
        print(f"ì¡°ë¬¸ {i+1}:")
        print(f"  ë²ˆí˜¸: {article['article_number']}")
        print(f"  ?œëª©: {article.get('article_title', 'N/A')}")
        print(f"  ë¶€ì¹??¬ë?: {article.get('is_supplementary', False)}")
        print(f"  ?´ìš©: {article['article_content'][:100]}...")
        print()

if __name__ == "__main__":
    print("ë¶€ì¹??Œì‹± ê°œì„  ?ŒìŠ¤??)
    print("=" * 50)
    
    # ê¸°ë³¸ ?ŒìŠ¤??
    test_enhanced_supplementary_parsing()
    
    print("\n" + "="*80 + "\n")
    
    # ?¤ì œ ë²•ë¥  ë¬¸ì„œ ?ŒìŠ¤??
    test_real_law_with_supplementary()
    
    print("\në¶€ì¹??Œì‹± ê°œì„  ?„ë£Œ!")
