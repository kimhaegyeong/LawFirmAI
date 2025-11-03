#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?•í™•??ë¶€ì¹??Œì‹± êµ¬í˜„
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

class AccurateSupplementaryParser(ImprovedArticleParser):
    """?•í™•??ë¶€ì¹??Œì‹±??êµ¬í˜„??ì¡°ë¬¸ ?Œì„œ"""
    
    def __init__(self):
        super().__init__()
    
    def _find_supplementary_section(self, content: str) -> Optional[str]:
        """ë¶€ì¹??¹ì…˜???•í™•??ì°¾ê¸°"""
        # ë¶€ì¹??œì‘ ?¨í„´??
        supplementary_patterns = [
            r'ë¶€ì¹?s*<[^>]*>?¼ì¹˜ê¸°ì ‘ê¸?s*(.*?)$',
            r'ë¶€ì¹?s*<[^>]*>\s*(.*?)$',
            r'ë¶€ì¹?s*?¼ì¹˜ê¸°ì ‘ê¸?s*(.*?)$',
            r'ë¶€ì¹?s*(.*?)$'
        ]
        
        for pattern in supplementary_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _parse_supplementary_content(self, supplementary_content: str) -> List[Dict[str, Any]]:
        """ë¶€ì¹??´ìš© ?Œì‹±"""
        articles = []
        
        # ë¶€ì¹?ì¡°ë¬¸ ?¨í„´ (??ì¡??œí–‰?? ?•íƒœ)
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
            # ?œí–‰?¼ë§Œ ?ˆëŠ” ê²½ìš°
            if re.search(r'?œí–‰?œë‹¤', supplementary_content):
                articles.append({
                    'article_number': 'ë¶€ì¹?,
                    'article_title': '',
                    'article_content': supplementary_content.strip(),
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(supplementary_content.split()),
                    'char_count': len(supplementary_content),
                    'is_supplementary': True
                })
        
        return articles
    
    def parse_law_document(self, content: str) -> dict:
        """ë²•ë¥  ë¬¸ì„œ ?Œì‹± (ë¶€ì¹??¬í•¨)"""
        try:
            # ?´ìš© ?•ë¦¬
            cleaned_content = self._clean_content(content)
            
            # ë¶€ì¹??¹ì…˜ ì°¾ê¸°
            supplementary_content = self._find_supplementary_section(cleaned_content)
            
            if supplementary_content:
                print(f"ë¶€ì¹??¹ì…˜ ë°œê²¬!")
                print(f"ë¶€ì¹??´ìš©: {supplementary_content[:200]}...")
                
                # ë¶€ì¹??Œì‹±
                supplementary_articles = self._parse_supplementary_content(supplementary_content)
                
                # ë³¸ì¹™ ?´ìš©?ì„œ ë¶€ì¹??œê±°
                main_content = re.sub(r'ë¶€ì¹?s*<[^>]*>?¼ì¹˜ê¸°ì ‘ê¸?*$', '', cleaned_content, flags=re.DOTALL)
                main_content = re.sub(r'ë¶€ì¹?s*<[^>]*>.*$', '', main_content, flags=re.DOTALL)
                main_content = re.sub(r'ë¶€ì¹?s*?¼ì¹˜ê¸°ì ‘ê¸?*$', '', main_content, flags=re.DOTALL)
                main_content = re.sub(r'ë¶€ì¹?s*.*$', '', main_content, flags=re.DOTALL)
                
                # ë³¸ì¹™ ì¡°ë¬¸ ?Œì‹±
                main_articles = self._parse_articles_from_text(main_content)
            else:
                print("ë¶€ì¹??¹ì…˜??ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
                # ?„ì²´ ?´ìš©??ë³¸ì¹™?¼ë¡œ ?Œì‹±
                main_articles = self._parse_articles_from_text(cleaned_content)
                supplementary_articles = []
            
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
                    'main_content_length': len(main_content) if supplementary_content else len(cleaned_content),
                    'supplementary_content_length': len(supplementary_content) if supplementary_content else 0,
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

def test_accurate_supplementary_parsing():
    """?•í™•??ë¶€ì¹??Œì‹± ?ŒìŠ¤??""
    
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
    
    print("=== ?•í™•??ë¶€ì¹??Œì‹± ?ŒìŠ¤??===")
    print("?ë³¸ ?´ìš©:")
    print(test_content)
    print("\n" + "="*80 + "\n")
    
    # ?•í™•???Œì„œë¡??ŒìŠ¤??
    parser = AccurateSupplementaryParser()
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
    
    # ?•í™•???Œì„œë¡??ŒìŠ¤??
    parser = AccurateSupplementaryParser()
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
    print("?•í™•??ë¶€ì¹??Œì‹± ?ŒìŠ¤??)
    print("=" * 50)
    
    # ê¸°ë³¸ ?ŒìŠ¤??
    test_accurate_supplementary_parsing()
    
    print("\n" + "="*80 + "\n")
    
    # ?¤ì œ ë²•ë¥  ë¬¸ì„œ ?ŒìŠ¤??
    test_real_law_with_supplementary()
    
    print("\n?•í™•??ë¶€ì¹??Œì‹± ?„ë£Œ!")
