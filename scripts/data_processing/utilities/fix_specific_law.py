#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?¹ì • ë²•ë¥  ë¬¸ì„œ??ì¡°ë¬¸ ?Œì‹± ë¬¸ì œ ?˜ì • ?¤í¬ë¦½íŠ¸
??ì¡?2???„ë½ ë¬¸ì œ ?´ê²°
"""

import json
import sys
import os
from pathlib import Path
import logging

# Windows ì½˜ì†”?ì„œ UTF-8 ?¸ì½”???¤ì •
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# ?Œì„œ ëª¨ë“ˆ ê²½ë¡œ ì¶”ê?
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_specific_law_file():
    """?¹ì • ë²•ë¥  ?Œì¼ ?˜ì •"""
    
    # ?ë³¸ ?°ì´??(?„ì „???´ìš©)
    original_content = """??ì¡?ëª©ì ) ??ê·œì¹™?€ ?€?œë?êµ?ë²•ì›???¬ë²•ì£¼ê¶Œ???Œë³µ??? ì„ ê¸°ë…?˜ê¸° ?„í•˜???ë??œë?êµ?ë²•ì›??? ã€ì„ ?œì •?˜ê³ , ?¬ë²•?…ë¦½ê³?ë²•ì¹˜ì£¼ì˜??ì¤‘ìš”?±ì„ ?Œë¦¬ë©?ê·??˜ì˜ë¥?ê¸°ë…?˜ê¸° ?„í•œ ?‰ì‚¬ ?±ì„ ì§„í–‰?¨ì— ?ˆì–´ ?„ìš”???¬í•­??ê·œì •?¨ì„ ëª©ì ?¼ë¡œ ?œë‹¤.
??ì¡??•ì˜ ë°?ëª…ì¹­) ????ì¡°ì—???¬ë²•ì£¼ê¶Œ???Œë³µ??? ì´???¨ì?, ?¼ì œ???¬ë²•ì£¼ê¶Œ??ë¹¼ì•—ê²¼ë‹¤ê°€ ?€?œë?êµ?´ 1948??9??13??ë¯¸êµ°?•ìœ¼ë¡œë????¬ë²•ê¶Œì„ ?´ì–‘ë°›ìŒ?¼ë¡œ???Œë²•ê¸°ê????€?œë?êµ?ë²•ì›???¤ì§ˆ?ìœ¼ë¡??˜ë¦½??? ì„ ?˜ë??œë‹¤.
???ë??œë?êµ?ë²•ì›??? ã€ì? ë§¤ë…„ 9??13?¼ë¡œ ?œë‹¤.
??ì¡?ê¸°ë…??ë°??‰ì‚¬) ??ë²•ì›?€ ?ë??œë?êµ?ë²•ì›??? ã€ì— ê¸°ë…?ê³¼ ê·¸ì— ë¶€?˜ë˜???‰ì‚¬ë¥??¤ì‹œ?????ˆë‹¤.
??ì¡??¬ìƒ) ???€ë²•ì›?¥ì? ??ì¡°ì œ1??— ê·œì •??ê¸°ë…?¼ì˜ ?˜ì‹?ì„œ ?¬ë²•ë¶€??ë°œì „ ?ëŠ” ë²•ë¥ ë¬¸í™”???¥ìƒ??ê³µí—Œ???‰ì ???œë ·???¬ëŒ?ê²Œ ?¬ìƒ?????ˆë‹¤.
ë¶€ì¹?<??605?? 2015.6.29.>?¼ì¹˜ê¸°ì ‘ê¸?
??ê·œì¹™?€ ê³µí¬??? ë????œí–‰?œë‹¤."""
    
    # ê·œì¹™ ê¸°ë°˜ ?Œì„œë¡??¬ë°”ë¥??Œì‹±
    parser = ImprovedArticleParser()
    result = parser.parse_law_document(original_content)
    
    # ë©”í??°ì´??êµ¬ì„±
    fixed_data = {
        "law_id": "assembly_law_1951",
        "law_name": "?Œë??œë?êµ?ë²•ì›??? ã€ì œ?•ì— ê´€??ê·œì¹™",
        "law_type": "?€ë²•ì›ê·œì¹™",
        "category": "????ë²•ì›?‰ì •",
        "promulgation_number": "??605??,
        "promulgation_date": "2015.6.29",
        "enforcement_date": "2015.6.29",
        "amendment_type": "?œì •",
        "ministry": "",
        "articles": result['all_articles']
    }
    
    # ?˜ì •???Œì¼ ?€??
    output_path = "data/processed/assembly/law/ml_enhanced/20251013/_?€?œë?êµ?ë²•ì›?????œì •??ê´€??ê·œì¹™_assembly_law_1951.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=4)
    
    print(f"?˜ì •???Œì¼ ?€?? {output_path}")
    print(f"ì´?ì¡°ë¬¸ ?? {len(fixed_data['articles'])}")
    
    # ê°?ì¡°ë¬¸ ?•ì¸
    for i, article in enumerate(fixed_data['articles']):
        print(f"\nì¡°ë¬¸ {i+1}:")
        print(f"  ë²ˆí˜¸: {article['article_number']}")
        print(f"  ?œëª©: {article.get('article_title', 'N/A')}")
        print(f"  ???? {len(article.get('sub_articles', []))}")
        
        if article.get('sub_articles'):
            for j, sub in enumerate(article['sub_articles']):
                print(f"    ??{j+1}: {sub['content'][:50]}...")
    
    return fixed_data

def validate_fix():
    """?˜ì • ê²°ê³¼ ê²€ì¦?""
    
    file_path = "data/processed/assembly/law/ml_enhanced/20251013/_?€?œë?êµ?ë²•ì›?????œì •??ê´€??ê·œì¹™_assembly_law_1951.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n=== ?˜ì • ê²°ê³¼ ê²€ì¦?===")
    print(f"ì´?ì¡°ë¬¸ ?? {len(data['articles'])}")
    
    # ??ì¡??•ì¸
    article2 = None
    for article in data['articles']:
        if article['article_number'] == '??ì¡?:
            article2 = article
            break
    
    if article2:
        print(f"\n??ì¡??•ì¸:")
        print(f"  ?œëª©: {article2.get('article_title', 'N/A')}")
        print(f"  ???? {len(article2.get('sub_articles', []))}")
        
        if article2.get('sub_articles'):
            for i, sub in enumerate(article2['sub_articles']):
                print(f"  ??{i+1}: {sub['content']}")
        
        # ??ì¡?2??´ ?ˆëŠ”ì§€ ?•ì¸
        has_paragraph2 = any('ë§¤ë…„ 9??13?? in sub['content'] for sub in article2.get('sub_articles', []))
        if has_paragraph2:
            print("\n[OK] ??ì¡?2??´ ?¬ë°”ë¥´ê²Œ ?¬í•¨?˜ì—ˆ?µë‹ˆ??")
        else:
            print("\n[ERROR] ??ì¡?2??´ ?¬ì „???„ë½?˜ì—ˆ?µë‹ˆ??")
    else:
        print("\n[ERROR] ??ì¡°ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤.")

if __name__ == "__main__":
    print("?¹ì • ë²•ë¥  ë¬¸ì„œ ì¡°ë¬¸ ?Œì‹± ë¬¸ì œ ?˜ì •")
    print("=" * 50)
    
    # ?Œì¼ ?˜ì •
    fixed_data = fix_specific_law_file()
    
    # ?˜ì • ê²°ê³¼ ê²€ì¦?
    validate_fix()
    
    print("\n?˜ì • ?„ë£Œ!")
