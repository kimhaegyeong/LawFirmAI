#!/usr/bin/env python3
"""
ë²•ë¥  ?°ì´???•ì¶• ? í‹¸ë¦¬í‹° ëª¨ë“ˆ
?˜ì§‘ ë°??„ì²˜ë¦?ê³¼ì •?ì„œ ?°ì´?°ë? ì¦‰ì‹œ ?•ì¶•?˜ëŠ” ê¸°ëŠ¥ ?œê³µ
"""

import json
import re
from typing import Dict, Any, List

def compress_legal_text(text: str) -> str:
    """ë²•ë¥  ?ìŠ¤???•ì¶•"""
    if not text:
        return ""
    
    # ë¶ˆí•„?”í•œ ê³µë°± ?œê±°
    text = re.sub(r'\s+', ' ', text)
    
    # ë°˜ë³µ?˜ëŠ” ë²•ë¥  ?©ì–´ ì¶•ì•½
    replacements = {
        '??ë²•ì— ?°ë¥´ë©?: '??ë²•ì— ?°ë¼',
        '?¤ìŒ ê°??¸ì˜ ?´ëŠ ?˜ë‚˜???´ë‹¹?˜ëŠ”': '?¤ìŒ???´ë‹¹?˜ëŠ”',
        '?¹ë³„?œì¥Â·ê´‘ì—­?œì¥Â·?¹ë³„?ì¹˜?œì¥Â·?„ì???: '?œÂ·ë„ì§€??,
        '?¹ë³„?ì¹˜?„ì???: '?¹ë³„?ì¹˜?„ì???,
        'ì¤‘ì•™?‰ì •ê¸°ê?????: 'ì¤‘ì•™?‰ì •ê¸°ê???,
        'ì§€ë°©ìì¹˜ë‹¨ì²´ì˜ ??: 'ì§€ë°©ìì¹˜ë‹¨ì²´ì¥',
        'êµ?? ?ëŠ” ì§€ë°©ìì¹˜ë‹¨ì²?: 'êµ??Â·ì§€ë°©ìì¹˜ë‹¨ì²?,
        '?´í•˜ "?œÂ·ë„ì§€?????œë‹¤': '?´í•˜ ?œÂ·ë„ì§€?¬ë¼ ??,
        '?´í•˜ "?¹ë??????œë‹¤': '?´í•˜ ?¹ë??œë¼ ??,
        '?´í•˜ "?±ë¡ë¹„ì˜ë¦¬ë?ê°„ë‹¨ì²????œë‹¤': '?´í•˜ ?±ë¡ë¹„ì˜ë¦¬ë?ê°„ë‹¨ì²´ë¼ ??,
        '?´í•˜ "ê³µìµ?¬ì—…"?´ë¼ ?œë‹¤': '?´í•˜ ê³µìµ?¬ì—…?´ë¼ ??
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

# generate_compressed_search_text ?¨ìˆ˜ ?œê±°??- ???´ìƒ ?¬ìš©?˜ì? ?ŠìŒ

def compress_law_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """ê°œë³„ ë²•ë¥  ?°ì´???•ì¶•"""
    # ?„ìˆ˜ ?„ë“œë§?? ì?
    compressed = {
        'law_id': data.get('law_id'),
        'law_name': data.get('law_name'),
        'law_type': data.get('law_type'),
        'category': data.get('category'),
        'promulgation_number': data.get('promulgation_number'),
        'promulgation_date': data.get('promulgation_date'),
        'enforcement_date': data.get('enforcement_date'),
        'amendment_type': data.get('amendment_type'),
        'ministry': data.get('ministry'),
        'articles': data.get('articles', [])
    }
    
    # articles ?´ë? ?ìŠ¤?¸ë„ ?•ì¶•
    for article in compressed['articles']:
        if 'article_content' in article:
            article['article_content'] = compress_legal_text(article['article_content'])
        
        # sub_articles???•ì¶•
        for sub_article in article.get('sub_articles', []):
            if 'content' in sub_article:
                sub_article['content'] = compress_legal_text(sub_article['content'])
    
    return compressed

def save_compressed_law_data(data: Dict[str, Any], file_path: str) -> int:
    """?•ì¶•??ë²•ë¥  ?°ì´?°ë? ?Œì¼???€?¥í•˜ê³??Œì¼ ?¬ê¸° ë°˜í™˜"""
    compressed_data = compress_law_data(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(compressed_data, f, ensure_ascii=False, separators=(',', ':'))
    
    # ?Œì¼ ?¬ê¸° ë°˜í™˜
    import os
    return os.path.getsize(file_path)

def compress_and_save_page_data(page_data: Dict[str, Any], file_path: str) -> int:
    """?˜ì´ì§€ ?°ì´?°ë? ?•ì¶•?˜ì—¬ ?€??""
    compressed_page_data = {
        'page_number': page_data.get('page_number'),
        'total_pages': page_data.get('total_pages'),
        'laws_count': page_data.get('laws_count'),
        'collected_at': page_data.get('collected_at'),
        'laws': []
    }
    
    # ê°?ë²•ë¥  ?°ì´???•ì¶•
    for law in page_data.get('laws', []):
        compressed_law = compress_law_data(law)
        compressed_page_data['laws'].append(compressed_law)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(compressed_page_data, f, ensure_ascii=False, separators=(',', ':'))
    
    # ?Œì¼ ?¬ê¸° ë°˜í™˜
    import os
    return os.path.getsize(file_path)
