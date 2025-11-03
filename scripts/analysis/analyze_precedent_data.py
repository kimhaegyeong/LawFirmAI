#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?°ì´??ë¶„ì„ ë°?ê°œì„  ë°©ì•ˆ ?„ì¶œ
"""

import json
import sys
from pathlib import Path

def analyze_precedent_data():
    """?ë? ?°ì´??ë¶„ì„"""
    print("?ë? ?°ì´??ë¶„ì„ ?œì‘...")
    
    # ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ?°ì´?°ë§Œ ?„í„°ë§?
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"ì´??ë? ë¬¸ì„œ ?? {len(precedents)}")
    print("\n?ë? ?œëª© ?˜í”Œ (?ìœ„ 20ê°?:")
    
    for i, precedent in enumerate(precedents[:20]):
        title = precedent['metadata']['original_document']
        print(f"  {i+1:2d}. {title}")
    
    # ?œëª© ?¨í„´ ë¶„ì„
    print("\n?œëª© ?¨í„´ ë¶„ì„:")
    patterns = {}
    for precedent in precedents:
        title = precedent['metadata']['original_document']
        if not title or title.strip() == "":
            patterns['ë¹??œëª©'] = patterns.get('ë¹??œëª©', 0) + 1
        elif '?€ë²•ì›' in title:
            patterns['?€ë²•ì›'] = patterns.get('?€ë²•ì›', 0) + 1
        elif 'ì§€ë°©ë²•?? in title:
            patterns['ì§€ë°©ë²•??] = patterns.get('ì§€ë°©ë²•??, 0) + 1
        elif 'ê³ ë“±ë²•ì›' in title:
            patterns['ê³ ë“±ë²•ì›'] = patterns.get('ê³ ë“±ë²•ì›', 0) + 1
        elif '?ê²°' in title:
            patterns['?ê²°'] = patterns.get('?ê²°', 0) + 1
        elif '?ë?' in title:
            patterns['?ë?'] = patterns.get('?ë?', 0) + 1
        else:
            patterns['ê¸°í?'] = patterns.get('ê¸°í?', 0) + 1
    
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}ê°?({count/len(precedents)*100:.1f}%)")
    
    # ë¹??œëª© ë¬¸ì œ ë¶„ì„
    empty_titles = [p for p in precedents if not p['metadata']['original_document'] or p['metadata']['original_document'].strip() == ""]
    print(f"\në¹??œëª© ë¬¸ì œ:")
    print(f"  ë¹??œëª© ë¬¸ì„œ ?? {len(empty_titles)}ê°?)
    
    if empty_titles:
        print("  ë¹??œëª© ë¬¸ì„œ ?˜í”Œ:")
        for i, precedent in enumerate(empty_titles[:5]):
            print(f"    {i+1}. ID: {precedent['id']}")
            print(f"       ?´ìš© ë¯¸ë¦¬ë³´ê¸°: {precedent['text'][:100]}...")
    
    # ê°œì„  ë°©ì•ˆ ?œì‹œ
    print("\nê°œì„  ë°©ì•ˆ:")
    print("1. ë¹??œëª© ë¬¸ì œ ?´ê²°:")
    print("   - ?ë? ?°ì´?°ì—??case_name ?ëŠ” title ?„ë“œ ?œìš©")
    print("   - ?ë? IDë¥?ê¸°ë°˜?¼ë¡œ ?œëª© ?ì„±")
    print("   - ?ë? ?´ìš©?ì„œ ì²?ë¬¸ì¥???œëª©?¼ë¡œ ?¬ìš©")
    
    print("2. ?œëª© ?•ê·œ??")
    print("   - '?€ë²•ì›', 'ì§€ë°©ë²•??, 'ê³ ë“±ë²•ì›' ?¤ì›Œ??ê°•í™”")
    print("   - ?ë? ê´€???¤ì›Œ??ì¶”ê? ('?ê²°', '?ë?', '?¬ê±´' ??")
    print("   - ë²•ì›ëª??œì???)
    
    print("3. ?„ë² ??ê°œì„ :")
    print("   - ?ë? ?¹í™” ?¤ì›Œ??ì¶”ê?")
    print("   - ë²•ì›ëª…ê³¼ ?¬ê±´ë²ˆí˜¸ ?•ë³´ ê°•í™”")
    print("   - ?ë? ?´ìš©??ë²•ì  ë§¥ë½ ê°•ì¡°")

if __name__ == "__main__":
    analyze_precedent_data()
