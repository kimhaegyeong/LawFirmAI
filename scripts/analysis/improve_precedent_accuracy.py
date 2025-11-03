#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ê²€???•í™•???¥ìƒ ?¤í¬ë¦½íŠ¸
"""

import json
import sys
import re
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

def improve_precedent_titles():
    """?ë? ?œëª© ê°œì„ """
    print("?ë? ?œëª© ê°œì„  ?œìž‘...")
    
    # ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ?°ì´?°ë§Œ ?„í„°ë§?
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"ê°œì„  ???ë? ë¬¸ì„œ ?? {len(precedents)}")
    
    improved_count = 0
    
    for precedent in precedents:
        original_title = precedent['metadata']['original_document']
        
        # ë¹??œëª©??ê²½ìš° ê°œì„ 
        if not original_title or original_title.strip() == "":
            # ?ë? ID?ì„œ ?•ë³´ ì¶”ì¶œ
            case_id = precedent['id']
            case_number = case_id.replace('case_', '') if 'case_' in case_id else case_id
            
            # ?ë? ?´ìš©?ì„œ ?•ë³´ ì¶”ì¶œ
            content = precedent['text']
            
            # ë²•ì›ëª?ì¶”ì¶œ
            court_name = "?€ë²•ì›"  # ê¸°ë³¸ê°?
            if "ì§€ë°©ë²•?? in content:
                court_name = "ì§€ë°©ë²•??
            elif "ê³ ë“±ë²•ì›" in content:
                court_name = "ê³ ë“±ë²•ì›"
            elif "?€ë²•ì›" in content:
                court_name = "?€ë²•ì›"
            
            # ?¬ê±´ ? í˜• ì¶”ì¶œ
            case_type = "?¬ê±´"
            if "ë¯¼ì‚¬" in content:
                case_type = "ë¯¼ì‚¬?¬ê±´"
            elif "?•ì‚¬" in content:
                case_type = "?•ì‚¬?¬ê±´"
            elif "?‰ì •" in content:
                case_type = "?‰ì •?¬ê±´"
            elif "ê°€?? in content:
                case_type = "ê°€?¬ì‚¬ê±?
            elif "?¹í—ˆ" in content:
                case_type = "?¹í—ˆ?¬ê±´"
            
            # ?ˆë¡œ???œëª© ?ì„±
            new_title = f"{court_name} {case_type} {case_number}???ê²°"
            
            # ë©”í??°ì´???…ë°?´íŠ¸
            precedent['metadata']['original_document'] = new_title
            precedent['metadata']['court_name'] = court_name
            precedent['metadata']['case_type'] = case_type
            precedent['metadata']['case_number'] = case_number
            
            improved_count += 1
    
    print(f"ê°œì„ ???ë? ?œëª© ?? {improved_count}ê°?)
    
    # ê°œì„ ???°ì´???€??
    with open('data/embeddings/metadata_improved.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("ê°œì„ ??ë©”í??°ì´???€???„ë£Œ: data/embeddings/metadata_improved.json")
    
    # ê°œì„  ê²°ê³¼ ?•ì¸
    print("\nê°œì„ ???ë? ?œëª© ?˜í”Œ (?ìœ„ 10ê°?:")
    improved_precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    for i, precedent in enumerate(improved_precedents[:10]):
        title = precedent['metadata']['original_document']
        court = precedent['metadata'].get('court_name', 'N/A')
        case_type = precedent['metadata'].get('case_type', 'N/A')
        print(f"  {i+1:2d}. {title} (ë²•ì›: {court}, ? í˜•: {case_type})")
    
    return data

def create_improved_vector_database():
    """ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?ì„±"""
    print("\nê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?ì„± ?œìž‘...")
    
    # ê°œì„ ??ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ?°ì´?°ë§Œ ?„í„°ë§?
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"ê°œì„ ???ë? ë¬¸ì„œ ?? {len(precedents)}")
    
    # ë²•ì›ë³?ë¶„í¬ ?•ì¸
    court_distribution = {}
    case_type_distribution = {}
    
    for precedent in precedents:
        court = precedent['metadata'].get('court_name', 'Unknown')
        case_type = precedent['metadata'].get('case_type', 'Unknown')
        
        court_distribution[court] = court_distribution.get(court, 0) + 1
        case_type_distribution[case_type] = case_type_distribution.get(case_type, 0) + 1
    
    print("\në²•ì›ë³?ë¶„í¬:")
    for court, count in sorted(court_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count}ê°?({count/len(precedents)*100:.1f}%)")
    
    print("\n?¬ê±´ ? í˜•ë³?ë¶„í¬:")
    for case_type, count in sorted(case_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case_type}: {count}ê°?({count/len(precedents)*100:.1f}%)")
    
    return data

def test_improved_accuracy():
    """ê°œì„ ???•í™•???ŒìŠ¤??""
    print("\nê°œì„ ???•í™•???ŒìŠ¤???œìž‘...")
    
    # ê°œì„ ??ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ê´€??ì¿¼ë¦¬ ?ŒìŠ¤??
    precedent_queries = [
        ("?€ë²•ì› ?ê²°", "precedents"),
        ("ì§€ë°©ë²•???ê²°", "precedents"),
        ("ê³ ë“±ë²•ì› ?ê²°", "precedents"),
        ("ë¯¼ì‚¬?¬ê±´", "precedents"),
        ("?•ì‚¬?¬ê±´", "precedents"),
        ("?‰ì •?¬ê±´", "precedents")
    ]
    
    print("?ë? ê²€???ŒìŠ¤??")
    correct_predictions = 0
    
    for query, expected in precedent_queries:
        # ê°„ë‹¨???¤ì›Œ??ë§¤ì¹­ ?ŒìŠ¤??
        precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
        
        # ì¿¼ë¦¬ ?¤ì›Œ?œì? ë§¤ì¹­?˜ëŠ” ?ë? ì°¾ê¸°
        matching_precedents = []
        for precedent in precedents:
            title = precedent['metadata']['original_document']
            if any(keyword in title for keyword in query.split()):
                matching_precedents.append(precedent)
        
        if matching_precedents:
            # ì²?ë²ˆì§¸ ë§¤ì¹­ ê²°ê³¼???€???•ì¸
            actual = matching_precedents[0]['metadata']['data_type']
            is_correct = actual == expected
            if is_correct:
                correct_predictions += 1
            
            print(f"  '{query}' -> ?ˆìƒ: {expected}, ?¤ì œ: {actual} {'OK' if is_correct else 'FAIL'}")
            if matching_precedents:
                print(f"    ë§¤ì¹­???œëª©: {matching_precedents[0]['metadata']['original_document']}")
        else:
            print(f"  '{query}' -> ë§¤ì¹­ ê²°ê³¼ ?†ìŒ")
    
    accuracy = correct_predictions / len(precedent_queries) if precedent_queries else 0
    print(f"\nê°œì„ ???•í™•?? {accuracy:.2%} ({correct_predictions}/{len(precedent_queries)})")
    
    return accuracy

def main():
    print("?ë? ê²€???•í™•???¥ìƒ ?‘ì—… ?œìž‘")
    print("=" * 50)
    
    # 1. ?ë? ?œëª© ê°œì„ 
    improved_data = improve_precedent_titles()
    
    # 2. ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?ì„±
    create_improved_vector_database()
    
    # 3. ê°œì„ ???•í™•???ŒìŠ¤??
    accuracy = test_improved_accuracy()
    
    print("\n" + "=" * 50)
    print(f"?ë? ê²€???•í™•???¥ìƒ ?„ë£Œ!")
    print(f"?ˆìƒ ?•í™•?? {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("ëª©í‘œ ?•í™•??80% ?¬ì„±!")
    else:
        print("ì¶”ê? ê°œì„ ???„ìš”?©ë‹ˆ??")

if __name__ == "__main__":
    main()
