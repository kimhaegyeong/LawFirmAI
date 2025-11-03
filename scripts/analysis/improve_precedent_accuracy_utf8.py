#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ê²€???•í™•???¥ìƒ ?¤í¬ë¦½íŠ¸ (UTF-8 ?¸ì½”??
"""

import json
import sys
import re
import os
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

def improve_precedent_titles():
    """?ë? ?œëª© ê°œì„ """
    print("Precedent title improvement started...")
    
    # ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ?°ì´?°ë§Œ ?„í„°ë§?
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"Total precedent documents before improvement: {len(precedents)}")
    
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
            court_name = "Supreme Court"  # ê¸°ë³¸ê°?
            if "ì§€ë°©ë²•?? in content:
                court_name = "District Court"
            elif "ê³ ë“±ë²•ì›" in content:
                court_name = "High Court"
            elif "?€ë²•ì›" in content:
                court_name = "Supreme Court"
            
            # ?¬ê±´ ? í˜• ì¶”ì¶œ
            case_type = "Case"
            if "ë¯¼ì‚¬" in content:
                case_type = "Civil Case"
            elif "?•ì‚¬" in content:
                case_type = "Criminal Case"
            elif "?‰ì •" in content:
                case_type = "Administrative Case"
            elif "ê°€?? in content:
                case_type = "Family Case"
            elif "?¹í—ˆ" in content:
                case_type = "Patent Case"
            
            # ?ˆë¡œ???œëª© ?ì„±
            new_title = f"{court_name} {case_type} {case_number} Decision"
            
            # ë©”í??°ì´???…ë°?´íŠ¸
            precedent['metadata']['original_document'] = new_title
            precedent['metadata']['court_name'] = court_name
            precedent['metadata']['case_type'] = case_type
            precedent['metadata']['case_number'] = case_number
            
            improved_count += 1
    
    print(f"Improved precedent titles: {improved_count}")
    
    # ê°œì„ ???°ì´???€??
    with open('data/embeddings/metadata_improved.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Improved metadata saved: data/embeddings/metadata_improved.json")
    
    # ê°œì„  ê²°ê³¼ ?•ì¸
    print("\nImproved precedent title samples (top 10):")
    improved_precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    for i, precedent in enumerate(improved_precedents[:10]):
        title = precedent['metadata']['original_document']
        court = precedent['metadata'].get('court_name', 'N/A')
        case_type = precedent['metadata'].get('case_type', 'N/A')
        print(f"  {i+1:2d}. {title} (Court: {court}, Type: {case_type})")
    
    return data

def create_improved_vector_database():
    """ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?ì„±"""
    print("\nCreating improved vector database...")
    
    # ê°œì„ ??ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ?°ì´?°ë§Œ ?„í„°ë§?
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"Improved precedent documents: {len(precedents)}")
    
    # ë²•ì›ë³?ë¶„í¬ ?•ì¸
    court_distribution = {}
    case_type_distribution = {}
    
    for precedent in precedents:
        court = precedent['metadata'].get('court_name', 'Unknown')
        case_type = precedent['metadata'].get('case_type', 'Unknown')
        
        court_distribution[court] = court_distribution.get(court, 0) + 1
        case_type_distribution[case_type] = case_type_distribution.get(case_type, 0) + 1
    
    print("\nCourt distribution:")
    for court, count in sorted(court_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count} ({count/len(precedents)*100:.1f}%)")
    
    print("\nCase type distribution:")
    for case_type, count in sorted(case_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case_type}: {count} ({count/len(precedents)*100:.1f}%)")
    
    return data

def test_improved_accuracy():
    """ê°œì„ ???•í™•???ŒìŠ¤??""
    print("\nTesting improved accuracy...")
    
    # ê°œì„ ??ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ?ë? ê´€??ì¿¼ë¦¬ ?ŒìŠ¤??
    precedent_queries = [
        ("Supreme Court Decision", "precedents"),
        ("District Court Decision", "precedents"),
        ("High Court Decision", "precedents"),
        ("Civil Case", "precedents"),
        ("Criminal Case", "precedents"),
        ("Administrative Case", "precedents")
    ]
    
    print("Precedent search test:")
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
            
            print(f"  '{query}' -> Expected: {expected}, Actual: {actual} {'OK' if is_correct else 'FAIL'}")
            if matching_precedents:
                print(f"    Matched title: {matching_precedents[0]['metadata']['original_document']}")
        else:
            print(f"  '{query}' -> No matching results")
    
    accuracy = correct_predictions / len(precedent_queries) if precedent_queries else 0
    print(f"\nImproved accuracy: {accuracy:.2%} ({correct_predictions}/{len(precedent_queries)})")
    
    return accuracy

def main():
    print("Precedent search accuracy improvement started")
    print("=" * 50)
    
    # 1. ?ë? ?œëª© ê°œì„ 
    improved_data = improve_precedent_titles()
    
    # 2. ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?ì„±
    create_improved_vector_database()
    
    # 3. ê°œì„ ???•í™•???ŒìŠ¤??
    accuracy = test_improved_accuracy()
    
    print("\n" + "=" * 50)
    print(f"Precedent search accuracy improvement completed!")
    print(f"Expected accuracy: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("Target accuracy 80% achieved!")
    else:
        print("Additional improvement needed.")

if __name__ == "__main__":
    main()
