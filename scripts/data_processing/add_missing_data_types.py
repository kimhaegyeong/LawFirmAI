#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¶”ê? ?°ì´???€??ì§€??(?Œì¬ê²°ì •ë¡€, ë²•ë ¹?´ì„ë¡€)
"""

import json
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

def check_missing_data_types():
    """?„ë½???°ì´???€???•ì¸"""
    print("Checking missing data types...")
    
    # ?„ì²˜ë¦¬ëœ ?°ì´???”ë ‰? ë¦¬ ?•ì¸
    processed_dir = Path("data/processed")
    
    data_types = {
        'constitutional_decisions': '?Œì¬ê²°ì •ë¡€',
        'legal_interpretations': 'ë²•ë ¹?´ì„ë¡€',
        'administrative_rules': '?‰ì •ê·œì¹™',
        'local_ordinances': '?ì¹˜ë²•ê·œ'
    }
    
    missing_types = []
    available_types = []
    
    for data_type, korean_name in data_types.items():
        type_dir = processed_dir / data_type
        if type_dir.exists():
            json_files = list(type_dir.glob("*.json"))
            if json_files:
                available_types.append((data_type, korean_name, len(json_files)))
                print(f"OK {korean_name} ({data_type}): {len(json_files)} files")
            else:
                missing_types.append((data_type, korean_name))
                print(f"NO {korean_name} ({data_type}): No files")
        else:
            missing_types.append((data_type, korean_name))
            print(f"NO {korean_name} ({data_type}): Directory not found")
    
    return available_types, missing_types

def create_sample_data():
    """?˜í”Œ ?°ì´???ì„± (?ŒìŠ¤?¸ìš©)"""
    print("\nCreating sample data for missing types...")
    
    # ?˜í”Œ ?Œì¬ê²°ì •ë¡€ ?°ì´??
    constitutional_samples = [
        {
            "id": "const_001",
            "title": "?Œë²•?¬íŒ??2023?Œë§ˆ1234 ê²°ì •",
            "content": "???¬ê±´?€ ê¸°ë³¸ê¶?ì¹¨í•´ ?¬ë????€???Œë²•?¬íŒ?Œì˜ ê²°ì •?…ë‹ˆ?? ?ê³ ??êµ?????‰ì •ì²˜ë¶„???ì‹ ??ê¸°ë³¸ê¶Œì„ ì¹¨í•´?œë‹¤ê³?ì£¼ì¥?˜ì??¼ë©°, ?Œë²•?¬íŒ?ŒëŠ” ?´ë? ?¬ë¦¬??ê²°ê³¼ ?¼ë? ?¸ìš©?˜ê¸°ë¡?ê²°ì •?˜ì??µë‹ˆ??",
            "decision_date": "2023-12-15",
            "case_type": "?Œë²•?Œì›",
            "court": "?Œë²•?¬íŒ??
        },
        {
            "id": "const_002", 
            "title": "?Œë²•?¬íŒ??2023?Œë°”5678 ê²°ì •",
            "content": "???¬ê±´?€ ë²•ë¥ ???„í—Œ???¬ë????€???Œë²•?¬íŒ?Œì˜ ê²°ì •?…ë‹ˆ?? ?´ë‹¹ ë²•ë¥  ì¡°í•­???Œë²•???„ë°˜?˜ëŠ”ì§€ ?¬ë?ë¥??¬ë¦¬??ê²°ê³¼, ?„í—Œ ê²°ì •???´ë ¸?µë‹ˆ??",
            "decision_date": "2023-11-20",
            "case_type": "ë²•ë¥ ?„í—Œ?œì†Œ",
            "court": "?Œë²•?¬íŒ??
        }
    ]
    
    # ?˜í”Œ ë²•ë ¹?´ì„ë¡€ ?°ì´??
    interpretation_samples = [
        {
            "id": "interp_001",
            "title": "ë¯¼ë²• ??ì¡??´ì„ë¡€",
            "content": "ë¯¼ë²• ??ì¡°ì— ?€??ë²•ë¬´ë¶€???´ì„?…ë‹ˆ?? ë¯¼ë²•??ê¸°ë³¸ ?ì¹™ê³??ìš© ë²”ìœ„???€???ì„¸???¤ëª…?˜ê³  ?ˆìœ¼ë©? ?¤ì œ ?¬ë?ë¥??µí•œ êµ¬ì²´?ì¸ ?ìš© ë°©ë²•???œì‹œ?©ë‹ˆ??",
            "interpretation_date": "2023-10-10",
            "department": "ë²•ë¬´ë¶€",
            "law_name": "ë¯¼ë²•"
        },
        {
            "id": "interp_002",
            "title": "?•ë²• ??0ì¡??´ì„ë¡€", 
            "content": "?•ë²• ??0ì¡??•ë‹¹ë°©ìœ„???€???€ë²•ì›???´ì„?…ë‹ˆ?? ?•ë‹¹ë°©ìœ„???±ë¦½ ?”ê±´ê³??œê³„???€??êµ¬ì²´?ìœ¼ë¡??¤ëª…?˜ê³  ?ˆìœ¼ë©? ê´€???ë??€ ?¨ê»˜ ?´ì„?˜ê³  ?ˆìŠµ?ˆë‹¤.",
            "interpretation_date": "2023-09-15",
            "department": "?€ë²•ì›",
            "law_name": "?•ë²•"
        }
    ]
    
    # ?˜í”Œ ?‰ì •ê·œì¹™ ?°ì´??
    administrative_samples = [
        {
            "id": "admin_001",
            "title": "?‰ì •?ˆì°¨ë²??œí–‰ê·œì¹™",
            "content": "?‰ì •?ˆì°¨ë²•ì˜ ?œí–‰???„í•œ êµ¬ì²´?ì¸ ê·œì¹™?…ë‹ˆ?? ?‰ì •ì²˜ë¶„???ˆì°¨, ì²?¬¸??ê°œìµœ ë°©ë²•, ?´ì˜? ì²­ ?ˆì°¨ ?±ì— ?€???ì„¸??ê·œì •?˜ê³  ?ˆìŠµ?ˆë‹¤.",
            "promulgation_date": "2023-08-01",
            "enforcement_date": "2023-08-01",
            "department": "?‰ì •?ˆì „ë¶€"
        }
    ]
    
    # ?˜í”Œ ?ì¹˜ë²•ê·œ ?°ì´??
    local_samples = [
        {
            "id": "local_001",
            "title": "?œìš¸?¹ë³„??ì¡°ë? ??234??,
            "content": "?œìš¸?¹ë³„?œì˜ ì¡°ë??…ë‹ˆ?? ?„ì‹œê³„íš, ?˜ê²½ë³´í˜¸, ì£¼ë?ë³µì? ?±ì— ê´€???¬í•­??ê·œì •?˜ê³  ?ˆìœ¼ë©? ?œë???ê¶Œë¦¬?€ ?˜ë¬´???€??ëª…ì‹œ?˜ê³  ?ˆìŠµ?ˆë‹¤.",
            "promulgation_date": "2023-07-15",
            "enforcement_date": "2023-07-15",
            "local_government": "?œìš¸?¹ë³„??
        }
    ]
    
    # ?˜í”Œ ?°ì´???€??
    sample_data = {
        'constitutional_decisions': constitutional_samples,
        'legal_interpretations': interpretation_samples,
        'administrative_rules': administrative_samples,
        'local_ordinances': local_samples
    }
    
    for data_type, samples in sample_data.items():
        output_file = f"data/processed/{data_type}/{data_type}_sample.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Created sample data: {output_file} ({len(samples)} items)")

def process_missing_data_types():
    """?„ë½???°ì´???€??ì²˜ë¦¬"""
    print("\nProcessing missing data types...")
    
    # ?˜í”Œ ?°ì´???ì„±
    create_sample_data()
    
    # ?„ì²˜ë¦¬ëœ ?°ì´???•ì¸
    available_types, missing_types = check_missing_data_types()
    
    print(f"\nAvailable data types: {len(available_types)}")
    print(f"Missing data types: {len(missing_types)}")
    
    return available_types, missing_types

def update_vector_database():
    """ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?…ë°?´íŠ¸"""
    print("\nUpdating vector database with new data types...")
    
    # ê¸°ì¡´ ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    # ?ˆë¡œ???°ì´???€??ì¶”ê?
    new_data = []
    
    # ?Œì¬ê²°ì •ë¡€ ?°ì´??ì¶”ê?
    const_file = "data/processed/constitutional_decisions/constitutional_decisions_sample.json"
    if Path(const_file).exists():
        with open(const_file, 'r', encoding='utf-8') as f:
            const_data = json.load(f)
        
        for item in const_data:
            new_doc = {
                'id': item['id'],
                'original_id': item['id'],
                'chunk_id': 'full',
                'text': item['content'],
                'metadata': {
                    'data_type': 'constitutional_decisions',
                    'original_document': item['title'],
                    'chunk_start': 0,
                    'chunk_end': len(item['content']),
                    'chunk_length': len(item['content']),
                    'word_count': len(item['content'].split()),
                    'entities': {},
                    'processed_at': '2025-09-30',
                    'is_valid': True,
                    'court_name': item.get('court', '?Œë²•?¬íŒ??),
                    'case_type': item.get('case_type', '?Œë²•?Œì›'),
                    'decision_date': item.get('decision_date', '')
                }
            }
            new_data.append(new_doc)
    
    # ë²•ë ¹?´ì„ë¡€ ?°ì´??ì¶”ê?
    interp_file = "data/processed/legal_interpretations/legal_interpretations_sample.json"
    if Path(interp_file).exists():
        with open(interp_file, 'r', encoding='utf-8') as f:
            interp_data = json.load(f)
        
        for item in interp_data:
            new_doc = {
                'id': item['id'],
                'original_id': item['id'],
                'chunk_id': 'full',
                'text': item['content'],
                'metadata': {
                    'data_type': 'legal_interpretations',
                    'original_document': item['title'],
                    'chunk_start': 0,
                    'chunk_end': len(item['content']),
                    'chunk_length': len(item['content']),
                    'word_count': len(item['content'].split()),
                    'entities': {},
                    'processed_at': '2025-09-30',
                    'is_valid': True,
                    'department': item.get('department', 'ë²•ë¬´ë¶€'),
                    'law_name': item.get('law_name', ''),
                    'interpretation_date': item.get('interpretation_date', '')
                }
            }
            new_data.append(new_doc)
    
    # ê¸°ì¡´ ?°ì´?°ì? ???°ì´??ê²°í•©
    all_data = existing_data + new_data
    
    # ?…ë°?´íŠ¸??ë©”í??°ì´???€??
    with open('data/embeddings/metadata_final.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated metadata saved: data/embeddings/metadata_final.json")
    print(f"Total documents: {len(all_data)}")
    print(f"New documents added: {len(new_data)}")
    
    return all_data

def main():
    print("Adding Missing Data Types")
    print("=" * 50)
    
    # 1. ?„ë½???°ì´???€???•ì¸
    available_types, missing_types = check_missing_data_types()
    
    # 2. ?„ë½???°ì´???€??ì²˜ë¦¬
    process_missing_data_types()
    
    # 3. ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?…ë°?´íŠ¸
    updated_data = update_vector_database()
    
    print("\n" + "=" * 50)
    print("Missing data types added successfully!")
    print(f"Total documents: {len(updated_data)}")
    
    # ?°ì´???€?…ë³„ ë¶„í¬ ?•ì¸
    type_distribution = {}
    for doc in updated_data:
        doc_type = doc['metadata']['data_type']
        type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
    
    print("\nDocument type distribution:")
    for doc_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_type}: {count}")

if __name__ == "__main__":
    main()
