#!/usr/bin/env python3
"""?…ë°?´íŠ¸???Œì¼ ?•ì¸ ?¤í¬ë¦½íŠ¸"""

import json

def main():
    # ?…ë°?´íŠ¸???Œì¼ ë¡œë“œ
    with open('data/raw/assembly/law/20251010/law_page_001_181503.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"?…ë°?´íŠ¸ ?µê³„: {data['update_stats']}")
    print(f"?…ë°?´íŠ¸ ?œê°„: {data['content_updated_at']}")
    print(f"?…ë°?´íŠ¸ ë²„ì „: {data['content_update_version']}")
    print()
    
    print("ê°?ë²•ë ¹ë³??…ë°?´íŠ¸ ê²°ê³¼:")
    for i, law in enumerate(data['laws']):
        original_len = law['original_content_length']
        updated_len = law['updated_content_length']
        ratio = law['content_improvement_ratio']
        print(f"{i+1}. {law['law_name']}: {original_len} -> {updated_len} ë¬¸ì (ê°œì„ ë¹„ìœ¨: {ratio:.2f})")
    
    print()
    print("ì²?ë²ˆì§¸ ë²•ë ¹ ?…ë°?´íŠ¸???´ìš© ë¯¸ë¦¬ë³´ê¸°:")
    first_law = data['laws'][0]
    print(f"ë²•ë ¹ëª? {first_law['law_name']}")
    print(f"?…ë°?´íŠ¸???´ìš© ê¸¸ì´: {len(first_law['law_content'])}")
    print(f"?´ìš© ë¯¸ë¦¬ë³´ê¸°:")
    print(first_law['law_content'][:500] + "...")

if __name__ == "__main__":
    main()



