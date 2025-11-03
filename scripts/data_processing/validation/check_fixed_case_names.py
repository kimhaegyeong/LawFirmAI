#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?˜ì •???ë?ëª?ì¶”ì¶œ ê²°ê³¼ ?•ì¸
"""

import json
import os

def check_fixed_case_names():
    """?˜ì •???ë?ëª?ì¶”ì¶œ ê²°ê³¼ ?•ì¸"""
    
    # ìµœì‹  ?Œì¼ ì°¾ê¸°
    civil_dir = "data/raw/assembly/precedent/20251010/civil"
    files = [f for f in os.listdir(civil_dir) if f.startswith("precedent_civil_page_") and f.endswith(".json")]
    latest_file = sorted(files)[-1]
    
    print(f"?“„ Checking file: {latest_file}")
    
    with open(os.path.join(civil_dir, latest_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"??FIXED CASE NAME EXTRACTION RESULTS")
    print(f"{'='*60}")
    
    for i, precedent in enumerate(data['precedents']):
        print(f"\n?“‹ Precedent {i+1}:")
        print(f"   Case Name: {precedent['case_name']}")
        print(f"   Case Number: {precedent['case_number']}")
        print(f"   Court: {precedent['court']}")
        print(f"   Decision Date: {precedent['decision_date']}")
        print(f"   Field: {precedent['field']}")
        
        # ?ë?ëª…ì´ ?¬ë°”ë¥´ê²Œ ì¶”ì¶œ?˜ì—ˆ?”ì? ?•ì¸
        case_name = precedent['case_name']
        if case_name and case_name != "êµ?šŒë²•ë¥ ?•ë³´?œìŠ¤?? and "National Assembly Law Information" not in case_name:
            print(f"   ??Case name correctly extracted!")
        else:
            print(f"   ??Case name extraction failed!")
    
    print(f"\n{'='*60}")
    print(f"?‰ CASE NAME EXTRACTION VERIFICATION COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    check_fixed_case_names()
