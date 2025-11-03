#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
êµ¬ì¡°?”ëœ ?ë? ?°ì´???•ì¸
"""

import json

def check_structured_precedent():
    """êµ¬ì¡°?”ëœ ?ë? ?°ì´???•ì¸"""
    
    # ìµœì‹  ?Œì¼ ì°¾ê¸°
    import os
    civil_dir = "data/raw/assembly/precedent/20251010/civil"
    files = [f for f in os.listdir(civil_dir) if f.startswith("precedent_civil_page_") and f.endswith(".json")]
    latest_file = sorted(files)[-1]
    
    print(f"?“„ Checking file: {latest_file}")
    
    with open(os.path.join(civil_dir, latest_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"?“Š STRUCTURED PRECEDENT DATA ANALYSIS")
    print(f"{'='*60}")
    
    precedent = data['precedents'][0]
    
    print(f"\n?“‹ Basic Information:")
    print(f"   Case Name: {precedent['case_name']}")
    print(f"   Case Number: {precedent['case_number']}")
    print(f"   Court: {precedent['court']}")
    print(f"   Decision Date: {precedent['decision_date']}")
    print(f"   Field: {precedent['field']}")
    
    print(f"\n?“Š Data Structure:")
    print(f"   Total Keys: {len(precedent.keys())}")
    print(f"   Keys: {list(precedent.keys())}")
    
    structured = precedent['structured_content']
    
    print(f"\n?›ï¸?Case Information:")
    case_info = structured['case_info']
    for key, value in case_info.items():
        print(f"   {key}: {value}")
    
    print(f"\n?“š Legal Sections:")
    legal_sections = structured['legal_sections']
    for key, value in legal_sections.items():
        if value:
            print(f"   ??{key}: {len(value)} chars")
            print(f"      Preview: {value[:100]}...")
        else:
            print(f"   ??{key}: (empty)")
    
    print(f"\n?‘¥ Parties Information:")
    parties = structured['parties']
    for key, value in parties.items():
        if value:
            print(f"   ??{key}: {value}")
        else:
            print(f"   ??{key}: (empty)")
    
    print(f"\n?–ï¸ Procedural Information:")
    procedural = structured['procedural_info']
    for key, value in procedural.items():
        if value:
            print(f"   ??{key}: {value}")
        else:
            print(f"   ??{key}: (empty)")
    
    print(f"\n?“ˆ Extraction Metadata:")
    metadata = structured['extraction_metadata']
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    print(f"\n{'='*60}")
    print(f"??STRUCTURED DATA ANALYSIS COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    check_structured_precedent()
