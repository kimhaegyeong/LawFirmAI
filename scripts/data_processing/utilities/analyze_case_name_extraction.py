#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë?ëª?ì¶”ì¶œ ë¬¸ì œ ë¶„ì„ ë°??˜ì •
"""

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient

def analyze_case_name_extraction():
    """?ë?ëª?ì¶”ì¶œ ë¬¸ì œ ë¶„ì„"""
    
    with AssemblyPlaywrightClient(headless=False) as client:
        print("?” Analyzing case name extraction issue...")
        
        # ë¯¼ì‚¬ ?ë? ì²?ë²ˆì§¸ ??ª©?¼ë¡œ ?´ë™
        precedents = client.get_precedent_list_page_by_category("PREC00_001", 1, 10)
        
        if precedents:
            first_precedent = precedents[0]
            print(f"?“‹ List page case name: {first_precedent['case_name']}")
            
            # ?ì„¸ ?˜ì´ì§€ë¡??´ë™
            url = f"{client.BASE_URL}/law/lawsPrecInqyDetl1010.do"
            url_params = []
            for key, value in first_precedent['params'].items():
                if value:
                    url_params.append(f"{key}={value}")
            
            params_str = "&".join(url_params)
            full_url = f"{url}?{params_str}"
            
            print(f"?Œ Navigating to: {full_url}")
            client.page.goto(full_url, wait_until='domcontentloaded')
            client.page.wait_for_timeout(5000)
            
            # ?˜ì´ì§€?ì„œ ?ë?ëª??„ë³´??ì°¾ê¸°
            print(f"\n?“Š Case Name Candidates Analysis:")
            
            # 1. ëª¨ë“  h1 ?œê·¸ ?•ì¸
            print(f"\n1ï¸âƒ£ H1 Tags:")
            h1_elements = client.page.locator("h1").all()
            for i, h1 in enumerate(h1_elements):
                text = h1.inner_text().strip()
                print(f"   H1-{i+1}: {text}")
            
            # 2. ëª¨ë“  h2 ?œê·¸ ?•ì¸
            print(f"\n2ï¸âƒ£ H2 Tags:")
            h2_elements = client.page.locator("h2").all()
            for i, h2 in enumerate(h2_elements):
                text = h2.inner_text().strip()
                print(f"   H2-{i+1}: {text}")
            
            # 3. ?ë?ëª…ì´ ?¬í•¨?????ˆëŠ” ?¹ì • ?¨í„´ ì°¾ê¸°
            print(f"\n3ï¸âƒ£ Case Name Patterns:")
            
            # ?€ê´„í˜¸ë¡??˜ëŸ¬?¸ì¸ ?ìŠ¤??(?? [ë°°ë‹¹?´ì˜?˜ì†Œ])
            bracket_patterns = client.page.locator("text=/\\[.*?\\]/").all()
            for i, pattern in enumerate(bracket_patterns):
                text = pattern.inner_text().strip()
                print(f"   Bracket-{i+1}: {text}")
            
            # ?ë? ê´€???¤ì›Œ?œê? ?ˆëŠ” ?ìŠ¤??
            case_keywords = ["??, "?¬ê±´", "ì²?µ¬", "?Œì†¡", "?´ì˜", "??³ ", "??†Œ", "?ê³ "]
            for keyword in case_keywords:
                elements = client.page.locator(f"text={keyword}").all()
                if elements:
                    print(f"   Keyword '{keyword}': {len(elements)} occurrences")
                    for i, elem in enumerate(elements[:3]):  # ì²˜ìŒ 3ê°œë§Œ
                        text = elem.inner_text().strip()
                        if len(text) < 100:  # ?ˆë¬´ ê¸??ìŠ¤???œì™¸
                            print(f"     {i+1}: {text}")
            
            # 4. ?˜ì´ì§€ ?œëª© ?•ì¸
            print(f"\n4ï¸âƒ£ Page Title:")
            title = client.page.title()
            print(f"   Title: {title}")
            
            # 5. ?¹ì • ?´ë˜?¤ë‚˜ IDê°€ ?ˆëŠ” ?”ì†Œ???•ì¸
            print(f"\n5ï¸âƒ£ Specific Elements:")
            
            # contents ?´ë˜???•ì¸
            contents_elements = client.page.locator(".contents").all()
            for i, elem in enumerate(contents_elements):
                text = elem.inner_text().strip()
                print(f"   Contents-{i+1}: {text[:200]}...")
            
            # 6. ?„ì²´ ?ìŠ¤?¸ì—???ë?ëª??¨í„´ ì°¾ê¸°
            print(f"\n6ï¸âƒ£ Text Pattern Analysis:")
            full_text = client.page.locator("body").inner_text()
            lines = full_text.split('\n')
            
            # ?€ê´„í˜¸ë¡??˜ëŸ¬?¸ì¸ ?ìŠ¤??ì°¾ê¸°
            import re
            bracket_matches = re.findall(r'\[([^\]]+)\]', full_text)
            if bracket_matches:
                print(f"   Bracket matches: {bracket_matches}")
            
            # ?ë?ëª…ìœ¼ë¡?ë³´ì´??ì§§ì? ?ìŠ¤?¸ë“¤ ì°¾ê¸°
            short_lines = [line.strip() for line in lines if 5 <= len(line.strip()) <= 50 and line.strip()]
            print(f"   Short lines (5-50 chars): {len(short_lines)}")
            for i, line in enumerate(short_lines[:10]):  # ì²˜ìŒ 10ê°œë§Œ
                print(f"     {i+1}: {line}")

if __name__ == "__main__":
    analyze_case_name_extraction()
