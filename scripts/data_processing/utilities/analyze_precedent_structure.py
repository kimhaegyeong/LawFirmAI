#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?êÎ? ?ÅÏÑ∏ ?òÏù¥ÏßÄ Î™©Ï∞® Íµ¨Ï°∞ Î∂ÑÏÑù
"""

import sys
from pathlib import Path

# ?ÑÎ°ú?ùÌä∏ Î£®Ìä∏Î•?Python Í≤ΩÎ°ú??Ï∂îÍ?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient

def analyze_precedent_structure():
    """?êÎ? ?ÅÏÑ∏ ?òÏù¥ÏßÄ??Î™©Ï∞® Íµ¨Ï°∞ Î∂ÑÏÑù"""
    
    with AssemblyPlaywrightClient(headless=False) as client:
        # ÎØºÏÇ¨ ?êÎ? Ï≤?Î≤àÏß∏ ??™©?ºÎ°ú ?¥Îèô
        print("?îç Analyzing precedent detail page structure...")
        
        # ?êÎ? Î™©Î°ù ?òÏù¥ÏßÄÎ°??¥Îèô
        precedents = client.get_precedent_list_page_by_category("PREC00_001", 1, 10)
        
        if precedents:
            first_precedent = precedents[0]
            print(f"?ìã Analyzing: {first_precedent['case_name']}")
            
            # ?ÅÏÑ∏ ?òÏù¥ÏßÄÎ°??¥Îèô
            url = f"{client.BASE_URL}/law/lawsPrecInqyDetl1010.do"
            url_params = []
            for key, value in first_precedent['params'].items():
                if value:
                    url_params.append(f"{key}={value}")
            
            params_str = "&".join(url_params)
            full_url = f"{url}?{params_str}"
            
            print(f"?åê Navigating to: {full_url}")
            client.page.goto(full_url, wait_until='domcontentloaded')
            client.page.wait_for_timeout(5000)
            
            # ?òÏù¥ÏßÄ Íµ¨Ï°∞ Î∂ÑÏÑù
            print("\n?ìä Page Structure Analysis:")
            
            # 1. ?§Îçî ?ïÎ≥¥
            print("\n1Ô∏è‚É£ Header Information:")
            try:
                h1_elements = client.page.locator("h1").all()
                for i, h1 in enumerate(h1_elements):
                    print(f"   H1-{i+1}: {h1.inner_text().strip()}")
            except Exception as e:
                print(f"   H1 Error: {e}")
            
            # 2. Î™©Ï∞®/?πÏÖò Íµ¨Ï°∞
            print("\n2Ô∏è‚É£ Table of Contents / Sections:")
            try:
                # ?§Ïñë??Î™©Ï∞® ?®ÌÑ¥ ?úÎèÑ
                toc_selectors = [
                    "div.toc", "div.table-of-contents", "div.contents",
                    "ul.toc", "ol.toc", "div.section",
                    "div.content-section", "div.precedent-section"
                ]
                
                for selector in toc_selectors:
                    elements = client.page.locator(selector).all()
                    if elements:
                        print(f"   Found {selector}: {len(elements)} elements")
                        for i, elem in enumerate(elements[:3]):  # Ï≤òÏùå 3Í∞úÎßå
                            text = elem.inner_text().strip()
                            if text:
                                print(f"     {i+1}: {text[:100]}...")
                
                # ?ºÎ∞ò?ÅÏù∏ ?§Îî© ?úÍ∑∏??
                heading_tags = ["h2", "h3", "h4", "h5", "h6"]
                for tag in heading_tags:
                    elements = client.page.locator(tag).all()
                    if elements:
                        print(f"   {tag.upper()} headings: {len(elements)}")
                        for i, elem in enumerate(elements[:5]):  # Ï≤òÏùå 5Í∞úÎßå
                            text = elem.inner_text().strip()
                            if text:
                                print(f"     {i+1}: {text}")
                
            except Exception as e:
                print(f"   TOC Error: {e}")
            
            # 3. ?êÎ? Í¥Ä???πÏ†ï ?πÏÖò??
            print("\n3Ô∏è‚É£ Legal Sections:")
            legal_sections = [
                "?êÏãú?¨Ìï≠", "?êÍ≤∞?îÏ?", "?¨Í±¥", "?êÍ≥†", "?ºÍ≥†", "?êÏã¨", "?ÅÍ≥†", 
                "??Üå", "?åÏÜ°", "Í≥ÑÏïΩ", "?êÌï¥", "Î∞∞ÏÉÅ", "?ÑÏïΩ", "?¥Ï?", "Î¨¥Ìö®", "Ï∑®ÏÜå",
                "Ï£ºÎ¨∏", "?¥Ïú†", "Ï∞∏Ï°∞Ï°∞Î¨∏", "Ï∞∏Ï°∞?êÎ?", "?êÎ?Ïß?, "Î≤ïÏõê", "?†Í≥†??
            ]
            
            for section in legal_sections:
                try:
                    elements = client.page.locator(f"text={section}").all()
                    if elements:
                        print(f"   '{section}': {len(elements)} occurrences")
                        for i, elem in enumerate(elements[:2]):  # Ï≤òÏùå 2Í∞úÎßå
                            parent_text = elem.locator("..").inner_text().strip()
                            if parent_text:
                                print(f"     {i+1}: {parent_text[:100]}...")
                except Exception as e:
                    print(f"   '{section}' Error: {e}")
            
            # 4. ?åÏù¥Î∏?Íµ¨Ï°∞
            print("\n4Ô∏è‚É£ Table Structure:")
            try:
                tables = client.page.locator("table").all()
                print(f"   Tables found: {len(tables)}")
                for i, table in enumerate(tables):
                    rows = table.locator("tr").all()
                    print(f"   Table {i+1}: {len(rows)} rows")
                    if rows:
                        # Ï≤?Î≤àÏß∏ ??(?§Îçî) ?ïÏù∏
                        first_row = rows[0]
                        cells = first_row.locator("td, th").all()
                        headers = [cell.inner_text().strip() for cell in cells]
                        print(f"     Headers: {headers}")
            except Exception as e:
                print(f"   Table Error: {e}")
            
            # 5. ?ÑÏ≤¥ ?çÏä§??Íµ¨Ï°∞ Î∂ÑÏÑù
            print("\n5Ô∏è‚É£ Text Structure Analysis:")
            try:
                full_text = client.page.locator("body").inner_text()
                lines = full_text.split('\n')
                
                print(f"   Total lines: {len(lines)}")
                print(f"   Total characters: {len(full_text)}")
                
                # Îπ?Ï§ÑÏù¥ ?ÑÎãå ?ºÏù∏??
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                print(f"   Non-empty lines: {len(non_empty_lines)}")
                
                # Í∏??ºÏù∏??(50???¥ÏÉÅ)
                long_lines = [line for line in non_empty_lines if len(line) >= 50]
                print(f"   Long lines (50+ chars): {len(long_lines)}")
                
                # Ï≤?10Í∞?Í∏??ºÏù∏ Ï∂úÎ†•
                print("   Sample long lines:")
                for i, line in enumerate(long_lines[:10]):
                    print(f"     {i+1}: {line[:80]}...")
                    
            except Exception as e:
                print(f"   Text Analysis Error: {e}")
            
            # 6. HTML Íµ¨Ï°∞ ?Ä??
            print("\n6Ô∏è‚É£ Saving HTML structure for analysis...")
            try:
                html_content = client.page.content()
                with open("precedent_structure_analysis.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
                print("   HTML saved to: precedent_structure_analysis.html")
            except Exception as e:
                print(f"   HTML Save Error: {e}")

if __name__ == "__main__":
    analyze_precedent_structure()
