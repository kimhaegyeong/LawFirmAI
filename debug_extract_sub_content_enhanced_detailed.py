#!/usr/bin/env python3
"""
Debug _extract_sub_content_enhanced method with detailed analysis
"""

import json
import sys
import re
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser

def debug_extract_sub_content_enhanced_detailed():
    """Debug _extract_sub_content_enhanced method with detailed analysis."""
    
    # Test content
    article_4_content = "제4조(포상)\n①대법원장은제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.②포상의 종류와 절차 등은 『법원표창내규』가 정하는 바에 따른다."
    
    print("=== Debugging _extract_sub_content_enhanced with Detailed Analysis ===")
    print(f"Content: {article_4_content}")
    
    parser = ArticleParser()
    
    # Find paragraph symbols manually
    paragraph_pattern = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]')
    
    for match in paragraph_pattern.finditer(article_4_content):
        print(f"\nFound paragraph symbol at position {match.start()}: '{match.group()}'")
        
        # Simulate _extract_sub_content_enhanced logic with detailed analysis
        start_pos = match.start()
        remaining_text = article_4_content[start_pos:]
        print(f"Remaining text: '{remaining_text[:50]}...'")
        
        # Find next paragraph symbol (skip current one)
        next_paragraph_match = paragraph_pattern.search(remaining_text[1:])
        if next_paragraph_match:
            next_pos = start_pos + 1 + next_paragraph_match.start()
            print(f"Next paragraph at position: {next_pos}")
            extracted_content = article_4_content[start_pos:next_pos]
            print(f"Raw extracted content: '{extracted_content}'")
            
            # Clean the content
            cleaned_content = parser._clean_legal_content(extracted_content)
            print(f"After _clean_legal_content: '{cleaned_content}'")
            
            # Remove 호(號) items from 항(項) content
            final_content = parser._remove_ho_items_from_hang_content(cleaned_content)
            print(f"After _remove_ho_items_from_hang_content: '{final_content}'")
        else:
            extracted_content = article_4_content[start_pos:]
            print(f"Raw extracted content (last): '{extracted_content}'")
            
            # Clean the content
            cleaned_content = parser._clean_legal_content(extracted_content)
            print(f"After _clean_legal_content: '{cleaned_content}'")
            
            # Remove 호(號) items from 항(項) content
            final_content = parser._remove_ho_items_from_hang_content(cleaned_content)
            print(f"After _remove_ho_items_from_hang_content: '{final_content}'")

if __name__ == "__main__":
    debug_extract_sub_content_enhanced_detailed()