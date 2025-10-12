#!/usr/bin/env python3
"""
Debug _extract_paragraphs_korean method with step-by-step analysis
"""

import json
import sys
import re
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser

def debug_extract_paragraphs_korean_step_by_step():
    """Debug _extract_paragraphs_korean method with step-by-step analysis."""
    
    # Test content
    article_4_content = "제4조(포상)\n①대법원장은제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.②포상의 종류와 절차 등은 『법원표창내규』가 정하는 바에 따른다."
    
    print("=== Debugging _extract_paragraphs_korean Step by Step ===")
    print(f"Content: {article_4_content}")
    
    parser = ArticleParser()
    
    # Manually simulate _extract_paragraphs_korean
    paragraphs = []
    paragraph_pattern = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]')
    
    for match in paragraph_pattern.finditer(article_4_content):
        print(f"\nFound paragraph symbol at position {match.start()}: '{match.group()}'")
        
        # Extract paragraph number
        paragraph_number = parser._extract_paragraph_number_enhanced(match)
        print(f"Paragraph number: {paragraph_number}")
        
        # Extract content using _extract_sub_content_enhanced
        paragraph_content = parser._extract_sub_content_enhanced(article_4_content, match.start())
        print(f"Extracted content: '{paragraph_content}'")
        
        # Validate content
        is_valid = parser._validate_paragraph_content(paragraph_content)
        print(f"Is valid: {is_valid}")
        
        if is_valid:
            paragraph_data = {
                'type': '항',
                'number': paragraph_number,
                'content': paragraph_content,
                'position': match.start()
            }
            paragraphs.append(paragraph_data)
            print(f"Added paragraph: {paragraph_data}")
    
    print(f"\nFinal paragraphs: {len(paragraphs)}")
    for i, para in enumerate(paragraphs):
        print(f"  {i+1}. {para['number']}항: '{para['content']}'")

if __name__ == "__main__":
    debug_extract_paragraphs_korean_step_by_step()