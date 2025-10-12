#!/usr/bin/env python3
"""
Test script for Article 4 content parsing issue
"""

import json
import sys
import re
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser

def test_article_4_content_parsing():
    """Test Article 4 content parsing issue."""
    
    # The problematic content from Article 4
    article_4_content = "제4조(포상)\n①대법원장은제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.②포상의 종류와 절차 등은 『법원표창내규』가 정하는 바에 따른다."
    
    print("=== Testing Article 4 Content Parsing ===")
    print(f"Content: {article_4_content}")
    print(f"Content length: {len(article_4_content)}")
    
    # Find paragraph symbols manually
    paragraph_pattern = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]')
    
    for match in paragraph_pattern.finditer(article_4_content):
        print(f"\nFound paragraph symbol at position {match.start()}: '{match.group()}'")
        
        # Extract content from this position
        remaining_text = article_4_content[match.start():]
        print(f"Remaining text: '{remaining_text[:50]}...'")
        
        # Find next paragraph symbol (skip current one)
        next_paragraph_match = paragraph_pattern.search(remaining_text[1:])
        if next_paragraph_match:
            next_pos = match.start() + 1 + next_paragraph_match.start()
            print(f"Next paragraph at position: {next_pos}")
            extracted_content = article_4_content[match.start():next_pos]
            print(f"Extracted content: '{extracted_content}'")
        else:
            extracted_content = article_4_content[match.start():]
            print(f"Extracted content (last): '{extracted_content}'")
    
    # Test with actual parser
    print(f"\n=== Testing with Actual Parser ===")
    parser = ArticleParser()
    parsed_articles = parser.parse_articles(article_4_content, '')
    
    if parsed_articles:
        article = parsed_articles[0]
        sub_articles = article.get('sub_articles', [])
        
        print(f"Found {len(sub_articles)} sub_articles:")
        for i, sub_article in enumerate(sub_articles):
            print(f"  {i+1}. Type: {sub_article.get('type')}, Number: {sub_article.get('number')}")
            print(f"     Content: '{sub_article.get('content', '')}'")
            print(f"     Position: {sub_article.get('position')}")
    
    # Expected results
    print(f"\n=== Expected Results ===")
    print(f"1항 should be: '대법원장은 제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.'")
    print(f"2항 should be: '포상의 종류와 절차 등은 『법원표창내규』가 정하는 바에 따른다.'")

if __name__ == "__main__":
    test_article_4_content_parsing()
