#!/usr/bin/env python3
"""
Test the improved parser with original raw data
"""

import json
import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def test_with_original_data():
    """Test parser with original raw data"""
    print("Testing Improved Parser with Original Raw Data")
    print("=" * 60)
    
    # Load original raw data
    raw_file_path = Path("../../data/raw/assembly/law/20251012/law_page_730_205132.json")
    
    if not raw_file_path.exists():
        print(f"Raw file not found: {raw_file_path}")
        return
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Find the 조선재단저당령 law
    target_law = None
    for law in raw_data.get('laws', []):
        if '조선재단저당령' in law.get('law_name', ''):
            target_law = law
            break
    
    if not target_law:
        print("Target law not found")
        return
    
    print(f"Found law: {target_law['law_name']}")
    print(f"Law content length: {len(target_law['law_content'])} characters")
    print()
    
    # Test the improved parser
    parser = ImprovedArticleParser()
    parsed_data = parser.parse_law(target_law['law_content'])
    
    print("=== Parsing Results ===")
    print(f"Parsing status: {parsed_data.get('parsing_status')}")
    print(f"Total articles: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print(f"Supplementary articles: {len(parsed_data.get('supplementary_articles', []))}")
    print()
    
    print("=== Main Articles with Paragraphs ===")
    for i, article in enumerate(parsed_data.get('main_articles', []), 1):
        print(f"{i}. {article.get('article_number')} - {article.get('article_title')}")
        print(f"   Content length: {len(article.get('article_content', ''))} characters")
        print(f"   Sub-articles: {len(article.get('sub_articles', []))}")
        
        # Show sub-articles (paragraphs)
        for j, sub_article in enumerate(article.get('sub_articles', []), 1):
            print(f"   {j}. {sub_article.get('type')} {sub_article.get('number')}: {sub_article.get('content', '')[:50]}...")
        print()
    
    # Validate the results
    is_valid, errors = parser.validate_parsed_data(parsed_data)
    print("=== Validation Results ===")
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    
    print()
    print("=== Original Content Sample ===")
    content_lines = target_law['law_content'].split('\n')
    for line in content_lines[:10]:  # Show first 10 lines
        if line.strip():
            print(f"  {line}")

if __name__ == "__main__":
    test_with_original_data()
